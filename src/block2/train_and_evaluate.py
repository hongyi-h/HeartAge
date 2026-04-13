"""
Step 2 (Block 2): Train all ECG student variants and evaluate.

Runs:
  R20: Full student    — encoder + 12-concept bottleneck + main head, distilled
  R21: Chrono student  — encoder + head → chronological age
  R22: No-BN student   — encoder + direct head → teacher structural_age
  R23: Direct-outcome  — encoder + head → incident HF (binary)

Post-training evaluation (R24):
  - structural_heart_age vs CMR structural_age: Pearson r, MAE
  - structural gap vs chrono gap correlation (Fisher z-test)
  - Differential CMR deviation alignment

Usage:
    python -m src.block2.train_and_evaluate --device cuda [--skip_r23]

Dependencies:
    data/processed/block2_ukb_paired.pt
    data/processed/block2_ptbxl.pt  (for R20 concept supervision)
    results/block1/ (teacher checkpoint + predictions)
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.block2.config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR, PREDICTIONS_DIR,
    TRAIN_CFG, STUDENT_CFG, BLOCK1_DIR,
    STRUCTURAL_CONCEPT_IDX, RHYTHM_CONCEPT_IDX, QUALITY_CONCEPT_IDX,
    DATA_DIR,
)
from src.block2.models import (
    FullStudent, NoBnStudent, ChronoStudent, DirectOutcomeStudent,
    loss_distill, loss_rank_ecg, loss_concept_mse, loss_concept_bce,
    compute_remodeling_burden, compute_perturbation_index,
)

DS_CONFIG_PATH = Path(__file__).parent / "ds_config.json"


def _is_main_rank() -> bool:
    """Return True if this process is rank 0 (or non-distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _local_rank() -> int:
    """Return local rank from environment (set by deepspeed launcher)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def _print_rank0(msg: str):
    """Print only on rank 0."""
    if _is_main_rank():
        print(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_np(t: torch.Tensor) -> np.ndarray:
    try:
        return t.detach().cpu().numpy()
    except RuntimeError:
        return np.array(t.detach().cpu().tolist(), dtype=np.float32)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_amp_context(device: torch.device):
    """Return autocast context for mixed-precision (skip on CPU).

    Note: Only used during inference. DeepSpeed handles AMP during training.
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", enabled=False)


class IndexedDataset(Dataset):
    """Index into existing tensors without copying — saves ~50% RAM."""
    def __init__(self, indices: np.ndarray, *tensors: torch.Tensor):
        self.indices = indices
        self.tensors = tensors

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return tuple(t[real_idx] for t in self.tensors)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ukb_paired(device: torch.device):
    """Load UKB paired ECG + teacher labels."""
    path = PROCESSED_DIR / "block2_ukb_paired.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"UKB paired data not found: {path}\n"
            "Run: python -m src.block2.prepare_data")
    data = torch.load(path, map_location="cpu", weights_only=False)

    is_stub = data.get("is_stub", False)
    if is_stub:
        print("  WARNING: Using stub data — results will not be meaningful.")

    return {
        "waveforms": data["waveforms"],      # (N, 12, 5000) float32
        "structural_age": data["structural_age"],  # (N,)
        "chrono_age": data["chrono_age"],          # (N,)
        "deviation": data["deviation"],            # (N,)
        "domain_scores": data["domain_scores"],    # (N, 5)
        "eids": data["eids"],                      # (N,)
        "split": data["split"],                    # (N,) str array
        "is_stub": is_stub,
    }


def load_ptbxl(device: torch.device):
    """Load PTB-XL concept supervision data."""
    path = PROCESSED_DIR / "block2_ptbxl.pt"
    if not path.exists():
        print("  PTB-XL data not found — R20 concept supervision limited to "
              "structural concepts only.")
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "waveforms": data["waveforms"],          # (N, 12, 5000)
        "rhythm_labels": data["rhythm_labels"],  # (N, 4)
        "quality_labels": data["quality_labels"],  # (N, 3)
        "split": data["split"],                  # (N,) str array
    }


def load_hf_labels(eids: np.ndarray) -> np.ndarray:
    """Load incident HF labels (ICD-10 I50) for R23.

    Checks block1_data.parquet for is_healthy column, then refines to
    HF-specific using hesin_diag if available.
    Returns: (N,) float32 binary array.
    """
    # Try loading from block1_data.parquet (has all participants)
    parquet_path = PROCESSED_DIR / "block1_data.parquet"
    if not parquet_path.exists():
        print("  WARNING: block1_data.parquet not found — skipping R23.")
        return None

    b1 = pd.read_parquet(parquet_path, columns=["eid", "is_healthy"])
    b1_set = set(b1["eid"].values)

    # Try hesin_diag for HF-specific labels
    hesin_path = DATA_DIR / "hesin_diag.rds"
    if hesin_path.exists():
        try:
            import pyreadr
            result = pyreadr.read_r(str(hesin_path))
            df_diag = result[None]
            del result; gc.collect()

            diag_col = "diag_icd10"
            if diag_col not in df_diag.columns:
                icd_cols = [c for c in df_diag.columns if "icd10" in c.lower()]
                diag_col = icd_cols[0] if icd_cols else None

            if diag_col:
                codes = df_diag[diag_col].astype(str).str.strip()
                hf_mask = codes.str.startswith("I50")
                hf_eids = set(df_diag.loc[hf_mask, "eid"].unique())
                del df_diag, codes; gc.collect()

                labels = np.array([1.0 if e in hf_eids else 0.0
                                   for e in eids], dtype=np.float32)
                n_pos = int(labels.sum())
                print(f"  HF labels from hesin_diag: {n_pos}/{len(eids)} "
                      f"positive ({100*n_pos/len(eids):.1f}%)")
                return labels
        except Exception as e:
            print(f"  WARNING: Failed to load hesin_diag: {e}")

    # Fallback: use is_healthy as proxy (inverted — unhealthy ≈ at-risk)
    eid_to_healthy = dict(zip(b1["eid"].values, b1["is_healthy"].values))
    labels = np.array([0.0 if eid_to_healthy.get(e, True) else 1.0
                       for e in eids], dtype=np.float32)
    n_pos = int(labels.sum())
    print(f"  HF labels (proxy from is_healthy): {n_pos}/{len(eids)} "
          f"positive ({100*n_pos/len(eids):.1f}%)")
    return labels


def make_split_indices(split_arr: np.ndarray):
    """Return boolean masks for train/val/test."""
    return {
        "train": split_arr == "train",
        "val": split_arr == "val",
        "test": split_arr == "test",
    }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _make_ds_param_groups(model, cfg):
    """Create parameter groups with lower LR for encoder."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters()
                    if id(p) not in encoder_ids and p.requires_grad]
    return [
        {"params": encoder_params, "lr": cfg["lr"] * cfg["encoder_lr_mult"]},
        {"params": other_params, "lr": cfg["lr"]},
    ]


def train_full_student(ukb, ptbxl, device, cfg, args):
    """R20: Full student with concept bottleneck, distilled from teacher.

    Uses DeepSpeed for distributed training, FP16, and ZeRO Stage 1.
    UKB batches: L_main + L_rank + L_concept_struct
    PTB-XL batches (if available): L_concept_rhythm + L_concept_quality
    """
    _print_rank0("\n" + "=" * 60)
    _print_rank0("R20: Full Student (concept bottleneck)")
    _print_rank0("=" * 60)

    model = FullStudent()
    _print_rank0(f"  Parameters: {_count_params(model):,}")

    param_groups = _make_ds_param_groups(model, cfg)

    # DeepSpeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=param_groups,
        config=str(DS_CONFIG_PATH),
    )
    device = model_engine.local_rank
    device = torch.device("cuda", device) if torch.cuda.is_available() \
        else torch.device("cpu")

    # UKB data
    splits = make_split_indices(ukb["split"])
    tr_idx = np.where(splits["train"])[0]
    va_idx = np.where(splits["val"])[0]

    train_dataset = IndexedDataset(
        tr_idx,
        ukb["waveforms"],
        ukb["structural_age"],
        ukb["chrono_age"],
        ukb["domain_scores"],
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True, seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=0, pin_memory=True,
    )

    # Validation tensors (keep small — only used for loss computation)
    val_sample = min(len(va_idx), 512)
    val_sub = va_idx[:val_sample]
    val_ecg = ukb["waveforms"][val_sub].to(device)
    val_sa = ukb["structural_age"][val_sub].to(device)
    val_ca = ukb["chrono_age"][val_sub].to(device)
    val_ds = ukb["domain_scores"][val_sub].to(device)

    # PTB-XL iterator (concept supervision for rhythm/quality)
    ptbxl_iter = None
    ptbxl_loader = None
    if ptbxl is not None:
        pt_tr_idx = np.where(ptbxl["split"] == "train")[0]
        ptbxl_loader = DataLoader(
            IndexedDataset(
                pt_tr_idx,
                ptbxl["waveforms"],
                ptbxl["rhythm_labels"],
                ptbxl["quality_labels"],
            ),
            batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
        )
        ptbxl_iter = iter(ptbxl_loader)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ecg_b, sa_b, ca_b, ds_b = [x.to(device) for x in batch]

            # DeepSpeed handles FP16 autocast internally
            emb, concepts, pred_sa = model_engine(ecg_b)

            loss = (cfg["w_main"] * loss_distill(pred_sa, sa_b)
                    + cfg["w_rank"] * loss_rank_ecg(pred_sa, ca_b)
                    + cfg["w_concept_struct"] * loss_concept_mse(
                        concepts, ds_b, STRUCTURAL_CONCEPT_IDX))

            # PTB-XL concept supervision (interleaved)
            if ptbxl_iter is not None:
                try:
                    pt_ecg, pt_rhy, pt_qual = next(ptbxl_iter)
                except StopIteration:
                    ptbxl_iter = iter(ptbxl_loader)
                    pt_ecg, pt_rhy, pt_qual = next(ptbxl_iter)
                pt_ecg = pt_ecg.to(device)
                pt_rhy = pt_rhy.to(device)
                pt_qual = pt_qual.to(device)

                pt_emb, pt_concepts, _ = model_engine(pt_ecg)
                loss = loss + (
                    cfg["w_concept_rhythm"] * loss_concept_bce(
                        pt_concepts, pt_rhy, RHYTHM_CONCEPT_IDX)
                    + cfg["w_concept_quality"] * loss_concept_mse(
                        pt_concepts, pt_qual, QUALITY_CONCEPT_IDX))

            # DeepSpeed backward + step (handles loss scaling, grad clip)
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation (run on all ranks, compare on rank 0)
        model_engine.eval()
        with torch.no_grad(), _get_amp_context(device):
            _, v_concepts, v_sa = model_engine.module(val_ecg)
            vl = (cfg["w_main"] * loss_distill(v_sa, val_sa)
                  + cfg["w_rank"] * loss_rank_ecg(v_sa, val_ca)
                  + cfg["w_concept_struct"] * loss_concept_mse(
                      v_concepts, val_ds, STRUCTURAL_CONCEPT_IDX)).item()
        history["val_loss"].append(vl)

        if epoch < 5 or (epoch + 1) % 10 == 0:
            _print_rank0(f"  Epoch {epoch+1:3d}  train={avg_train:.4f}  "
                         f"val={vl:.4f}")

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_cnt = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model_engine.module.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                _print_rank0(f"  Early stop at epoch {epoch+1}")
                break

    # Restore best weights into the raw model (not the engine)
    raw_model = model_engine.module
    if best_state:
        raw_model.load_state_dict(best_state)
    raw_model = raw_model.to(device)
    raw_model.eval()
    del val_ecg, val_sa, val_ca, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw_model, history


def train_simple_student(model, ukb, device, cfg, target_key,
                         loss_fn, tag, is_classification=False, args=None):
    """Generic training for R21, R22, R23 (no concept supervision).

    Uses DeepSpeed for distributed training, FP16, and ZeRO Stage 1.

    Args:
        target_key: key in ukb dict for target labels
        loss_fn: callable(pred, target) → loss
        is_classification: use BCE instead of MSE
        args: argparse namespace (needed by DeepSpeed)
    """
    _print_rank0(f"\n{'=' * 60}")
    _print_rank0(f"{tag}")
    _print_rank0("=" * 60)
    _print_rank0(f"  Parameters: {_count_params(model):,}")

    param_groups = _make_ds_param_groups(model, cfg)

    # DeepSpeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=param_groups,
        config=str(DS_CONFIG_PATH),
    )
    device = torch.device("cuda", model_engine.local_rank) \
        if torch.cuda.is_available() else torch.device("cpu")

    splits = make_split_indices(ukb["split"])
    tr_idx = np.where(splits["train"])[0]
    va_idx = np.where(splits["val"])[0]

    targets = ukb[target_key]
    chrono_age = ukb["chrono_age"]

    train_dataset = IndexedDataset(tr_idx, ukb["waveforms"], targets,
                                   chrono_age)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True, seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        num_workers=0, pin_memory=True,
    )

    val_sample = min(len(va_idx), 512)
    val_sub = va_idx[:val_sample]
    val_ecg = ukb["waveforms"][val_sub].to(device)
    val_tgt = targets[val_sub].to(device)
    val_ca = chrono_age[val_sub].to(device)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        epoch_loss = 0.0
        n_batches = 0

        for ecg_b, tgt_b, ca_b in train_loader:
            ecg_b = ecg_b.to(device)
            tgt_b = tgt_b.to(device)
            ca_b = ca_b.to(device)

            _, pred = model_engine(ecg_b)
            loss = loss_fn(pred, tgt_b)
            if not is_classification:
                loss = loss + cfg["w_rank"] * loss_rank_ecg(pred, ca_b)

            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        model_engine.eval()
        with torch.no_grad(), _get_amp_context(device):
            _, v_pred = model_engine.module(val_ecg)
            vl = loss_fn(v_pred, val_tgt).item()
        history["val_loss"].append(vl)

        if epoch < 5 or (epoch + 1) % 10 == 0:
            _print_rank0(f"  Epoch {epoch+1:3d}  train={avg_train:.4f}  val={vl:.4f}")

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_cnt = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model_engine.module.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                _print_rank0(f"  Early stop at epoch {epoch+1}")
                break

    raw_model = model_engine.module
    if best_state:
        raw_model.load_state_dict(best_state)
    raw_model = raw_model.to(device)
    raw_model.eval()
    del val_ecg, val_tgt, val_ca
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw_model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_full_student(model: FullStudent, waveforms: torch.Tensor,
                         device: torch.device,
                         batch_size: int = 64) -> dict:
    """Run full student inference in batches, return numpy arrays."""
    model.eval()
    all_emb, all_concepts, all_sa = [], [], []
    n = len(waveforms)

    for i in range(0, n, batch_size):
        ecg_b = waveforms[i:i+batch_size].to(device)
        with _get_amp_context(device):
            emb, concepts, sa = model(ecg_b)
        all_emb.append(_to_np(emb))
        all_concepts.append(_to_np(concepts))
        all_sa.append(_to_np(sa))

    return {
        "embedding": np.concatenate(all_emb, axis=0),
        "concepts": np.concatenate(all_concepts, axis=0),
        "structural_age": np.concatenate(all_sa, axis=0),
    }


@torch.no_grad()
def predict_simple(model, waveforms: torch.Tensor,
                   device: torch.device,
                   batch_size: int = 64) -> np.ndarray:
    """Run simple student inference, return predictions as numpy."""
    model.eval()
    all_pred = []
    for i in range(0, len(waveforms), batch_size):
        ecg_b = waveforms[i:i+batch_size].to(device)
        with _get_amp_context(device):
            _, pred = model(ecg_b)
        all_pred.append(_to_np(pred))
    return np.concatenate(all_pred, axis=0)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def fisher_z_test(r1: float, r2: float, n: int, r12: float = 0.0) -> dict:
    """Steiger's test for comparing two dependent correlations.

    Tests H0: r1 == r2, where both correlations share one variable.
    r12 is the correlation between the two predictor variables.
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    # Steiger (1980) formula for dependent correlations
    r_m = (r1 + r2) / 2
    denom = (1 - r12) * (1 + r12)
    if denom <= 0:
        denom = 1e-8
    f = (1 - r12**2 - r1**2 - r2**2 + 2 * r12 * r1 * r2) / denom
    if f <= 0:
        f = 1e-8
    # Asymptotic z-statistic
    z_diff = (z1 - z2) * math.sqrt((n - 3) / (2 * (1 - r12) + f))
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_diff)))
    return {
        "z1": float(z1), "z2": float(z2),
        "z_diff": float(z_diff), "p_value": float(p_value),
        "r1": float(r1), "r2": float(r2), "r12": float(r12), "n": n,
    }


def evaluate_all(ukb, predictions, hf_labels=None):
    """Compute all Block 2 metrics (R20–R24).

    Args:
        ukb: loaded UKB paired data dict
        predictions: dict of model_name → numpy predictions
        hf_labels: optional (N,) HF binary array for R23
    Returns:
        dict of results for JSON serialization
    """
    splits = make_split_indices(ukb["split"])
    te = splits["test"]
    test_idx = np.where(te)[0]

    sa_teacher = _to_np(ukb["structural_age"][test_idx])
    ca_true = _to_np(ukb["chrono_age"][test_idx])
    dev_teacher = _to_np(ukb["deviation"][test_idx])
    n_test = len(test_idx)

    results = {}

    # --- R20: Full student ---
    if "full_student" in predictions:
        ps = predictions["full_student"]
        sa_pred = ps["structural_age"][test_idx]
        r_sa, p_sa = sp_stats.pearsonr(sa_pred, sa_teacher)
        mae_sa = float(np.mean(np.abs(sa_pred - sa_teacher)))
        rmse_sa = float(np.sqrt(np.mean((sa_pred - sa_teacher) ** 2)))

        # structural gap = struct_age - chrono_age
        struct_gap = sa_pred - ca_true

        # Correlation of structural gap with teacher deviation
        r_gap_dev, p_gap_dev = sp_stats.pearsonr(struct_gap, dev_teacher)

        results["R20_full_student"] = {
            "structural_age_vs_teacher": {
                "pearson_r": float(r_sa), "p_value": float(p_sa),
                "mae": mae_sa, "rmse": rmse_sa,
            },
            "structural_gap_vs_deviation": {
                "pearson_r": float(r_gap_dev), "p_value": float(p_gap_dev),
            },
            "n_test": n_test,
        }

    # --- R21: Chrono student ---
    if "chrono_student" in predictions:
        ca_pred = predictions["chrono_student"][test_idx]
        r_ca, p_ca = sp_stats.pearsonr(ca_pred, ca_true)
        mae_ca = float(np.mean(np.abs(ca_pred - ca_true)))

        # chrono gap = pred_chrono_age - true_chrono_age
        chrono_gap = ca_pred - ca_true

        # Correlation of chrono gap with teacher deviation
        r_cgap_dev, p_cgap_dev = sp_stats.pearsonr(chrono_gap, dev_teacher)

        results["R21_chrono_student"] = {
            "chrono_age_prediction": {
                "pearson_r": float(r_ca), "p_value": float(p_ca),
                "mae": mae_ca,
            },
            "chrono_gap_vs_deviation": {
                "pearson_r": float(r_cgap_dev), "p_value": float(p_cgap_dev),
            },
            "n_test": n_test,
        }

    # --- R22: No-bottleneck student ---
    if "nobn_student" in predictions:
        sa_nobn = predictions["nobn_student"][test_idx]
        r_nobn, p_nobn = sp_stats.pearsonr(sa_nobn, sa_teacher)
        mae_nobn = float(np.mean(np.abs(sa_nobn - sa_teacher)))

        results["R22_nobn_student"] = {
            "structural_age_vs_teacher": {
                "pearson_r": float(r_nobn), "p_value": float(p_nobn),
                "mae": mae_nobn,
            },
            "n_test": n_test,
        }
        # Accuracy drop vs full student
        if "full_student" in predictions:
            sa_full = predictions["full_student"]["structural_age"][test_idx]
            mae_full = float(np.mean(np.abs(sa_full - sa_teacher)))
            results["R22_nobn_student"]["accuracy_drop_vs_full"] = {
                "mae_full": mae_full,
                "mae_nobn": mae_nobn,
                "drop_pct": float((mae_nobn - mae_full) / mae_full * 100),
                "note": ">5% drop → consider widening bottleneck to 16",
            }

    # --- R23: Direct-outcome student ---
    if "direct_outcome" in predictions and hf_labels is not None:
        logits = predictions["direct_outcome"][test_idx]
        y_true = hf_labels[test_idx]
        # Compute AUROC if there are both positive and negative cases
        n_pos = int(y_true.sum())
        if n_pos > 0 and n_pos < len(y_true):
            from sklearn.metrics import roc_auc_score
            probs = 1.0 / (1.0 + np.exp(-logits))
            auroc = float(roc_auc_score(y_true, probs))
        else:
            auroc = float("nan")
        results["R23_direct_outcome"] = {
            "hf_auroc": auroc,
            "n_positive": n_pos,
            "n_total": int(len(y_true)),
            "note": "Comparator — should NOT outperform structural gap",
        }

    # --- R24: Cross-analysis ---
    if "full_student" in predictions and "chrono_student" in predictions:
        sa_full = predictions["full_student"]["structural_age"][test_idx]
        ca_pred = predictions["chrono_student"][test_idx]

        struct_gap = sa_full - ca_true
        chrono_gap = ca_pred - ca_true

        # (1) Correlation between the two gaps
        r_gaps, p_gaps = sp_stats.pearsonr(struct_gap, chrono_gap)

        # (2) Differential correlation with CMR deviation
        r_struct_dev, _ = sp_stats.pearsonr(struct_gap, dev_teacher)
        r_chrono_dev, _ = sp_stats.pearsonr(chrono_gap, dev_teacher)
        r_between, _ = sp_stats.pearsonr(struct_gap, chrono_gap)

        fisher = fisher_z_test(r_struct_dev, r_chrono_dev, n_test,
                               r12=r_between)

        results["R24_cross_analysis"] = {
            "gap_correlation": {
                "pearson_r": float(r_gaps), "p_value": float(p_gaps),
                "criterion": "expect < 0.85",
                "pass": bool(abs(r_gaps) < 0.85),
            },
            "differential_cmr_alignment": {
                "struct_gap_vs_deviation_r": float(r_struct_dev),
                "chrono_gap_vs_deviation_r": float(r_chrono_dev),
                "fisher_z_test": fisher,
                "criterion": "structural > chrono (p < 0.05)",
                "pass": bool(fisher["p_value"] < 0.05
                             and r_struct_dev > r_chrono_dev),
            },
        }

        # Overall Block 2 success criteria
        results["block2_success"] = {
            "C1_gap_correlation_lt_085": bool(abs(r_gaps) < 0.85),
            "C2_structural_better_alignment": bool(
                fisher["p_value"] < 0.05 and r_struct_dev > r_chrono_dev),
        }

    return results


# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------

def save_predictions(ukb, predictions, hf_labels=None):
    """Save all predictions to CSV for downstream blocks."""
    eids = ukb["eids"]
    splits = ukb["split"]
    ca = _to_np(ukb["chrono_age"])

    # Full student
    if "full_student" in predictions:
        ps = predictions["full_student"]
        df = pd.DataFrame({
            "eid": eids,
            "split": splits,
            "chrono_age": ca,
            "structural_heart_age": ps["structural_age"],
            "structural_gap": ps["structural_age"] - ca,
        })
        # Add concept columns
        for i, name in enumerate([
            "LV_dev", "RV_dev", "Atrial_dev", "Aortic_dev", "Mechanics_dev",
            "sinus_conf", "QRS_norm", "ST_dev", "P_wave_qual",
            "SNR", "baseline_stab", "lead_complete",
        ]):
            df[f"concept_{name}"] = ps["concepts"][:, i]

        # Add derived scores
        concepts_t = torch.from_numpy(ps["concepts"])
        df["remodeling_burden"] = _to_np(
            compute_remodeling_burden(concepts_t))
        df["perturbation_index"] = _to_np(
            compute_perturbation_index(concepts_t))

        df.to_csv(PREDICTIONS_DIR / "full_student_pred.csv", index=False)
        print(f"  Saved full_student_pred.csv ({len(df)} rows)")

    # Chrono student
    if "chrono_student" in predictions:
        df = pd.DataFrame({
            "eid": eids,
            "split": splits,
            "chrono_age": ca,
            "pred_chrono_age": predictions["chrono_student"],
            "chrono_gap": predictions["chrono_student"] - ca,
        })
        df.to_csv(PREDICTIONS_DIR / "chrono_student_pred.csv", index=False)
        print(f"  Saved chrono_student_pred.csv ({len(df)} rows)")

    # No-BN student
    if "nobn_student" in predictions:
        df = pd.DataFrame({
            "eid": eids,
            "split": splits,
            "chrono_age": ca,
            "structural_age_nobn": predictions["nobn_student"],
        })
        df.to_csv(PREDICTIONS_DIR / "nobn_student_pred.csv", index=False)
        print(f"  Saved nobn_student_pred.csv ({len(df)} rows)")

    # Direct-outcome student
    if "direct_outcome" in predictions:
        df = pd.DataFrame({
            "eid": eids,
            "split": splits,
            "hf_logit": predictions["direct_outcome"],
            "hf_prob": 1.0 / (1.0 + np.exp(-predictions["direct_outcome"])),
        })
        if hf_labels is not None:
            df["hf_label"] = hf_labels
        df.to_csv(PREDICTIONS_DIR / "direct_outcome_pred.csv", index=False)
        print(f"  Saved direct_outcome_pred.csv ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 2: ECG Student Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_r23", action="store_true",
                        help="Skip R23 (direct-outcome student)")
    parser.add_argument("--skip_ptbxl", action="store_true",
                        help="Skip PTB-XL concept supervision for R20")
    # DeepSpeed adds its own args (--local_rank, etc.)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # DeepSpeed sets LOCAL_RANK via environment; determine device from it
    local_rank = _local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # DeepSpeed handles dist.init_process_group internally via
    # deepspeed.initialize(), but we can also init here for early use
    if not dist.is_initialized() and torch.cuda.is_available():
        deepspeed.init_distributed()

    for d in [MODELS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = TRAIN_CFG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- Load data ----
    _print_rank0("Loading UKB paired data …")
    ukb = load_ukb_paired(device)
    _print_rank0(f"  Waveforms: {ukb['waveforms'].shape}")
    _print_rank0(f"  Splits: { {k: int(v.sum()) for k, v in make_split_indices(ukb['split']).items()} }")

    ptbxl = None
    if not args.skip_ptbxl:
        _print_rank0("Loading PTB-XL data …")
        ptbxl = load_ptbxl(device)
        if ptbxl is not None:
            _print_rank0(f"  PTB-XL waveforms: {ptbxl['waveforms'].shape}")

    # HF labels for R23
    hf_labels = None
    if not args.skip_r23:
        _print_rank0("Loading HF labels …")
        hf_labels = load_hf_labels(ukb["eids"])

    predictions = {}
    all_histories = {}

    # ================================================================
    # R20: Full Student
    # ================================================================
    model_r20, hist_r20 = train_full_student(ukb, ptbxl, device, cfg, args)
    if _is_main_rank():
        torch.save(model_r20.state_dict(), MODELS_DIR / "full_student.pt")
    all_histories["R20_full_student"] = hist_r20

    # Inference on all data
    preds_full = predict_full_student(model_r20, ukb["waveforms"], device,
                                      batch_size=cfg["batch_size"])
    predictions["full_student"] = preds_full
    del model_r20; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # R21: Chrono-age Student
    # ================================================================
    model_r21 = ChronoStudent()
    model_r21, hist_r21 = train_simple_student(
        model_r21, ukb, device, cfg,
        target_key="chrono_age",
        loss_fn=loss_distill,
        tag="R21: Chrono-age Student",
        args=args,
    )
    if _is_main_rank():
        torch.save(model_r21.state_dict(), MODELS_DIR / "chrono_student.pt")
    all_histories["R21_chrono_student"] = hist_r21

    preds_chrono = predict_simple(model_r21, ukb["waveforms"], device,
                                  batch_size=cfg["batch_size"])
    predictions["chrono_student"] = preds_chrono
    del model_r21; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # R22: No-Bottleneck Student
    # ================================================================
    model_r22 = NoBnStudent()
    model_r22, hist_r22 = train_simple_student(
        model_r22, ukb, device, cfg,
        target_key="structural_age",
        loss_fn=loss_distill,
        tag="R22: No-Bottleneck Student",
        args=args,
    )
    if _is_main_rank():
        torch.save(model_r22.state_dict(), MODELS_DIR / "nobn_student.pt")
    all_histories["R22_nobn_student"] = hist_r22

    preds_nobn = predict_simple(model_r22, ukb["waveforms"], device,
                                batch_size=cfg["batch_size"])
    predictions["nobn_student"] = preds_nobn
    del model_r22; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ================================================================
    # R23: Direct-Outcome Student (HF)
    # ================================================================
    if not args.skip_r23 and hf_labels is not None:
        # Add HF labels to ukb dict temporarily
        ukb["hf_label"] = torch.from_numpy(hf_labels)

        model_r23 = DirectOutcomeStudent()
        model_r23, hist_r23 = train_simple_student(
            model_r23, ukb, device, cfg,
            target_key="hf_label",
            loss_fn=F.binary_cross_entropy_with_logits,
            tag="R23: Direct-Outcome Student (HF)",
            is_classification=True,
            args=args,
        )
        if _is_main_rank():
            torch.save(model_r23.state_dict(),
                        MODELS_DIR / "direct_outcome.pt")
        all_histories["R23_direct_outcome"] = hist_r23

        preds_hf = predict_simple(model_r23, ukb["waveforms"], device,
                                  batch_size=cfg["batch_size"])
        predictions["direct_outcome"] = preds_hf
        del model_r23; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        _print_rank0("\nR23 skipped.")

    # ================================================================
    # Evaluate all (R24 included)
    # ================================================================
    _print_rank0("\n" + "=" * 60)
    _print_rank0("Evaluation (R20–R24)")
    _print_rank0("=" * 60)

    results = evaluate_all(ukb, predictions, hf_labels)

    # Print summary (rank 0 only)
    if _is_main_rank():
        if "R24_cross_analysis" in results:
            r24 = results["R24_cross_analysis"]
            print(f"\n  Gap correlation: r = "
                  f"{r24['gap_correlation']['pearson_r']:.3f} "
                  f"(criterion: < 0.85, pass: "
                  f"{r24['gap_correlation']['pass']})")
            print(f"  Struct gap ↔ deviation: r = "
                  f"{r24['differential_cmr_alignment']['struct_gap_vs_deviation_r']:.3f}")
            print(f"  Chrono gap ↔ deviation: r = "
                  f"{r24['differential_cmr_alignment']['chrono_gap_vs_deviation_r']:.3f}")
            fz = r24["differential_cmr_alignment"]["fisher_z_test"]
            print(f"  Fisher z-test: z={fz['z_diff']:.3f}, p={fz['p_value']:.4f}")

        if "block2_success" in results:
            s = results["block2_success"]
            print(f"\n  Block 2 success criteria:")
            print(f"    C1 gap corr < 0.85:       {s['C1_gap_correlation_lt_085']}")
            print(f"    C2 struct > chrono align:  {s['C2_structural_better_alignment']}")

    # ================================================================
    # Save (rank 0 only)
    # ================================================================
    if _is_main_rank():
        print("\nSaving predictions …")
        save_predictions(ukb, predictions, hf_labels)

        print("Saving results …")
        results_path = RESULTS_DIR / "block2_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results → {results_path}")

        history_path = RESULTS_DIR / "training_curves.json"
        with open(history_path, "w") as f:
            json.dump(all_histories, f, indent=2)
        print(f"  Training curves → {history_path}")

        print("\nDone. Key file: results/block2/block2_results.json")


if __name__ == "__main__":
    main()
