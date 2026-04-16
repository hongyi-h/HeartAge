"""SparK1D pretraining for ECG ResNet1D encoder.

Uses SparK (Sparse and Hierarchical Masked Modeling, ICLR 2023) adapted
for 1D convolutional networks.  Masked positions are zeroed after every
sparse-conv layer so the CNN encoder never sees masked content — solving
the train/test distribution mismatch inherent in zero-masking MAE for CNNs.

After pretraining the decoder is discarded; only the encoder checkpoint
is kept as initialisation for Block 2 student models.

Usage:
  deepspeed --num_gpus=N --module src.block2.pretrain_mae \\
      --deepspeed_config src/block2/ds_config_pretrain.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import wandb

from src.block2.config import (
    PROJECT_ROOT, ENCODER_CFG, ECG_N_LEADS,
)
from src.block2.models import ResNet1DEncoder
from src.block2.pretrain_data import build_pretrain_dataset

from SparK1D.pretrain.encoder import SparseEncoder1D
from SparK1D.pretrain.decoder import LightDecoder1D
from SparK1D.pretrain.spark import SparK1D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRETRAIN_DIR = PROJECT_ROOT / "results" / "block2" / "pretrain"
PRETRAIN_CKPT = PRETRAIN_DIR / "encoder_pretrained.pt"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAE_CFG = {
    "mask_ratio": 0.60,         # SparK/ConvNeXt-V2 optimal for CNN
    "decoder_width": 512,       # matches encoder last-stage channels
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 10,
    "val_ratio": 0.02,
    "seed": 42,
    # Truncate 5000→4992 so that L / downsample_ratio(32) is integer.
    # 8 samples = 16 ms — clinically negligible.
    "input_size": 4992,
}


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _print_rank0(msg: str):
    if _is_main_rank():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_spark_model(cfg: dict):
    """Create SparK1D model: ResNet1DEncoder → SparseEncoder → Decoder → SparK."""
    cnn = ResNet1DEncoder()
    sparse_encoder = SparseEncoder1D(
        cnn, input_size=cfg["input_size"], sbn=False,
    )
    decoder = LightDecoder1D(
        up_sample_ratio=sparse_encoder.downsample_ratio,
        width=cfg["decoder_width"],
        out_channels=ECG_N_LEADS,
        sbn=False,
    )
    model = SparK1D(
        sparse_encoder=sparse_encoder,
        dense_decoder=decoder,
        mask_ratio=cfg["mask_ratio"],
        input_size=cfg["input_size"],
        densify_norm='bn',
        sbn=False,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mae(args):
    """Main SparK1D pretraining loop."""
    cfg = MAE_CFG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Init distributed
    deepspeed.init_distributed()
    local_rank = _local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() \
        else torch.device("cpu")

    # wandb
    if _is_main_rank():
        wandb.init(
            project="heartage-ecg-pretrain",
            config={**cfg, **dict(ENCODER_CFG)},
            name=f"spark1d-{time.strftime('%m%d-%H%M')}",
        )

    # Build dataset
    _print_rank0("Loading pretrain datasets …")
    full_dataset = build_pretrain_dataset()
    total = len(full_dataset)

    # Train / val split
    n_val = max(1, int(total * cfg["val_ratio"]))
    n_train = total - n_val
    rng = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=rng,
    )
    _print_rank0(f"  Train: {n_train}, Val: {n_val}")

    # Data loaders
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        sampler=train_sampler, num_workers=8, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
        drop_last=False,
    )

    # Model
    model = build_spark_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.sparse_encoder.parameters())
    _print_rank0(f"  SparK1D params: {total_params:,}  "
                 f"(encoder: {enc_params:,}, "
                 f"decoder+densify: {total_params - enc_params:,})")
    _print_rank0(f"  Mask ratio: {cfg['mask_ratio']}, "
                 f"Input size: {cfg['input_size']}, "
                 f"fmap_len: {model.fmap_len}, "
                 f"Decoder width: {cfg['decoder_width']}")

    # DeepSpeed init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # Training state
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_encoder_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    global_step = 0
    input_size = cfg["input_size"]

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        # --- Train ---
        model_engine.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ecg = batch.to(device).float()  # float16 cache → float32
            # Scrub NaN/Inf from corrupt records
            ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            # Truncate to input_size (5000 → 4992)
            ecg = ecg[:, :, :input_size]

            loss = model_engine(ecg)

            # Guard against NaN loss (prevents poisoning optimizer state)
            if torch.isnan(loss) or torch.isinf(loss):
                if _is_main_rank() and n_batches == 0:
                    print("WARNING: NaN/Inf loss, skipping batch", flush=True)
                continue

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            if _is_main_rank() and global_step % 50 == 0:
                wandb.log({"train/batch_loss": loss.item()}, step=global_step)

        avg_train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model_engine.eval()
        val_loss_sum = 0.0
        val_n = 0
        raw_model = model_engine.module

        with torch.no_grad():
            for batch in val_loader:
                ecg = batch.to(device).float()
                ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
                ecg = ecg[:, :, :input_size]

                loss = raw_model(ecg)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                val_loss_sum += loss.item() * ecg.shape[0]
                val_n += ecg.shape[0]

        avg_val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        _print_rank0(
            f"Epoch {epoch+1:3d}/{cfg['max_epochs']} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        if _is_main_rank():
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/epoch_loss": avg_val_loss,
                "lr": current_lr,
                "epoch": epoch + 1,
                "epoch_time_s": elapsed,
            }, step=global_step)

        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if _is_main_rank():
                best_encoder_state = {
                    k: v.cpu().clone()
                    for k, v in raw_model.get_encoder_state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                _print_rank0(f"Early stopping at epoch {epoch+1}")
                break

    # --- Save ---
    if _is_main_rank() and best_encoder_state is not None:
        torch.save({
            "encoder_state_dict": best_encoder_state,
            "config": ENCODER_CFG,
            "mae_config": cfg,
            "best_val_loss": best_val_loss,
            "epoch": epoch + 1,
            "method": "SparK1D",
            "datasets_used": "PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG",
        }, PRETRAIN_CKPT)
        _print_rank0(f"\nEncoder saved to {PRETRAIN_CKPT}")
        _print_rank0(f"Best val loss: {best_val_loss:.6f}")

        with open(PRETRAIN_DIR / "pretrain_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Cleanup
    if _is_main_rank():
        wandb.finish()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SparK1D ECG Pretraining for ResNet1D Encoder")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank (set by DeepSpeed launcher)")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train_mae(args)


if __name__ == "__main__":
    main()
