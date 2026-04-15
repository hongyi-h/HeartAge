"""Masked Autoencoder (MAE) pretraining for ECG ResNet1D encoder.

Strategy:
  1. Randomly mask contiguous segments of the input ECG signal
  2. Encoder processes the partially-masked input
  3. Lightweight decoder reconstructs the masked regions
  4. Loss = MSE on masked positions only

After pretraining, the decoder is discarded and the encoder is used
as initialisation for Block 2 student models.

Usage:
  deepspeed --num_gpus=N --module src.block2.pretrain_mae \\
      --deepspeed_config src/block2/ds_config_pretrain.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import wandb

from src.block2.config import (
    PROJECT_ROOT, ENCODER_CFG, ECG_N_LEADS, ECG_SEQ_LEN, ECG_SAMPLE_RATE,
)
from src.block2.models import ResNet1DEncoder
from src.block2.pretrain_data import build_pretrain_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRETRAIN_DIR = PROJECT_ROOT / "results" / "block2" / "pretrain"
PRETRAIN_CKPT = PRETRAIN_DIR / "encoder_pretrained.pt"

# ---------------------------------------------------------------------------
# MAE config
# ---------------------------------------------------------------------------
MAE_CFG = {
    "mask_ratio": 0.40,        # fraction of time steps to mask
    "mask_segment_len": 250,   # each mask segment is 250 samples (0.5s at 500Hz)
    "decoder_dim": 128,        # lightweight decoder hidden dim
    "decoder_layers": 2,       # number of ConvTranspose layers in decoder
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 10,
    "val_ratio": 0.02,         # 2% held out for validation
    "seed": 42,
    "warmup_epochs": 3,
}


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def generate_mask(seq_len: int, mask_ratio: float, segment_len: int,
                  batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate binary mask: 1 = keep, 0 = masked.

    Returns: (B, 1, T) broadcastable over leads.
    """
    n_segments = seq_len // segment_len
    n_mask = max(1, int(n_segments * mask_ratio))

    mask = torch.ones(batch_size, 1, seq_len, device=device)
    for i in range(batch_size):
        # Pick random segments to mask
        perm = torch.randperm(n_segments, device=device)[:n_mask]
        for seg_idx in perm:
            start = seg_idx * segment_len
            end = start + segment_len
            mask[i, :, start:end] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Decoder (lightweight — discarded after pretraining)
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """Reconstruct masked ECG from encoder embedding.

    Architecture: project embedding → upsample via ConvTranspose1d → predict
    all 12 leads at full temporal resolution.
    """

    def __init__(self, embedding_dim: int = None,
                 decoder_dim: int = None,
                 n_leads: int = ECG_N_LEADS,
                 seq_len: int = ECG_SEQ_LEN):
        super().__init__()
        embedding_dim = embedding_dim or ENCODER_CFG["embedding_dim"]
        decoder_dim = decoder_dim or MAE_CFG["decoder_dim"]

        # Determine the encoder's output spatial size before global pool
        # ResNet1DEncoder: stem (stride 2) + 4 blocks (each stride 2)
        # 5000 → 2500 → 1250 → 625 → 313 → 157 (approx)
        n_blocks = ENCODER_CFG.get("n_blocks", 4)
        self.spatial_len = seq_len // (2 ** (n_blocks + 1))  # ~78
        last_ch = ENCODER_CFG["base_filters"] * (2 ** (n_blocks - 1))  # 512

        # Instead of using the pooled embedding (which loses spatial info),
        # we'll take the feature map before global pooling.
        # This requires a hook into the encoder. We handle this in the
        # MAE wrapper.

        # Decoder: upsample from (B, last_ch, spatial_len) → (B, 12, 5000)
        layers = []
        ch = last_ch
        current_len = self.spatial_len
        target_len = seq_len

        # Progressive upsampling
        while current_len < target_len // 2:
            out_ch = max(ch // 2, decoder_dim)
            layers.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=4,
                                             stride=2, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            ch = out_ch
            current_len *= 2

        # Final layer to match exact output
        layers.append(nn.ConvTranspose1d(ch, n_leads, kernel_size=4,
                                         stride=2, padding=1))
        self.decoder = nn.Sequential(*layers)
        self._target_len = target_len

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """feature_map: (B, C, T_enc) → (B, 12, 5000)."""
        out = self.decoder(feature_map)
        # Adjust to exact target length
        if out.shape[2] > self._target_len:
            out = out[:, :, :self._target_len]
        elif out.shape[2] < self._target_len:
            out = F.interpolate(out, size=self._target_len, mode="linear",
                                align_corners=False)
        return out


# ---------------------------------------------------------------------------
# MAE Wrapper (encoder + decoder + masking)
# ---------------------------------------------------------------------------

class ECGMAE(nn.Module):
    """Masked Autoencoder for ECG.

    Wraps ResNet1DEncoder (without its final global pool + proj) and
    a lightweight ConvTranspose decoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.decoder = MAEDecoder()

        # We need the feature map before global pooling.
        # Store it via a hook.
        self._feature_map = None
        self.encoder.pool.register_forward_pre_hook(self._capture_feature_map)

    def _capture_feature_map(self, module, input):
        """Hook: capture tensor entering global avg pool."""
        self._feature_map = input[0]  # (B, C, T_enc)

    def forward(self, ecg: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            ecg:  (B, 12, 5000) original ECG
            mask: (B, 1, 5000)  binary mask, 1=keep 0=masked

        Returns:
            reconstruction: (B, 12, 5000) predicted ECG
            loss: scalar MSE on masked positions (computed in FP32)
        """
        # Match input dtype to model weights (BF16 under DeepSpeed)
        dtype = next(self.encoder.parameters()).dtype
        ecg = ecg.to(dtype)
        mask = mask.to(dtype)

        # Per-sample normalisation: zero-mean, unit-std per ECG
        # Keeps different datasets on the same scale
        ecg_mean = ecg.mean(dim=-1, keepdim=True)  # (B, 12, 1)
        ecg_std = ecg.std(dim=-1, keepdim=True).clamp(min=1e-6)
        ecg_normed = (ecg - ecg_mean) / ecg_std

        # Apply mask to input
        masked_input = ecg_normed * mask

        # Forward through encoder (we discard the embedding, keep feature map)
        _ = self.encoder(masked_input)
        feature_map = self._feature_map  # (B, C, T_enc)

        # Decode
        reconstruction = self.decoder(feature_map)

        # Loss in FP32 for numerical accuracy
        recon_f32 = reconstruction.float()
        target_f32 = ecg_normed.float()
        inv_mask_f32 = (1.0 - mask).float()  # 1 where masked
        n_masked = inv_mask_f32.sum().clamp(min=1.0)
        loss = ((recon_f32 - target_f32) ** 2 * inv_mask_f32).sum() / n_masked

        return reconstruction, loss

    def get_encoder_state_dict(self):
        """Extract encoder weights for downstream fine-tuning."""
        return self.encoder.state_dict()


# ---------------------------------------------------------------------------
# Distributed helpers (same as train_and_evaluate.py)
# ---------------------------------------------------------------------------

def _is_main_rank() -> bool:
    rank = int(os.environ.get("RANK", "0"))
    return rank == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _print_rank0(msg: str):
    if _is_main_rank():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
import os


def train_mae(args):
    """Main MAE pretraining loop."""
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
            name=f"mae-{time.strftime('%m%d-%H%M')}",
        )

    # Build dataset
    _print_rank0("Loading pretrain datasets …")
    full_dataset = build_pretrain_dataset()
    total = len(full_dataset)

    # Train/val split
    n_val = max(1, int(total * cfg["val_ratio"]))
    n_train = total - n_val
    rng = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=rng)
    _print_rank0(f"  Train: {n_train}, Val: {n_val}")

    # Distributed sampler
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        sampler=train_sampler, num_workers=8, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )

    # Val loader (no distributed — run on all ranks but only log on rank 0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
        drop_last=False,
    )

    # Model
    model = ECGMAE()
    param_count = sum(p.numel() for p in model.parameters())
    encoder_count = sum(p.numel() for p in model.encoder.parameters())
    _print_rank0(f"  MAE params: {param_count:,} "
                 f"(encoder: {encoder_count:,}, "
                 f"decoder: {param_count - encoder_count:,})")

    # DeepSpeed init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # Training
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_encoder_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    global_step = 0

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        # --- Train ---
        model_engine.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ecg = batch.to(device)  # (B, 12, 5000)
            mask = generate_mask(
                ECG_SEQ_LEN, cfg["mask_ratio"], cfg["mask_segment_len"],
                ecg.shape[0], device)

            _, loss = model_engine(ecg, mask)

            model_engine.backward(loss)
            model_engine.step()

            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1
            global_step += 1

            if _is_main_rank() and global_step % 50 == 0:
                wandb.log({"train/batch_loss": loss_val, "step": global_step})

        avg_train_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        model_engine.eval()
        val_loss_sum = 0.0
        val_n = 0
        raw_model = model_engine.module

        with torch.no_grad():
            for batch in val_loader:
                ecg = batch.to(device)
                mask = generate_mask(
                    ECG_SEQ_LEN, cfg["mask_ratio"], cfg["mask_segment_len"],
                    ecg.shape[0], device)
                _, loss = raw_model(ecg, mask)
                val_loss_sum += loss.item() * ecg.shape[0]
                val_n += ecg.shape[0]

        avg_val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t0

        # Get current LR
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

        # Early stopping
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
            "datasets_used": "PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG",
        }, PRETRAIN_CKPT)
        _print_rank0(f"\nEncoder saved to {PRETRAIN_CKPT}")
        _print_rank0(f"Best val loss: {best_val_loss:.6f}")

        # Save history
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
        description="ECG MAE Pretraining for ResNet1D Encoder")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank (set by DeepSpeed launcher)")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train_mae(args)


if __name__ == "__main__":
    main()
