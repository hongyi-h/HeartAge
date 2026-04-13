"""
Step 1: Prepare ECG data for Block 2 student distillation.

Handles:
  1. Load teacher soft labels from Block 1 (structural_age + domain_scores)
  2. Load & preprocess PTB-XL waveforms (concept supervision for rhythm/quality)
  3. Load & preprocess UKB ECG waveforms (main distillation, when available)
  4. Generate concept supervision targets
  5. Save processed datasets

Usage:
    python -m src.block2.prepare_data [--skip_ukb] [--skip_ptbxl]

Output:
    data/processed/block2_ptbxl.pt        (PTB-XL waveforms + labels)
    data/processed/block2_ukb_paired.pt   (UKB ECG + teacher labels, when available)
    results/block2/data_stats.json
"""

import argparse
import ast
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

from src.block2.config import (
    ECG_SAMPLE_RATE, ECG_DURATION_SEC, ECG_SEQ_LEN, ECG_N_LEADS,
    ECG_BANDPASS_LOW, ECG_BANDPASS_HIGH,
    PTBXL_DIR, PTBXL_PLUS_DIR, PROCESSED_DIR, RESULTS_DIR,
    BLOCK1_DIR, TEACHER_PRED, COHORT_STATS,
    PTBXL_RHYTHM_LABELS, DOMAIN_TO_CONCEPT,
    LEAD_NAMES,
)


# ---------------------------------------------------------------------------
# ECG signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: int,
                    low: float = ECG_BANDPASS_LOW,
                    high: float = ECG_BANDPASS_HIGH,
                    order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        signal: (n_samples, n_leads) or (n_samples,)
        fs: sampling frequency in Hz
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(np.float32)
    return np.stack([sosfiltfilt(sos, signal[:, i])
                     for i in range(signal.shape[1])], axis=1).astype(np.float32)


def resample_ecg(signal: np.ndarray, fs_orig: int,
                 fs_target: int = ECG_SAMPLE_RATE) -> np.ndarray:
    """Resample ECG signal to target frequency using polyphase method."""
    if fs_orig == fs_target:
        return signal
    g = gcd(fs_orig, fs_target)
    up = fs_target // g
    down = fs_orig // g
    if signal.ndim == 1:
        return resample_poly(signal, up, down).astype(np.float32)
    return np.stack([resample_poly(signal[:, i], up, down)
                     for i in range(signal.shape[1])], axis=1).astype(np.float32)


def pad_or_truncate(signal: np.ndarray,
                    target_len: int = ECG_SEQ_LEN) -> np.ndarray:
    """Pad (zero) or truncate signal to exact target length."""
    n = signal.shape[0]
    if n >= target_len:
        return signal[:target_len]
    pad_width = [(0, target_len - n)]
    if signal.ndim == 2:
        pad_width.append((0, 0))
    return np.pad(signal, pad_width, mode="constant").astype(np.float32)


def preprocess_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    """Full preprocessing pipeline: filter → resample → pad/truncate.

    Args:
        signal: (n_samples, n_leads)
        fs: original sampling rate
    Returns:
        (ECG_SEQ_LEN, n_leads) float32 array
    """
    # Bandpass filter at original sample rate
    filtered = bandpass_filter(signal, fs)
    # Resample to target rate
    resampled = resample_ecg(filtered, fs)
    # Pad or truncate to exact length
    return pad_or_truncate(resampled)


# ---------------------------------------------------------------------------
# PTB-XL loading
# ---------------------------------------------------------------------------

def load_ptbxl_waveforms(records_dir: Path, filenames: list,
                         fs: int = 500) -> np.ndarray:
    """Load PTB-XL waveform files (.dat/.hea via wfdb or numpy).

    Returns: (N, ECG_SEQ_LEN, 12) float32
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb package required for PTB-XL loading. "
                          "Install: pip install wfdb")

    waveforms = []
    n_failed = 0
    for i, fname in enumerate(filenames):
        fpath = records_dir / fname
        try:
            record = wfdb.rdrecord(str(fpath))
            sig = record.p_signal.astype(np.float32)  # (n_samples, n_leads)
            sig = preprocess_ecg(sig, fs)
            # Ensure 12 leads
            if sig.shape[1] < ECG_N_LEADS:
                pad = np.zeros((sig.shape[0], ECG_N_LEADS - sig.shape[1]),
                               dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=1)
            elif sig.shape[1] > ECG_N_LEADS:
                sig = sig[:, :ECG_N_LEADS]
            waveforms.append(sig)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  Warning: failed to load {fname}: {e}")
            waveforms.append(np.zeros((ECG_SEQ_LEN, ECG_N_LEADS),
                                      dtype=np.float32))

        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i+1}/{len(filenames)} waveforms "
                  f"({n_failed} failed)")

    print(f"  Total: {len(waveforms)} waveforms, {n_failed} failed")
    return np.stack(waveforms, axis=0)


def parse_ptbxl_scp_codes(scp_str: str) -> dict:
    """Parse PTB-XL scp_codes string to dict."""
    try:
        return ast.literal_eval(scp_str)
    except (ValueError, SyntaxError):
        return {}


def build_ptbxl_rhythm_labels(df: pd.DataFrame) -> np.ndarray:
    """Build binary rhythm/quality concept labels from PTB-XL SCP codes.

    Returns: (N, 4) float32 array for concepts 5-8
    """
    n = len(df)
    labels = np.zeros((n, len(PTBXL_RHYTHM_LABELS)), dtype=np.float32)

    for i, scp_str in enumerate(df["scp_codes"].values):
        codes = parse_ptbxl_scp_codes(scp_str)
        code_names = set(codes.keys())

        for j, (concept_name, target_codes) in enumerate(
                PTBXL_RHYTHM_LABELS.items()):
            if concept_name == "sinus_rhythm_conf":
                # Positive = sinus rhythm present
                labels[i, j] = 1.0 if code_names & set(target_codes) else 0.0
            elif concept_name == "QRS_duration_norm":
                # Positive = conduction abnormality present (inverted for "norm")
                labels[i, j] = 1.0 if code_names & set(target_codes) else 0.0
            elif concept_name == "ST_deviation_score":
                # Positive = ST changes present
                labels[i, j] = 1.0 if code_names & set(target_codes) else 0.0
            elif concept_name == "P_wave_quality":
                # Positive = atrial arrhythmia present (inverted for quality)
                labels[i, j] = 1.0 if code_names & set(target_codes) else 0.0

    return labels


def build_ptbxl_quality_labels(df: pd.DataFrame) -> np.ndarray:
    """Build quality concept labels from PTB-XL metadata.

    Returns: (N, 3) float32 array for concepts 9-11
    """
    n = len(df)
    labels = np.zeros((n, 3), dtype=np.float32)

    # Concept 9: signal_noise_ratio — invert noise columns
    for col in ["static_noise", "burst_noise", "baseline_drift"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            labels[:, 0] += vals  # higher = more noise

    # Normalize and invert: high quality = 1, low = 0
    max_noise = labels[:, 0].max()
    if max_noise > 0:
        labels[:, 0] = 1.0 - (labels[:, 0] / max_noise)
    else:
        labels[:, 0] = 1.0

    # Concept 10: baseline_stability — from baseline_drift column
    if "baseline_drift" in df.columns:
        drift = pd.to_numeric(df["baseline_drift"], errors="coerce").fillna(0)
        max_drift = drift.max()
        if max_drift > 0:
            labels[:, 1] = 1.0 - (drift.values / max_drift)
        else:
            labels[:, 1] = 1.0
    else:
        labels[:, 1] = 1.0

    # Concept 11: lead_completeness — from electrodes_problems column
    if "electrodes_problems" in df.columns:
        problems = pd.to_numeric(df["electrodes_problems"],
                                 errors="coerce").fillna(0)
        labels[:, 2] = (problems.values == 0).astype(np.float32)
    else:
        labels[:, 2] = 1.0

    return labels


def prepare_ptbxl():
    """Load and preprocess PTB-XL dataset for concept supervision."""
    print("\n[PTB-XL] Loading database …")
    db_path = PTBXL_DIR / "ptbxl_database.csv"
    df = pd.read_csv(db_path)
    print(f"  {len(df)} records")

    # Use 500Hz records
    records_dir = PTBXL_DIR
    filenames = df["filename_hr"].values.tolist()

    # Build concept supervision labels
    print("[PTB-XL] Building rhythm labels (concepts 5-8) …")
    rhythm_labels = build_ptbxl_rhythm_labels(df)
    print(f"  Rhythm label means: {rhythm_labels.mean(axis=0)}")

    print("[PTB-XL] Building quality labels (concepts 9-11) …")
    quality_labels = build_ptbxl_quality_labels(df)
    print(f"  Quality label means: {quality_labels.mean(axis=0)}")

    # Load waveforms
    print("[PTB-XL] Loading waveforms (500Hz) …")
    waveforms = load_ptbxl_waveforms(records_dir, filenames, fs=500)
    # Transpose to (N, leads, time) for PyTorch Conv1d
    waveforms = waveforms.transpose(0, 2, 1)  # (N, 12, 5000)

    # Use PTB-XL strat_fold for splits (fold 9=val, 10=test, rest=train)
    folds = df["strat_fold"].values
    split = np.where(folds <= 8, "train",
                     np.where(folds == 9, "val", "test"))

    # Metadata
    ages = pd.to_numeric(df["age"], errors="coerce").fillna(0).values.astype(
        np.float32)
    sex = pd.to_numeric(df["sex"], errors="coerce").fillna(0).values.astype(
        np.float32)
    ecg_ids = df["ecg_id"].values

    # Save
    out_path = PROCESSED_DIR / "block2_ptbxl.pt"
    torch.save({
        "waveforms": torch.from_numpy(waveforms),
        "rhythm_labels": torch.from_numpy(rhythm_labels),
        "quality_labels": torch.from_numpy(quality_labels),
        "ages": torch.from_numpy(ages),
        "sex": torch.from_numpy(sex),
        "ecg_ids": ecg_ids,
        "split": split,
    }, out_path)
    print(f"  Saved → {out_path}")
    print(f"  Shape: {waveforms.shape}")

    del waveforms; gc.collect()
    return len(df)


# ---------------------------------------------------------------------------
# UKB ECG + teacher labels
# ---------------------------------------------------------------------------

def prepare_ukb_paired():
    """Load UKB ECG waveforms paired with teacher soft labels.

    NOTE: UKB ECG waveforms (field 20205) must be downloaded separately.
    This function expects preprocessed waveforms at:
        data/processed/ukb_ecg_waveforms.npy  (N, 12, 5000)
        data/processed/ukb_ecg_eids.npy       (N,) participant IDs

    If not available, creates a stub dataset for pipeline testing.
    """
    print("\n[UKB Paired] Loading teacher predictions …")
    teacher_pred = pd.read_csv(TEACHER_PRED)
    print(f"  Teacher predictions: {len(teacher_pred)} participants")

    # Teacher soft labels: structural_age + domain scores
    domain_cols = [c for c in teacher_pred.columns if c.startswith("domain_")]
    print(f"  Domain score columns: {domain_cols}")

    # Check for UKB ECG waveforms
    ecg_path = PROCESSED_DIR / "ukb_ecg_waveforms.npy"
    eid_path = PROCESSED_DIR / "ukb_ecg_eids.npy"

    if ecg_path.exists() and eid_path.exists():
        print("[UKB Paired] Loading ECG waveforms …")
        waveforms = np.load(ecg_path)  # (N, 12, 5000)
        eids = np.load(eid_path)
        print(f"  Loaded {len(eids)} ECG waveforms, shape={waveforms.shape}")

        # Match with teacher predictions
        teacher_eids = set(teacher_pred["eid"].values)
        mask = np.array([e in teacher_eids for e in eids])
        print(f"  Matched with teacher: {mask.sum()}/{len(eids)}")

        waveforms = waveforms[mask]
        eids = eids[mask]

        # Get teacher labels for matched participants
        teacher_indexed = teacher_pred.set_index("eid")
        struct_ages = np.array([teacher_indexed.loc[e, "structural_age"]
                                for e in eids], dtype=np.float32)
        deviations = np.array([teacher_indexed.loc[e, "deviation"]
                               for e in eids], dtype=np.float32)
        domain_scores = np.zeros((len(eids), len(domain_cols)),
                                 dtype=np.float32)
        for j, col in enumerate(domain_cols):
            domain_scores[:, j] = np.array(
                [teacher_indexed.loc[e, col] for e in eids], dtype=np.float32)

        # Load Block 1 data for split info and chronological ages
        block1_data = pd.read_parquet(PROCESSED_DIR / "block1_data.parquet",
                                      columns=["eid", "age", "split"])
        block1_indexed = block1_data.set_index("eid")
        chrono_ages = np.array([block1_indexed.loc[e, "age"]
                                for e in eids], dtype=np.float32)
        splits = np.array([block1_indexed.loc[e, "split"]
                           for e in eids])

    else:
        print("[UKB Paired] ECG waveforms not found — creating stub dataset.")
        print(f"  Expected: {ecg_path}")
        print(f"  To prepare UKB ECGs, run the ECG extraction pipeline first.")
        print(f"  Creating a small synthetic stub for pipeline testing …")

        n_stub = 100
        rng = np.random.RandomState(42)
        waveforms = rng.randn(n_stub, ECG_N_LEADS,
                              ECG_SEQ_LEN).astype(np.float32) * 0.1
        eids = teacher_pred["eid"].values[:n_stub]

        teacher_sub = teacher_pred.iloc[:n_stub]
        struct_ages = teacher_sub["structural_age"].values.astype(np.float32)
        deviations = teacher_sub["deviation"].values.astype(np.float32)
        domain_scores = teacher_sub[domain_cols].values.astype(np.float32)

        block1_data = pd.read_parquet(PROCESSED_DIR / "block1_data.parquet",
                                      columns=["eid", "age", "split"])
        block1_indexed = block1_data.set_index("eid")
        chrono_ages = np.array([block1_indexed.loc[e, "age"]
                                for e in eids], dtype=np.float32)
        splits = np.array([block1_indexed.loc[e, "split"]
                           for e in eids])

    # Save
    out_path = PROCESSED_DIR / "block2_ukb_paired.pt"
    torch.save({
        "waveforms": torch.from_numpy(waveforms),
        "eids": eids,
        "structural_age": torch.from_numpy(struct_ages),
        "chrono_age": torch.from_numpy(chrono_ages),
        "deviation": torch.from_numpy(deviations),
        "domain_scores": torch.from_numpy(domain_scores),
        "domain_cols": domain_cols,
        "split": splits,
        "is_stub": not ecg_path.exists(),
    }, out_path)
    print(f"  Saved → {out_path}")
    print(f"  Waveforms shape: {waveforms.shape}")
    print(f"  Splits: {pd.Series(splits).value_counts().to_dict()}")

    del waveforms; gc.collect()
    return len(eids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ukb", action="store_true",
                        help="Skip UKB ECG preparation")
    parser.add_argument("--skip_ptbxl", action="store_true",
                        help="Skip PTB-XL preparation")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}

    if not args.skip_ptbxl:
        n_ptbxl = prepare_ptbxl()
        stats["n_ptbxl"] = n_ptbxl

    if not args.skip_ukb:
        n_ukb = prepare_ukb_paired()
        stats["n_ukb_paired"] = n_ukb

    # Save stats
    stats_path = RESULTS_DIR / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nData stats → {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
