"""Unified multi-dataset ECG loader for self-supervised pretraining.

Datasets loaded (all except UKB):
  - PTB-XL       ~987 records   (WFDB, 12-lead, 500Hz, 5000 samples)
  - SPH          ~25,770        (HDF5, 12-lead, 500Hz, 5000 samples)
  - CODE-15%     ~20,001        (HDF5, 12-lead, 400Hz→resample to 500Hz)
  - ECG-Arrhythmia ~1,000       (WFDB .mat, 12-lead, 500Hz, 5000 samples)
  - MIMIC-IV-ECG ~800,035       (WFDB, 12-lead, 500Hz, 5000 samples)

All outputs: (12, 5000) float32, bandpass-filtered, z-normalised per-lead.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from scipy.signal import resample

from src.block2.config import (
    PROJECT_ROOT, ECG_N_LEADS, ECG_SEQ_LEN, ECG_SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Cache directory (populated by cache_pretrain_data.py)
# ---------------------------------------------------------------------------
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "ecg_cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_leads(ecg: np.ndarray) -> np.ndarray:
    """Per-lead z-normalisation.  Input/output: (12, T)."""
    mu = ecg.mean(axis=1, keepdims=True)
    sigma = ecg.std(axis=1, keepdims=True) + 1e-8
    return ((ecg - mu) / sigma).astype(np.float32)


def _pad_or_truncate(ecg: np.ndarray, target_len: int = ECG_SEQ_LEN
                     ) -> np.ndarray:
    """Ensure temporal dimension is exactly target_len.  (C, T) → (C, T')."""
    T = ecg.shape[1]
    if T >= target_len:
        return ecg[:, :target_len]
    pad_width = target_len - T
    return np.pad(ecg, ((0, 0), (0, pad_width)), mode="constant")


def _resample_ecg(ecg: np.ndarray, src_hz: int, tgt_hz: int = ECG_SAMPLE_RATE
                  ) -> np.ndarray:
    """Resample (C, T_src) to (C, T_tgt) keeping duration constant."""
    if src_hz == tgt_hz:
        return ecg
    duration = ecg.shape[1] / src_hz
    tgt_len = int(round(duration * tgt_hz))
    return resample(ecg, tgt_len, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-dataset loaders — each returns an iterable of (12, 5000) np arrays
# ---------------------------------------------------------------------------

class PTBXLDataset(Dataset):
    """PTB-XL 500 Hz WFDB records."""

    def __init__(self, root: Path = None):
        root = root or (PROJECT_ROOT / "data" / "PTB-XL" / "records500")
        self.hea_files = sorted(glob.glob(str(root / "**" / "*.hea"),
                                          recursive=True))
        if not self.hea_files:
            raise FileNotFoundError(f"No PTB-XL .hea files in {root}")

    def __len__(self):
        return len(self.hea_files)

    def __getitem__(self, idx):
        import wfdb
        rec_path = self.hea_files[idx].replace(".hea", "")
        record = wfdb.rdrecord(rec_path)
        ecg = record.p_signal.T.astype(np.float32)  # (12, T)
        ecg = _pad_or_truncate(ecg)
        ecg = _normalise_leads(ecg)
        return torch.from_numpy(ecg)


class SPHDataset(Dataset):
    """SPH HDF5 — one file per record, key='ecg', shape (12, 5000)."""

    def __init__(self, root: Path = None):
        root = root or (PROJECT_ROOT / "data" / "SPH")
        self.paths = sorted(glob.glob(str(root / "*.h5")))
        if not self.paths:
            raise FileNotFoundError(f"No SPH .h5 files in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            ecg = f["ecg"][:].astype(np.float32)  # (12, 5000)
        ecg = _pad_or_truncate(ecg)
        ecg = _normalise_leads(ecg)
        return torch.from_numpy(ecg)


class CODE15Dataset(Dataset):
    """CODE-15% HDF5 — shape (N, 4096, 12), 400 Hz → resample to 500 Hz."""

    def __init__(self, root: Path = None):
        root = root or (PROJECT_ROOT / "data" / "CODE-15%")
        self.path = root / "exams_part0.hdf5"
        if not self.path.exists():
            raise FileNotFoundError(f"CODE-15% file not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            self.n = f["tracings"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            ecg = f["tracings"][idx]  # (4096, 12)
        ecg = ecg.T.astype(np.float32)  # (12, 4096)
        ecg = _resample_ecg(ecg, src_hz=400, tgt_hz=500)  # (12, 5120)
        ecg = _pad_or_truncate(ecg)  # (12, 5000)
        ecg = _normalise_leads(ecg)
        return torch.from_numpy(ecg)


class ECGArrhythmiaDataset(Dataset):
    """ECG-Arrhythmia WFDB (.hea + .mat), 500 Hz, 12-lead."""

    def __init__(self, root: Path = None):
        root = root or (PROJECT_ROOT / "data" / "ECG-Arrhythmia" / "WFDBRecords")
        self.hea_files = sorted(glob.glob(str(root / "**" / "*.hea"),
                                          recursive=True))
        if not self.hea_files:
            raise FileNotFoundError(f"No ECG-Arrhythmia .hea in {root}")

    def __len__(self):
        return len(self.hea_files)

    def __getitem__(self, idx):
        import wfdb
        rec_path = self.hea_files[idx].replace(".hea", "")
        record = wfdb.rdrecord(rec_path)
        ecg = record.p_signal.T.astype(np.float32)  # (12, T)
        ecg = _pad_or_truncate(ecg)
        ecg = _normalise_leads(ecg)
        return torch.from_numpy(ecg)


class MIMICECGDataset(Dataset):
    """MIMIC-IV-ECG WFDB records, 500 Hz, 12-lead.

    Uses record_list.csv to locate files.
    """

    def __init__(self, root: Path = None, max_records: Optional[int] = None):
        root = root or (PROJECT_ROOT / "data" / "MIMIC-IV-ECG")
        record_csv = root / "record_list.csv"
        if not record_csv.exists():
            raise FileNotFoundError(f"MIMIC record_list.csv not found: {root}")

        import csv
        self.root = root
        self.record_paths: List[str] = []
        with open(record_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.record_paths.append(row["path"])
                if max_records and len(self.record_paths) >= max_records:
                    break

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        import wfdb
        rec_full = str(self.root / self.record_paths[idx])
        try:
            record = wfdb.rdrecord(rec_full)
            ecg = record.p_signal.T.astype(np.float32)  # (12, T)
        except Exception:
            # Corrupted record → return zeros (will be masked in training)
            ecg = np.zeros((ECG_N_LEADS, ECG_SEQ_LEN), dtype=np.float32)
        ecg = _pad_or_truncate(ecg)
        ecg = _normalise_leads(ecg)
        return torch.from_numpy(ecg)


# ---------------------------------------------------------------------------
# Cached dataset (fast: numpy mmap, no per-sample I/O)
# ---------------------------------------------------------------------------

class CachedECGDataset(Dataset):
    """Read pre-cached .npy file via memory mapping — near-zero I/O latency."""

    def __init__(self, path: Path):
        self.data = np.load(str(path), mmap_mode="r")  # (N, 12, 5000) float16

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.float32))


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

_CACHE_MAP = {
    "ptbxl": ("PTB-XL", PTBXLDataset, {}),
    "sph": ("SPH", SPHDataset, {}),
    "code15": ("CODE-15%", CODE15Dataset, {}),
    "arrhythmia": ("ECG-Arrhythmia", ECGArrhythmiaDataset, {}),
    "mimic": ("MIMIC-IV-ECG", MIMICECGDataset, {}),
}


def build_pretrain_dataset(
    use_ptbxl: bool = True,
    use_sph: bool = True,
    use_code15: bool = True,
    use_arrhythmia: bool = True,
    use_mimic: bool = True,
    mimic_max: Optional[int] = None,
) -> ConcatDataset:
    """Build a ConcatDataset from all available ECG sources.

    Uses pre-cached .npy files if available (run cache_pretrain_data.py first).
    Falls back to lazy per-file loading otherwise.
    """
    flags = {
        "ptbxl": use_ptbxl, "sph": use_sph, "code15": use_code15,
        "arrhythmia": use_arrhythmia, "mimic": use_mimic,
    }

    datasets = []
    names = []
    cached_count = 0

    for key, (display_name, LazyClass, kwargs) in _CACHE_MAP.items():
        if not flags[key]:
            continue

        cache_path = CACHE_DIR / f"{key}.npy"
        if cache_path.exists():
            ds = CachedECGDataset(cache_path)
            datasets.append(ds)
            names.append(f"{display_name}: {len(ds)} [cached]")
            cached_count += 1
        else:
            try:
                kw = dict(kwargs)
                if key == "mimic" and mimic_max:
                    kw["max_records"] = mimic_max
                ds = LazyClass(**kw)
                datasets.append(ds)
                names.append(f"{display_name}: {len(ds)} [lazy]")
            except FileNotFoundError as e:
                print(f"  [SKIP] {display_name}: {e}")

    if not datasets:
        raise RuntimeError("No ECG datasets found for pretraining.")

    total_active = len(datasets)
    if cached_count < total_active:
        print(f"  WARNING: {total_active - cached_count} dataset(s) using "
              f"slow lazy loading. Run: python -m src.block2.cache_pretrain_data")

    print(f"Pretrain datasets loaded:")
    for n in names:
        print(f"  {n}")
    total = sum(len(d) for d in datasets)
    print(f"  TOTAL: {total}")

    return ConcatDataset(datasets)
