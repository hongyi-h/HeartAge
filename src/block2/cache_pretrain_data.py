"""Pre-cache all ECG datasets into numpy memory-mapped arrays.

Eliminates per-sample file I/O during training.  Each dataset is converted
into a single .npy file of shape (N, 12, 5000) in float16.

Usage:
    python -m src.block2.cache_pretrain_data

Output:
    data/processed/ecg_cache/ptbxl.npy
    data/processed/ecg_cache/sph.npy
    data/processed/ecg_cache/code15.npy
    data/processed/ecg_cache/arrhythmia.npy
    data/processed/ecg_cache/mimic.npy
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.block2.config import PROJECT_ROOT, ECG_N_LEADS, ECG_SEQ_LEN
from src.block2.pretrain_data import (
    PTBXLDataset, SPHDataset, CODE15Dataset,
    ECGArrhythmiaDataset, MIMICECGDataset,
)

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "ecg_cache"


def cache_one_dataset(dataset, name: str, num_workers: int = 16,
                      batch_size: int = 256):
    """Convert a lazy dataset into a memory-mapped .npy file."""
    N = len(dataset)
    outpath = CACHE_DIR / f"{name}.npy"

    if outpath.exists():
        existing = np.load(str(outpath), mmap_mode="r")
        if existing.shape[0] == N:
            print(f"  [CACHED] {name}: {N} records already at {outpath}")
            return
        print(f"  [STALE] {name}: cached {existing.shape[0]} != {N}, rebuilding")

    print(f"  Caching {name}: {N} records → {outpath} ...", flush=True)
    t0 = time.time()

    # Allocate output memmap (avoids holding full array in RAM)
    arr = np.lib.format.open_memmap(
        str(outpath), mode="w+", dtype=np.float16,
        shape=(N, ECG_N_LEADS, ECG_SEQ_LEN),
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=False, drop_last=False,
        persistent_workers=False,
    )

    offset = 0
    for i, batch in enumerate(loader):
        b = batch.numpy().astype(np.float16)  # (B, 12, 5000)
        n = b.shape[0]
        arr[offset:offset + n] = b
        offset += n

        if (i + 1) % 100 == 0:
            arr.flush()
            pct = offset / N * 100
            elapsed = time.time() - t0
            eta = elapsed / offset * (N - offset) if offset else 0
            print(f"    {name}: {offset}/{N} ({pct:.1f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    arr.flush()
    del arr
    elapsed = time.time() - t0
    size_gb = outpath.stat().st_size / 1e9
    print(f"  Done {name}: {N} records, {size_gb:.1f} GB, {elapsed:.0f}s")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ECG Cache directory: {CACHE_DIR}\n")

    # Datasets listed from smallest to largest (fail fast on small ones)
    tasks = []

    try:
        tasks.append((PTBXLDataset(), "ptbxl", 8))
    except FileNotFoundError as e:
        print(f"  [SKIP] PTB-XL: {e}")

    try:
        tasks.append((SPHDataset(), "sph", 8))
    except FileNotFoundError as e:
        print(f"  [SKIP] SPH: {e}")

    try:
        tasks.append((CODE15Dataset(), "code15", 4))  # fewer workers: HDF5 not fork-safe
    except FileNotFoundError as e:
        print(f"  [SKIP] CODE-15%: {e}")

    try:
        tasks.append((ECGArrhythmiaDataset(), "arrhythmia", 8))
    except FileNotFoundError as e:
        print(f"  [SKIP] ECG-Arrhythmia: {e}")

    try:
        tasks.append((MIMICECGDataset(), "mimic", 32))
    except FileNotFoundError as e:
        print(f"  [SKIP] MIMIC-IV-ECG: {e}")

    if not tasks:
        print("ERROR: No datasets found. Check data/ directory.")
        sys.exit(1)

    total_t0 = time.time()
    for dataset, name, workers in tasks:
        cache_one_dataset(dataset, name, num_workers=workers)

    total_elapsed = time.time() - total_t0
    print(f"\nAll caching complete in {total_elapsed:.0f}s")

    # Print summary
    print("\nCached files:")
    total_size = 0
    total_records = 0
    for f in sorted(CACHE_DIR.glob("*.npy")):
        arr = np.load(str(f), mmap_mode="r")
        size = f.stat().st_size / 1e9
        total_size += size
        total_records += arr.shape[0]
        print(f"  {f.name}: {arr.shape[0]:>8,} records, {size:.1f} GB")
    print(f"  {'TOTAL':>20s}: {total_records:>8,} records, {total_size:.1f} GB")


if __name__ == "__main__":
    main()
