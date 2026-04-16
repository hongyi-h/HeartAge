"""
Extract and preprocess UKB resting 12-lead ECG waveforms (field 20205).

UKB distributes ECG data as XML files in bulk downloads. This script:
  1. Scans the specified directory for ECG XML files
  2. Parses waveform data from the UKB ECG XML format
  3. Preprocesses: bandpass filter → resample → pad/truncate
  4. Matches with Block 1 cohort (by eid)
  5. Saves: data/processed/ukb_ecg_waveforms.npy, ukb_ecg_eids.npy

Usage:
    python scripts/extract_ukb_ecg.py --ecg_dir /path/to/ukb/ecg/bulk

    The ecg_dir should contain XML files named like:
        <eid>_20205_<instance>_0.xml
    or subdirectories with XML files.

    If your ECGs are in a different format (e.g. DICOM, WAV), modify the
    parse function accordingly.

Output:
    data/processed/ukb_ecg_waveforms.npy  (N, 12, 5000) float32
    data/processed/ukb_ecg_eids.npy       (N,) int64
"""

import argparse
import gc
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ECG_SAMPLE_RATE = 500
ECG_DURATION_SEC = 10
ECG_SEQ_LEN = ECG_SAMPLE_RATE * ECG_DURATION_SEC  # 5000
ECG_N_LEADS = 12
ECG_BANDPASS_LOW = 0.5
ECG_BANDPASS_HIGH = 40.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Standard 12-lead order
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ---------------------------------------------------------------------------
# Signal processing (same as src/block2/prepare_data.py)
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, low=ECG_BANDPASS_LOW, high=ECG_BANDPASS_HIGH):
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(np.float32)
    return np.stack([sosfiltfilt(sos, signal[:, i])
                     for i in range(signal.shape[1])], axis=1).astype(np.float32)


def resample_ecg(signal, fs_orig, fs_target=ECG_SAMPLE_RATE):
    if fs_orig == fs_target:
        return signal
    g = gcd(fs_orig, fs_target)
    up, down = fs_target // g, fs_orig // g
    if signal.ndim == 1:
        return resample_poly(signal, up, down).astype(np.float32)
    return np.stack([resample_poly(signal[:, i], up, down)
                     for i in range(signal.shape[1])], axis=1).astype(np.float32)


def pad_or_truncate(signal, target_len=ECG_SEQ_LEN):
    n = signal.shape[0]
    if n >= target_len:
        return signal[:target_len]
    pad_width = [(0, target_len - n)]
    if signal.ndim == 2:
        pad_width.append((0, 0))
    return np.pad(signal, pad_width, mode="constant").astype(np.float32)


def preprocess_ecg(signal, fs):
    filtered = bandpass_filter(signal, fs)
    resampled = resample_ecg(filtered, fs)
    return pad_or_truncate(resampled)


# ---------------------------------------------------------------------------
# UKB ECG XML parsing
# ---------------------------------------------------------------------------

def parse_ukb_ecg_xml(xml_path: Path) -> tuple:
    """Parse UKB ECG XML file (field 20205).

    UKB ECG XML typically has structure:
    <RestingECG>
        <Waveform>
            <WaveformType>Rhythm</WaveformType>
            <SampleRate>500</SampleRate>
            <LeadData>
                <LeadID>I</LeadID>
                <WaveFormData>comma-separated values</WaveFormData>
            </LeadData>
            ...
        </Waveform>
    </RestingECG>

    Returns: (signal_array (n_samples, 12), sample_rate)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Handle XML namespace if present
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Find Rhythm waveform (prefer over Median)
    waveforms = root.findall(f".//{ns}Waveform")
    target_wf = None
    for wf in waveforms:
        wf_type = wf.find(f"{ns}WaveformType")
        if wf_type is not None and "Rhythm" in (wf_type.text or ""):
            target_wf = wf
            break
    if target_wf is None and waveforms:
        target_wf = waveforms[0]
    if target_wf is None:
        raise ValueError(f"No waveform found in {xml_path}")

    # Sample rate
    sr_elem = target_wf.find(f"{ns}SampleRate") or \
              target_wf.find(f"{ns}SampleBase")
    sample_rate = int(sr_elem.text) if sr_elem is not None else 500

    # Parse lead data
    lead_data = {}
    for ld in target_wf.findall(f".//{ns}LeadData"):
        lead_id_elem = ld.find(f"{ns}LeadID")
        data_elem = ld.find(f"{ns}WaveFormData") or ld.find(f"{ns}WaveformData")
        if lead_id_elem is None or data_elem is None:
            continue
        lead_id = lead_id_elem.text.strip()
        # Parse comma- or space-separated values
        text = data_elem.text.strip()
        values = [float(v) for v in re.split(r"[,\s]+", text) if v]
        lead_data[lead_id] = np.array(values, dtype=np.float32)

    # Assemble in standard order
    if not lead_data:
        raise ValueError(f"No lead data parsed from {xml_path}")

    # Find max length
    max_len = max(len(v) for v in lead_data.values())

    # Map to standard order
    signal = np.zeros((max_len, ECG_N_LEADS), dtype=np.float32)
    for i, lead_name in enumerate(LEAD_ORDER):
        if lead_name in lead_data:
            arr = lead_data[lead_name]
            signal[:len(arr), i] = arr
        else:
            # Try alternate names (e.g., "Lead I" vs "I")
            for key in lead_data:
                if lead_name in key or key in lead_name:
                    arr = lead_data[key]
                    signal[:len(arr), i] = arr
                    break

    # Convert from ADC units to mV (UKB typically uses units of uV)
    # Check if values seem too large (ADC units)
    if np.abs(signal).max() > 100:
        signal = signal / 1000.0  # uV → mV

    return signal, sample_rate


# ---------------------------------------------------------------------------
# Discovery and extraction
# ---------------------------------------------------------------------------

def find_ecg_files(ecg_dir: Path, instance: int = 2) -> dict:
    """Find all UKB ECG XML files and map eid → path.

    Naming convention: <eid>_20205_<instance>_0.xml
    If instance filtering is desired, set instance parameter.
    instance=2 corresponds to the imaging visit.
    """
    eid_to_path = {}
    patterns = ["*.xml", "**/*.xml"]

    for pattern in patterns:
        for fpath in ecg_dir.glob(pattern):
            match = re.match(r"(\d+)_20205_(\d+)_(\d+)\.xml", fpath.name)
            if match:
                eid = int(match.group(1))
                inst = int(match.group(2))
                # Prefer imaging instance
                if instance is not None and inst != instance:
                    continue
                eid_to_path[eid] = fpath
            elif fpath.suffix == ".xml" and fpath.name[0].isdigit():
                # Try to extract eid from filename
                eid_match = re.match(r"(\d+)", fpath.name)
                if eid_match:
                    eid = int(eid_match.group(1))
                    if eid not in eid_to_path:
                        eid_to_path[eid] = fpath

        if eid_to_path:
            break  # Don't recurse if top-level has files

    return eid_to_path


def extract_all(ecg_dir: Path, cohort_eids: set = None,
                max_workers: int = 4) -> tuple:
    """Extract and preprocess all ECGs.

    Returns: (waveforms array, eids array)
    """
    print(f"Scanning {ecg_dir} for ECG XML files ...")

    # Try different instance numbers (2 = imaging, 3 = repeat)
    eid_to_path = find_ecg_files(ecg_dir, instance=2)
    if not eid_to_path:
        eid_to_path = find_ecg_files(ecg_dir, instance=None)
    print(f"  Found {len(eid_to_path)} ECG files")

    if cohort_eids:
        matched = {e: p for e, p in eid_to_path.items() if e in cohort_eids}
        print(f"  Matched with cohort: {len(matched)}/{len(eid_to_path)}")
        eid_to_path = matched

    if not eid_to_path:
        raise RuntimeError(f"No ECG files found in {ecg_dir}")

    # Process
    waveforms = []
    eids = []
    n_failed = 0
    sorted_items = sorted(eid_to_path.items())

    for i, (eid, fpath) in enumerate(sorted_items):
        try:
            signal, fs = parse_ukb_ecg_xml(fpath)
            processed = preprocess_ecg(signal, fs)
            # Transpose to (12, 5000) for Conv1d
            processed = processed.T  # (12, 5000)
            waveforms.append(processed)
            eids.append(eid)
        except Exception as e:
            n_failed += 1
            if n_failed <= 10:
                print(f"  Warning: Failed {fpath.name}: {e}")

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(sorted_items)} "
                  f"(ok={len(waveforms)}, failed={n_failed})")

    print(f"  Done: {len(waveforms)} successfully processed, "
          f"{n_failed} failed")

    waveforms_arr = np.stack(waveforms, axis=0).astype(np.float32)
    eids_arr = np.array(eids, dtype=np.int64)

    return waveforms_arr, eids_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract UKB ECG waveforms for Block 2 distillation")
    parser.add_argument("--ecg_dir", type=str, required=True,
                        help="Directory containing UKB ECG XML files")
    parser.add_argument("--instance", type=int, default=2,
                        help="UKB instance number (2=imaging visit)")
    parser.add_argument("--cohort_file", type=str, default=None,
                        help="Path to teacher predictions CSV (for eid filtering)")
    args = parser.parse_args()

    ecg_dir = Path(args.ecg_dir)
    if not ecg_dir.exists():
        print(f"ERROR: ECG directory not found: {ecg_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cohort eids (from teacher predictions) for filtering
    cohort_eids = None
    cohort_file = args.cohort_file or str(
        PROJECT_ROOT / "results" / "block1" / "predictions" /
        "full_teacher_pred.csv"
    )
    if Path(cohort_file).exists():
        import pandas as pd
        pred = pd.read_csv(cohort_file)
        cohort_eids = set(pred["eid"].values)
        print(f"Cohort filter: {len(cohort_eids)} eids from {cohort_file}")

    # Extract
    waveforms, eids = extract_all(ecg_dir, cohort_eids)

    # Save
    wf_path = OUTPUT_DIR / "ukb_ecg_waveforms.npy"
    eid_path = OUTPUT_DIR / "ukb_ecg_eids.npy"
    np.save(wf_path, waveforms)
    np.save(eid_path, eids)
    print(f"\nSaved waveforms: {wf_path}  shape={waveforms.shape}")
    print(f"Saved eids:      {eid_path}  shape={eids.shape}")

    # Summary
    print(f"\nNext steps:")
    print(f"  1. Run: python -m src.block2.prepare_data")
    print(f"  2. Run: bash run_block2.sh")


if __name__ == "__main__":
    main()
