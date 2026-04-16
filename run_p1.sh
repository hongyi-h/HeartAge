#!/usr/bin/env bash
# P1: Block 2 pilot — ECG student distillation
# Run from project root: bash run_p1.sh [NUM_GPUS]
#
# This runs the FULL Block 2 pipeline:
#   Step 0: Extract UKB ECG waveforms (skip if already done)
#   Step 1: Pretrain ECG encoder via SparK1D MAE
#   Step 2: Prepare distillation data (UKB paired + PTB-XL)
#   Step 3: Train all student variants via DeepSpeed
#
# Prerequisites:
#   - Block 1 complete (results/block1/ exists)
#   - UKB ECG XML files available (set UKB_ECG_DIR)
#   - ECG pretraining datasets in data/ (PTB-XL, SPH, etc.)
#   - pip install deepspeed wfdb h5py
#
# Environment variables:
#   UKB_ECG_DIR  — path to UKB ECG XML bulk download directory
#   NUM_GPUS     — override GPU count (default: auto-detect)
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "==== P1: Block 2 Pilot — ECG Student Distillation ===="
echo "GPUs: ${NUM_GPUS}"
echo ""

# --- Step 0: Extract UKB ECGs (if not already done) ---
ECG_WF_PATH="data/processed/ukb_ecg_waveforms.npy"
if [ -f "${ECG_WF_PATH}" ]; then
    echo "[Step 0] UKB ECG waveforms already extracted. Skipping."
else
    if [ -z "${UKB_ECG_DIR:-}" ]; then
        echo "[Step 0] WARNING: UKB_ECG_DIR not set and ${ECG_WF_PATH} not found."
        echo "  Set UKB_ECG_DIR to the directory containing UKB ECG XML files."
        echo "  Example: UKB_ECG_DIR=/data/ukb/ecg bash run_p1.sh"
        echo "  Continuing without UKB ECGs — will use stub data for pipeline testing."
        echo ""
    else
        echo "[Step 0] Extracting UKB ECG waveforms from ${UKB_ECG_DIR} ..."
        python scripts/extract_ukb_ecg.py --ecg_dir "${UKB_ECG_DIR}"
        echo ""
    fi
fi

# --- Step 1: Pretrain ECG encoder (SparK1D MAE) ---
PRETRAIN_CKPT="results/block2/pretrain/encoder_pretrained.pt"
if [ -f "${PRETRAIN_CKPT}" ]; then
    echo "[Step 1] Pretrained encoder already exists. Skipping."
else
    echo "[Step 1] Pretraining ECG encoder (SparK1D) ..."
    bash run_pretrain_ecg.sh "${NUM_GPUS}"
fi
echo ""

# --- Step 2: Prepare distillation data ---
echo "[Step 2] Preparing distillation data (PTB-XL + UKB paired) ..."
python -m src.block2.prepare_data
echo ""

# --- Step 3: Train student variants ---
echo "[Step 3] Training student variants (DeepSpeed, ${NUM_GPUS} GPUs) ..."

DS_ARGS="--num_gpus=${NUM_GPUS}"
if [ -n "${HOSTFILE:-}" ]; then
    DS_ARGS="${DS_ARGS} --hostfile=${HOSTFILE}"
fi
if [ -n "${MASTER_ADDR:-}" ]; then
    DS_ARGS="${DS_ARGS} --master_addr=${MASTER_ADDR}"
fi
if [ -n "${MASTER_PORT:-}" ]; then
    DS_ARGS="${DS_ARGS} --master_port=${MASTER_PORT}"
fi

deepspeed ${DS_ARGS} \
    --module src.block2.train_and_evaluate \
    --deepspeed_config src/block2/ds_config.json

echo ""
echo "==== P1 Complete ===="
echo "Results in results/block2/"
echo ""
echo "Key metrics to check:"
echo "  1. R20 (Full Student) structural_age Pearson r vs teacher — target > 0.70"
echo "  2. R20 vs R21 (Chrono Student) — FT target should beat chrono target"
echo "  3. R20 concept alignment — structural concepts should correlate with teacher domain_scores"
