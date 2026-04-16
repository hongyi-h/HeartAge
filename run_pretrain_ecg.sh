#!/usr/bin/env bash
# ECG Encoder Pretraining (SparK1D + DeepSpeed)
# Run from project root: bash run_pretrain_ecg.sh [NUM_GPUS]
#
# Method: SparK1D — Sparse and Hierarchical Masked Modeling for 1D CNNs
#   - Sparse convolutions prevent information leakage from masked to unmasked
#   - Hierarchical UNet decoder uses multi-scale encoder features
#   - 60% mask ratio (optimal for CNN per ConvNeXt V2 / SparK findings)
#
# Prerequisites:
#   - ECG datasets in data/ (PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG)
#   - pip install deepspeed wfdb h5py scipy wandb
#
# Output:
#   - results/block2/pretrain/encoder_pretrained.pt
#   - results/block2/pretrain/pretrain_history.json
set -euo pipefail
cd "$(dirname "$0")"

# --- Step 0: Cache datasets (one-time, skips if already done) ---
echo "==== Checking/building ECG data cache ===="
python -m src.block2.cache_pretrain_data
echo ""

NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "==== SparK1D ECG Pretraining (${NUM_GPUS} GPUs) ===="
echo "Datasets: PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG"
echo "Method: SparK1D (mask_ratio=0.60, input_size=4992)"
echo ""

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
    --module src.block2.pretrain_mae \
    --deepspeed_config src/block2/ds_config_pretrain.json

echo ""
echo "==== SparK1D pretraining complete ===="
echo "Encoder checkpoint: results/block2/pretrain/encoder_pretrained.pt"
echo ""
echo "Next: bash run_block2.sh"
