#!/usr/bin/env bash
# ECG Encoder Pretraining (MAE + DeepSpeed)
# Run from project root: bash run_pretrain_ecg.sh [NUM_GPUS]
#
# Prerequisites:
#   - ECG datasets in data/ (PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG)
#   - pip install deepspeed wfdb h5py scipy
#
# Output:
#   - results/block2/pretrain/encoder_pretrained.pt
#   - results/block2/pretrain/pretrain_history.json
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "==== ECG MAE Pretraining (${NUM_GPUS} GPUs) ===="
echo "Datasets: PTB-XL, SPH, CODE-15%, ECG-Arrhythmia, MIMIC-IV-ECG"
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
echo "==== Pretraining complete ===="
echo "Encoder checkpoint: results/block2/pretrain/encoder_pretrained.pt"
echo ""
echo "Next: bash run_block2.sh"
