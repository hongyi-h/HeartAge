#!/usr/bin/env bash
# Block 2: ECG Student — Structural Signal Recovery (DeepSpeed)
# Run from project root: bash run_block2.sh [NUM_GPUS]
#
# Examples:
#   bash run_block2.sh          # auto-detect GPUs
#   bash run_block2.sh 4        # use 4 GPUs
#   bash run_block2.sh 1        # single GPU
#
# Multi-node (set MASTER_ADDR, MASTER_PORT, HOSTFILE):
#   HOSTFILE=hostfile bash run_block2.sh
#
# Prerequisites:
#   - Block 1 complete (results/block1/ exists)
#   - Data prepared: python -m src.block2.prepare_data
#   - pip install deepspeed
set -euo pipefail
cd "$(dirname "$0")"

# Auto-detect number of GPUs if not specified
NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "==== Step 1: Prepare data (PTB-XL + UKB paired) ===="
# python -m src.block2.prepare_data

echo ""
echo "==== Step 2: Train + Evaluate all student variants (DeepSpeed, ${NUM_GPUS} GPUs) ===="

# Build deepspeed launch command
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
    -m src.block2.train_and_evaluate \
    --deepspeed_config src/block2/ds_config.json

echo ""
echo "==== Done ===="
echo "Results in results/block2/"
echo "Key file: results/block2/block2_results.json"
