#!/usr/bin/env bash
# Block 1: Teacher Stability and Manifold Validity
# Run from project root: bash run_block1.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==== Step 1: Data preparation ===="
python -m src.block1.prepare_data

echo ""
echo "==== Step 2: Train + Evaluate all systems ===="
python -m src.block1.train_and_evaluate --device cpu

echo ""
echo "==== Step 3: Bootstrap stability (optional, slow) ===="
echo "To run: python -m src.block1.bootstrap --n_bootstrap 50 --device cpu"
echo "Skipping by default. Uncomment the next line to run."
python -m src.block1.bootstrap --n_bootstrap 50 --device cpu

echo ""
echo "==== Done ===="
echo "Results in results/block1/"
echo "Key file: results/block1/block1_results.json"
