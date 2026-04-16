#!/usr/bin/env bash
# P0: CMR-side incident HF survival analysis (gating test)
# Run from project root: bash run_p0.sh
#
# Prerequisites:
#   - Block 1 complete (results/block1/predictions/*.csv)
#   - UKB .rds files in data/UKB/
#   - pip install lifelines pyreadr
#
# Output:
#   - results/block1/p0_survival_results.json
#   - results/block1/p0_survival_cohort.csv
#   - results/block1/figures/km_*.png, forest_plot_adjusted.png
set -euo pipefail
cd "$(dirname "$0")"

echo "==== P0: Incident HF Survival Analysis ===="
echo ""
echo "This is the GATING TEST for the project."
echo "If FT deviation C-index > baseline age-gap C-index → proceed to Block 2."
echo ""

python scripts/p0_survival_analysis.py

echo ""
echo "==== P0 Complete ===="
echo "Key output: results/block1/p0_survival_results.json"
echo ""
echo "Decision rule:"
echo "  - FT deviation adjusted C-index > BB age-gap adjusted C-index → PROCEED"
echo "  - FT deviation HR significant (p<0.05) after adjusting age, sex, BMI → PROCEED"
echo "  - Otherwise → reconsider Block 1 architecture"
