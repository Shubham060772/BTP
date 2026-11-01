#!/usr/bin/env bash
# Run MoEFuse experiments on Kaggle — can run in parallel with other variants.
#
# MoEFuse: 3-expert mixture-of-experts with a learned router that selects
# per-user blending of seq+graph representations, plus residual to graph.
#
# Prerequisites: lightgcn.pt and sasrec.pt must exist in the repo (run baselines first).
#
# Usage (in a Kaggle notebook cell):
#   !SEEDS="42" EPOCHS=100 bash run_kaggle_moefuse.sh book
#   !SEEDS="42" GIT_PUSH=true GIT_TOKEN=$GIT_TOKEN bash run_kaggle_moefuse.sh book

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-book}"

# Only train moefuse — lightgcn/sasrec load from checkpoints for initialization
export MODELS="${MODELS:-lightgcn sasrec moefuse}"

exec bash "$SCRIPT_DIR/run_kaggle.sh" "$DATASET" "${@:2}"
