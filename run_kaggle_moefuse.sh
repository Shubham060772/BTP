#!/usr/bin/env bash
# Run MoEFuse experiments on Kaggle — can run in parallel with other variants.
#
# MoEFuse: 3-expert mixture-of-experts with a learned router that selects
# per-user blending of seq+graph representations, plus residual to graph.
#
# Usage (in a Kaggle notebook cell):
#   !SEEDS="42" EPOCHS=100 bash run_kaggle_moefuse.sh book
#   !SEEDS="42" GIT_PUSH=true GIT_TOKEN=$GIT_TOKEN bash run_kaggle_moefuse.sh book
#
# This trains baselines (from checkpoint if available) + the moefuse model.
# Run on a separate Kaggle session in parallel with run_kaggle_crossattn.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-book}"

export MODELS="${MODELS:-popularity mf lightgcn sasrec moefuse}"

exec bash "$SCRIPT_DIR/run_kaggle.sh" "$DATASET" "${@:2}"
