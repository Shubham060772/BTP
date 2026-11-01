#!/usr/bin/env bash
# Run CrossAttnFuse experiments on Kaggle — can run in parallel with other variants.
#
# CrossAttnFuse: multi-head cross-attention between seq and graph representations
# with a per-dimension gated residual connection to graph embeddings.
#
# Usage (in a Kaggle notebook cell):
#   !SEEDS="42" EPOCHS=100 bash run_kaggle_crossattn.sh book
#   !SEEDS="42" GIT_PUSH=true GIT_TOKEN=$GIT_TOKEN bash run_kaggle_crossattn.sh book
#
# This trains baselines (from checkpoint if available) + the crossattn model.
# Run on a separate Kaggle session in parallel with run_kaggle_moefuse.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-book}"

export MODELS="${MODELS:-popularity mf lightgcn sasrec crossattn}"

exec bash "$SCRIPT_DIR/run_kaggle.sh" "$DATASET" "${@:2}"
