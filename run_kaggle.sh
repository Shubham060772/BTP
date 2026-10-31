#!/usr/bin/env bash
# Run AdaFuse experiments directly from a Kaggle notebook terminal/cell.
#
# Usage:
#   !bash run_kaggle.sh beauty
#   !bash run_kaggle.sh beauty /kaggle/input/<dataset>/All_Beauty.jsonl
#   !bash run_kaggle.sh book /kaggle/input/<dataset-containing-train-test>
#
# Optional overrides:
#   SEEDS="42 123" EPOCHS=50 BATCH_SIZE=1024 CORE=3 bash run_kaggle.sh beauty
#   MODELS="popularity mf lightgcn" bash run_kaggle.sh book
#   ALPHA_REG=0.1 GRAPH_LOSS_WEIGHT=0.3 bash run_kaggle.sh beauty

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-beauty}"
INPUT_PATH="${2:-}"
WORK_DIR="${KAGGLE_WORK_DIR:-/kaggle/working/adafuse}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS_TEXT="${SEEDS:-42 123 456 789 1024}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
ALPHA_REG="${ALPHA_REG:-0.0}"
GRAPH_LOSS_WEIGHT="${GRAPH_LOSS_WEIGHT:-0.0}"
SEQ_LAYERS="${SEQ_LAYERS:-2}"
GRAPH_ENRICHED="${GRAPH_ENRICHED:-false}"
LENGTH_AWARE="${LENGTH_AWARE:-false}"
# PATIENCE and PRETRAIN_EPOCHS have dataset-specific defaults (set in case block below)

mkdir -p "$WORK_DIR/data" "$WORK_DIR/results" "$WORK_DIR/logs"
LOG_FILE="$WORK_DIR/logs/kaggle_${DATASET}.log"
touch "$LOG_FILE"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

fail() {
  log "ERROR: $*"
  exit 1
}

find_first() {
  local name="$1"
  find /kaggle/input -type f -name "$name" -print -quit 2>/dev/null || true
}

download_beauty() {
  local destination="$1"
  local url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl?download=true"
  mkdir -p "$(dirname "$destination")"
  log "No attached Beauty file found; downloading the official dataset"
  log "Source: McAuley-Lab/Amazon-Reviews-2023"
  if command -v curl >/dev/null 2>&1; then
    if ! curl --location --fail --retry 3 --retry-delay 5 \
      --continue-at - --output "$destination" "$url"; then
      fail "Download failed. Enable Internet in Kaggle or attach All_Beauty.jsonl as notebook input."
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget --continue --output-document="$destination" "$url"; then
      fail "Download failed. Enable Internet in Kaggle or attach All_Beauty.jsonl as notebook input."
    fi
  else
    fail "Neither curl nor wget is installed; attach All_Beauty.jsonl as notebook input."
  fi
  [[ -s "$destination" ]] || fail "The downloaded Beauty file is empty."
}

log "Project: $PROJECT_DIR"
log "Dataset: $DATASET"
log "Outputs: $WORK_DIR"

if ! "$PYTHON_BIN" -c "import numpy, scipy, torch" >/dev/null 2>&1; then
  log "Installing Python dependencies"
  "$PYTHON_BIN" -m pip install --quiet -r "$PROJECT_DIR/requirements.txt"
fi

"$PYTHON_BIN" - <<'PY' | tee -a "$LOG_FILE"
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Kaggle GPU is disabled; enable a GPU accelerator before the full run.")
PY

case "$DATASET" in
  beauty)
    # Three interactions are the mathematical minimum for one train, one
    # validation, and one test event. This retains the largest defensible
    # timestamped leave-two-out dataset.
    PATIENCE="${PATIENCE:-20}"
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-0}"
    CORE="${CORE:-3}"
    BATCH_SIZE="${BATCH_SIZE:-1024}"
    RANKING="full"
    RESULT_DIR="$WORK_DIR/results/beauty_${CORE}core"
    PROCESSED_DATA="$WORK_DIR/data/beauty_${CORE}core.pkl"
    if [[ -z "$INPUT_PATH" ]]; then
      INPUT_PATH="$(find_first 'All_Beauty.jsonl')"
    fi
    if [[ -z "$INPUT_PATH" ]]; then
      INPUT_PATH="$WORK_DIR/data/All_Beauty.jsonl"
      download_beauty "$INPUT_PATH"
    fi
    [[ -f "$INPUT_PATH" ]] || fail "Beauty input is not a file: $INPUT_PATH"
    if [[ ! -f "$PROCESSED_DATA" ]]; then
      log "Preparing timestamped Beauty ${CORE}-core data from every JSONL row"
      "$PYTHON_BIN" "$PROJECT_DIR/src/prepare_amazon_beauty.py" \
        --input "$INPUT_PATH" \
        --output "$PROCESSED_DATA" \
        --min-interactions "$CORE" 2>&1 | tee -a "$LOG_FILE"
    else
      log "Using existing processed data: $PROCESSED_DATA"
    fi
    DEFAULT_MODELS="popularity mf lightgcn sasrec sasgcn concat adafuse"
    ;;

  book)
    # Book needs more epochs — MF/SASRec don't converge in 30 epochs on 2.3M interactions.
    # PATIENCE=30 gives models more time; PRETRAIN_EPOCHS=20 stabilises the AdaFuse gate.
    PATIENCE="${PATIENCE:-30}"
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
    BATCH_SIZE="${BATCH_SIZE:-4096}"
    RANKING="sampled"
    RESULT_DIR="$WORK_DIR/results/book"
    PROCESSED_DATA="$WORK_DIR/data/amazon_book.pkl"
    # Check for pkl in Kaggle input or the repo's own data directory
    if [[ ! -f "$PROCESSED_DATA" ]]; then
      EXISTING_PKL="$(find /kaggle/input -type f -name 'amazon_book.pkl' -print -quit 2>/dev/null || true)"
      [[ -z "$EXISTING_PKL" ]] && EXISTING_PKL="${PROJECT_DIR}/roadmap/data/amazon-book/amazon_book.pkl"
      if [[ -f "$EXISTING_PKL" ]]; then
        log "Found pre-processed pkl: $EXISTING_PKL"
        cp "$EXISTING_PKL" "$PROCESSED_DATA"
      fi
    fi
    if [[ ! -f "$PROCESSED_DATA" ]]; then
      if [[ -z "$INPUT_PATH" ]]; then
        TRAIN_FILE="$(find_first 'train.txt')"
        [[ -n "$TRAIN_FILE" ]] && INPUT_PATH="$(dirname "$TRAIN_FILE")"
      fi
      [[ -n "$INPUT_PATH" && -f "$INPUT_PATH/train.txt" && -f "$INPUT_PATH/test.txt" ]] || fail \
        "Amazon-Book data not found. Upload amazon_book.pkl or train.txt+test.txt as Kaggle input."
      log "WARNING: Amazon-Book has no timestamps. SASRec/SASGCN/AdaFuse use synthetic list order."
      "$PYTHON_BIN" "$PROJECT_DIR/src/prepare_amazon_book.py" \
        --data-dir "$INPUT_PATH" \
        --output "$PROCESSED_DATA" 2>&1 | tee -a "$LOG_FILE"
    else
      log "Using existing processed data: $PROCESSED_DATA"
    fi
    DEFAULT_MODELS="popularity mf lightgcn sasrec sasgcn concat adafuse"
    ;;

  *)
    fail "dataset must be 'beauty' or 'book'"
    ;;
esac

MODELS_TEXT="${MODELS:-$DEFAULT_MODELS}"
read -r -a SEED_ARRAY <<< "$SEEDS_TEXT"
read -r -a MODEL_ARRAY <<< "$MODELS_TEXT"
mkdir -p "$RESULT_DIR"

log "Seeds: $SEEDS_TEXT"
log "Models: $MODELS_TEXT"
log "Fusion: alpha_reg=$ALPHA_REG, graph_loss_weight=$GRAPH_LOSS_WEIGHT, seq_layers=$SEQ_LAYERS, graph_enriched=$GRAPH_ENRICHED, length_aware=$LENGTH_AWARE, pretrain_epochs=$PRETRAIN_EPOCHS"
log "Training is uncapped: every eligible training instance and evaluation user is used."

for seed in "${SEED_ARRAY[@]}"; do
  SEED_DIR="$RESULT_DIR/seed_${seed}"
  if [[ -f "$SEED_DIR/results.pkl" ]]; then
    log "Seed $seed is complete; skipping"
    continue
  fi

  log "Starting seed $seed"
  # Copy baseline checkpoints from repo if present (saves retraining MF/LightGCN/SASRec)
  REPO_CKPT_DIR="$PROJECT_DIR/results/${DATASET}/seed_${seed}/checkpoints"
  if [[ -d "$REPO_CKPT_DIR" ]]; then
    mkdir -p "$SEED_DIR/checkpoints"
    for ckpt in mf lightgcn sasrec; do
      src="$REPO_CKPT_DIR/${ckpt}.pt"
      dst="$SEED_DIR/checkpoints/${ckpt}.pt"
      if [[ -f "$src" && ! -f "$dst" ]]; then
        cp "$src" "$dst"
        log "Copied baseline checkpoint: ${ckpt}.pt (seed $seed)"
      fi
    done
  fi
  RESUME_ARGS=()
  if [[ -d "$SEED_DIR/checkpoints" ]]; then
    RESUME_ARGS=(--resume)
    log "Resuming seed $seed from available model checkpoints"
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/src/run_beauty.py" \
    --data "$PROCESSED_DATA" \
    --output-dir "$RESULT_DIR" \
    --seed "$seed" \
    --embedding-dim 64 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --eval-every "$EVAL_EVERY" \
    --patience "$PATIENCE" \
    --learning-rate "$LEARNING_RATE" \
    --alpha-reg "$ALPHA_REG" \
    --graph-loss-weight "$GRAPH_LOSS_WEIGHT" \
    --seq-layers "$SEQ_LAYERS" \
    $( [[ "$GRAPH_ENRICHED" == "true" ]] && echo "--graph-enriched" ) \
    $( [[ "$LENGTH_AWARE" == "true" ]] && echo "--length-aware" ) \
    --pretrain-epochs "$PRETRAIN_EPOCHS" \
    --ranking "$RANKING" \
    --models "${MODEL_ARRAY[@]}" \
    "${RESUME_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "Completed seed $seed"
done

log "Aggregating seed results"
"$PYTHON_BIN" "$PROJECT_DIR/src/aggregate_beauty_results.py" \
  --results-dir "$RESULT_DIR" \
  --seeds "${SEED_ARRAY[@]}" 2>&1 | tee -a "$LOG_FILE"

log "Done"
log "Results: $RESULT_DIR/comparison"
log "Log: $LOG_FILE"


