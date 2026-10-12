#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AdaFuse — Amazon-Book Experiment Pipeline
# ═══════════════════════════════════════════════════════════════
#
# Dataset: Amazon-Book (52,639 users, 91,599 items, 2.3M train)
# Sequence: synthetic (item list order from train.txt)
#
# Run from: B:\projects\BTP  (the project root)
# Usage:    bash run_book_experiments.sh
#
# KEY SETTINGS:
#   - sampled ranking (100 candidates) — full-item ranking infeasible at 91K items
#   - max-eval-users 5000 — subsample eval for speed
#   - max-train-samples 50000 — cap BPR/sequence pairs per epoch
#     (graph structure still uses ALL 2.3M interactions for propagation)
#     At 200K, SASRec alone takes ~4h on CPU; 50K brings it to ~30 min
#
# Estimated runtime: ~2.5 hours per seed, ~12 hours total (5 seeds)
#
# OUTPUT LAYOUT:
#   results/book/
#   ├── seed_{42,123,456,789,1024}/
#   │   ├── summary.json, results.pkl, training.log, checkpoints/
#   └── comparison/
#       ├── comparison.json, main_results.csv
#
# ═══════════════════════════════════════════════════════════════

set -eo pipefail

PROJECT="B:/projects/BTP"
PYTHON="$PROJECT/venv/Scripts/python.exe"
DATA="$PROJECT/roadmap/data/amazon-book/amazon_book.pkl"
RUN_SCRIPT="$PROJECT/src/run_beauty.py"
AGG_SCRIPT="$PROJECT/src/aggregate_beauty_results.py"
LOGDIR="$PROJECT/results/logs"
FINAL_DIR="$PROJECT/results/book"

mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/book_experiment_run.log"
: > "$LOGFILE"

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

COMMON="--epochs 50 --patience 15 --eval-every 5 --batch-size 2048 --ranking sampled --max-eval-users 5000 --max-train-samples 50000"
LR="1e-3"

ts "═══════════════════════════════════════════"
ts "  AdaFuse — Amazon-Book Experiments — START"
ts "═══════════════════════════════════════════"
ts "Data: $DATA"
ts "Python: $PYTHON"
ts "LR: $LR"
ts "Settings: $COMMON"
ts ""

if [ ! -f "$DATA" ]; then
    ts "ERROR: Data not found at $DATA"
    ts "Run: $PYTHON src/prepare_amazon_book.py"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# Run all 5 seeds
# ═══════════════════════════════════════════════════════════════

for SEED in 42 123 456 789 1024; do
    SEED_DIR="$FINAL_DIR/seed_$SEED"

    if [ -f "$SEED_DIR/results.pkl" ]; then
        ts "Seed $SEED already done, skipping."
        continue
    fi

    ts "--- Seed $SEED (started $(date '+%H:%M')) ---"

    # Run without piping to tee to avoid masking exit codes.
    # The Tee class inside run_beauty.py already logs to training.log.
    "$PYTHON" "$RUN_SCRIPT" \
        --data "$DATA" \
        --output-dir "$FINAL_DIR" \
        --seed "$SEED" \
        --learning-rate "$LR" \
        $COMMON \
        >> "$LOGFILE" 2>&1

    ts "Seed $SEED done at $(date '+%H:%M')."
    ts ""

    # Print seed results
    SUMMARY="$SEED_DIR/summary.json"
    if [ -f "$SUMMARY" ]; then
        "$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$SUMMARY').read_text())
agg = d['aggregate']
print(f\"{'Model':<14} {'Recall@10':>10} {'NDCG@10':>10} {'Recall@20':>10} {'NDCG@20':>10}\")
print('-' * 58)
for model in ['popularity','mf','lightgcn','sasrec','concat','sasgcn','adafuse']:
    if model in agg:
        m = agg[model]
        print(f\"{model:<14} {m['Recall@10']:10.4f} {m['NDCG@10']:10.4f} {m['Recall@20']:10.4f} {m['NDCG@20']:10.4f}\")
" 2>&1 | tee -a "$LOGFILE"
    fi
done

# ═══════════════════════════════════════════════════════════════
# Aggregate & Report
# ═══════════════════════════════════════════════════════════════
ts ""
ts "═══ Aggregate results ═══"

"$PYTHON" "$AGG_SCRIPT" \
    --results-dir "$FINAL_DIR" \
    --seeds 42 123 456 789 1024 \
    >> "$LOGFILE" 2>&1

ts ""
ts "═══ FINAL RESULTS (5-seed mean ± std) ═══"
COMPARISON="$FINAL_DIR/comparison/comparison.json"
if [ -f "$COMPARISON" ]; then
    "$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$COMPARISON').read_text())
agg = d['aggregate']
print(f\"{'Model':<14} {'Recall@10':>18} {'NDCG@10':>18} {'NDCG@20':>18}\")
print('=' * 72)
for model in ['popularity','mf','lightgcn','sasrec','concat','sasgcn','adafuse']:
    if model in agg:
        m = agg[model]
        def fmt(k):
            return f\"{m[k]['mean']:.4f} ± {m[k]['std']:.4f}\"
        print(f\"{model:<14} {fmt('Recall@10'):>18} {fmt('NDCG@10'):>18} {fmt('NDCG@20'):>18}\")

tests = d.get('paired_tests_adafuse_vs_baseline', {})
print()
print('AdaFuse vs. baseline paired tests (p-values):')
print(f\"{'Baseline':<14} {'Recall@10':>12} {'NDCG@10':>12} {'NDCG@20':>12}\")
print('-' * 54)
for baseline in ['mf','lightgcn','sasrec','sasgcn']:
    if baseline in tests:
        t = tests[baseline]
        def pval(k):
            p = t[k]['p_value']
            sig = '*' if p < 0.05 else ' '
            return f\"{p:.4f}{sig}\"
        print(f\"{baseline:<14} {pval('Recall@10'):>12} {pval('NDCG@10'):>12} {pval('NDCG@20'):>12}\")
print('(* = p < 0.05)')

alpha = d.get('alpha_analysis', {})
if 'pearson_r' in alpha:
    print()
    print(f\"Alpha-activity correlation: r={alpha['pearson_r']:.3f} (p={alpha['pearson_p']:.2e})\")
    for grp, vals in alpha.get('groups', {}).items():
        if vals.get('count', 0) > 0 and vals.get('mean') is not None:
            print(f\"  {grp:>8} ({vals['count']:>4} users): alpha = {vals['mean']:.4f} ± {vals['std']:.4f}\")
" 2>&1 | tee -a "$LOGFILE"
fi

ts ""
ts "═══════════════════════════════════════════"
ts "  DONE. Key outputs:"
ts "  Full run:   results/book/seed_*/summary.json"
ts "  Comparison: results/book/comparison/comparison.json"
ts "  CSV table:  results/book/comparison/main_results.csv"
ts "  Full log:   results/logs/book_experiment_run.log"
ts "═══════════════════════════════════════════"
