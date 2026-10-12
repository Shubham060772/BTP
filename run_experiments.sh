#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AdaFuse — Full Experiment Pipeline
# ═══════════════════════════════════════════════════════════════
#
# Run from: B:\projects\BTP  (the project root)
# Usage:    bash run_experiments.sh
#
# This script runs in 3 phases:
#   Phase 1 — LR tuning on 3-core data, seed 42 only
#   Phase 2 — Full 5-seed run with best LR
#   Phase 3 — Aggregate results and print report
#
# OUTPUT LAYOUT:
#   results/
#   ├── tuning/                          ← Phase 1
#   │   ├── lr1e-3/seed_42/summary.json  ← LR=0.001 results
#   │   ├── lr5e-4/seed_42/summary.json  ← LR=0.0005 results
#   │   └── lr1e-4/seed_42/summary.json  ← LR=0.0001 results
#   ├── beauty_3core/                    ← Phase 2
#   │   ├── seed_42/summary.json
#   │   ├── seed_123/summary.json
#   │   ├── seed_456/summary.json
#   │   ├── seed_789/summary.json
#   │   ├── seed_1024/summary.json
#   │   └── comparison/                  ← Phase 3 aggregation
#   │       ├── comparison.json          ← full stats + significance tests
#   │       ├── main_results.csv         ← results table
#   │       └── REPORT.md               ← (if generated)
#   └── logs/
#       └── experiment_run.log           ← this script's log
#
# Each seed directory also contains:
#   training.log   — per-epoch loss and val metrics
#   results.pkl    — per-user scores for significance testing
#   checkpoints/   — best model weights per model
#
# ═══════════════════════════════════════════════════════════════

set -e

PROJECT="B:/projects/BTP"
PYTHON="$PROJECT/venv/Scripts/python.exe"
DATA_3CORE="$PROJECT/roadmap/data/amazon-beauty/processed/beauty_3core.pkl"
RUN_SCRIPT="$PROJECT/src/run_beauty.py"
AGG_SCRIPT="$PROJECT/src/aggregate_beauty_results.py"
LOGDIR="$PROJECT/results/logs"

mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/experiment_run.log"

# Helper: print with timestamp to both terminal and log
ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

# Common training flags
COMMON="--epochs 100 --patience 20 --eval-every 5 --batch-size 512 --ranking full"

ts "═══════════════════════════════════════════"
ts "  AdaFuse Experiment Pipeline — START"
ts "═══════════════════════════════════════════"
ts "Data: $DATA_3CORE"
ts "Python: $PYTHON"
ts ""

# ─── Verify prerequisites ────────────────────────────────────
if [ ! -f "$DATA_3CORE" ]; then
    ts "ERROR: 3-core data not found. Run prepare first:"
    ts "  $PYTHON src/prepare_amazon_beauty.py --min-interactions 3"
    exit 1
fi
if [ ! -f "$PYTHON" ]; then
    ts "ERROR: Python venv not found at $PYTHON"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Hyperparameter Tuning (seed 42 only, 3 LR values)
# ═══════════════════════════════════════════════════════════════
#
# Output: results/tuning/lr{X}/seed_42/summary.json
# Check:  grep "adafuse" in each summary.json → pick best val NDCG@10
#
ts ""
ts "═══ PHASE 1: LR Tuning (seed 42, 3-core) ═══"

for LR in 1e-3 5e-4 1e-4; do
    OUTDIR="$PROJECT/results/tuning/lr${LR}"
    ts "--- LR=$LR → $OUTDIR ---"

    "$PYTHON" "$RUN_SCRIPT" \
        --data "$DATA_3CORE" \
        --output-dir "$OUTDIR" \
        --seed 42 \
        --learning-rate "$LR" \
        $COMMON \
        2>&1 | tee -a "$LOGFILE"

    ts "LR=$LR done."
    ts ""
done

# ─── Pick best LR ────────────────────────────────────────────
ts "═══ PHASE 1 RESULTS: AdaFuse NDCG@10 per LR ═══"
BEST_LR=""
BEST_NDCG="-1"

for LR in 1e-3 5e-4 1e-4; do
    SUMMARY="$PROJECT/results/tuning/lr${LR}/seed_42/summary.json"
    if [ -f "$SUMMARY" ]; then
        NDCG=$("$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$SUMMARY').read_text())
print(d['aggregate']['adafuse']['NDCG@10'])
" 2>/dev/null || echo "0")
        ts "  LR=$LR  →  AdaFuse NDCG@10 = $NDCG"

        # Compare (use python for float comparison)
        IS_BETTER=$("$PYTHON" -c "print(1 if float('$NDCG') > float('$BEST_NDCG') else 0)" 2>/dev/null)
        if [ "$IS_BETTER" = "1" ]; then
            BEST_LR="$LR"
            BEST_NDCG="$NDCG"
        fi
    else
        ts "  LR=$LR  →  MISSING (run failed?)"
    fi
done

if [ -z "$BEST_LR" ]; then
    ts "ERROR: No tuning run succeeded. Check logs above."
    exit 1
fi

ts ""
ts "★ Best LR = $BEST_LR (AdaFuse NDCG@10 = $BEST_NDCG)"
ts ""

# Also show ALL models for the best LR so we see the full picture
ts "Full results for best LR ($BEST_LR):"
BEST_SUMMARY="$PROJECT/results/tuning/lr${BEST_LR}/seed_42/summary.json"
"$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$BEST_SUMMARY').read_text())
agg = d['aggregate']
print(f\"{'Model':<14} {'Recall@10':>10} {'NDCG@10':>10} {'Recall@20':>10} {'NDCG@20':>10}\")
print('-' * 58)
for model in ['popularity','mf','lightgcn','sasrec','concat','sasgcn','adafuse']:
    if model in agg:
        m = agg[model]
        print(f\"{model:<14} {m['Recall@10']:10.4f} {m['NDCG@10']:10.4f} {m['Recall@20']:10.4f} {m['NDCG@20']:10.4f}\")
" 2>&1 | tee -a "$LOGFILE"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Full 5-Seed Run with Best LR
# ═══════════════════════════════════════════════════════════════
#
# Output: results/beauty_3core/seed_{42,123,456,789,1024}/
#
ts ""
ts "═══ PHASE 2: Full 5-seed run (LR=$BEST_LR, 3-core) ═══"
FINAL_DIR="$PROJECT/results/beauty_3core"

for SEED in 42 123 456 789 1024; do
    SEED_DIR="$FINAL_DIR/seed_$SEED"

    # Skip if already completed
    if [ -f "$SEED_DIR/results.pkl" ]; then
        ts "Seed $SEED already done, skipping."
        continue
    fi

    ts "--- Seed $SEED ---"

    "$PYTHON" "$RUN_SCRIPT" \
        --data "$DATA_3CORE" \
        --output-dir "$FINAL_DIR" \
        --seed "$SEED" \
        --learning-rate "$BEST_LR" \
        $COMMON \
        2>&1 | tee -a "$LOGFILE"

    ts "Seed $SEED done."
    ts ""
done

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Aggregate & Report
# ═══════════════════════════════════════════════════════════════
#
# Output: results/beauty_3core/comparison/{comparison.json, main_results.csv}
#
ts ""
ts "═══ PHASE 3: Aggregate results ═══"

"$PYTHON" "$AGG_SCRIPT" \
    --results-dir "$FINAL_DIR" \
    --seeds 42 123 456 789 1024 \
    2>&1 | tee -a "$LOGFILE"

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

# Significance tests
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

# Alpha
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
ts "  Tuning:     results/tuning/lr*/seed_42/summary.json"
ts "  Full run:   results/beauty_3core/seed_*/summary.json"
ts "  Comparison: results/beauty_3core/comparison/comparison.json"
ts "  CSV table:  results/beauty_3core/comparison/main_results.csv"
ts "  Full log:   results/logs/experiment_run.log"
ts "═══════════════════════════════════════════"
