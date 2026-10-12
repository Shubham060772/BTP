# Plan vs. Reality — AdaFuse Implementation Analysis
**BTP: Customer Product Recommendation System Using Transformers and GNNs**
**IIT Patna | October 2026**

---

## 1. What the Plan Promised

The `REVISED_RESEARCH_PLAN.md` set out the following targets:

| Dimension | Plan Target |
|-----------|-------------|
| **Dataset (Beauty)** | ~52K users, ~57K items, ~400K interactions (10-core) |
| **Models** | Popularity, MF-BPR, LightGCN, SASRec, BERT4Rec, SASGCN, AdaFuse (7 total) |
| **Evaluation** | Full-item ranking, 5 seeds [42,123,456,789,1024], mean ± std |
| **Hyperparameter tuning** | Grid search over LR ∈ {1e-3,5e-4,1e-4}, dropout ∈ {0.2,0.5}, layers, batch size |
| **AdaFuse expected Recall@10** | ~0.054 (5-8% over SASGCN ~0.050) |
| **Alpha range hypothesis** | 0.3–0.4 for sparse users, 0.6–0.7 for active users |
| **Extra experiments** | Sparsity robustness (100/75/50/25%), t-SNE embeddings, ablation (6 variants) |

---

## 2. What Was Actually Implemented

### 2.1 Data Pipeline

**Implemented ✓**
- `src/prepare_amazon_beauty.py` — streams `All_Beauty.jsonl`, deduplicates per user-product pair, runs iterative k-core filtering, performs temporal leave-two-out split (penultimate → val, last → test), serializes to `roadmap/data/amazon-beauty/processed/beauty_5core.pkl`
- Correct temporal ordering (timestamp sort before split)
- Bipartite graph construction with bidirectional edges and D⁻½AD⁻½ normalization

**Problem discovered during execution:**
- The plan assumed Amazon Reviews 2023 All_Beauty would yield ~52K users under 10-core filtering
- Reality: the file collapsed to **zero users under 10-core**
- At 5-core: only **253 users, 356 items, 2,535 interactions**
- This is ~200× smaller than planned, making statistical conclusions very noisy

**Root cause:** The plan's size estimates were based on the older Amazon Reviews 2018 Beauty dataset (often cited as "22,363 users" in prior work). The 2023 refresh has a different user/item distribution in the `All_Beauty` category — far more sparse after deduplication.

### 2.2 Models Implemented

| Model | Implemented | Notes |
|-------|-------------|-------|
| Popularity | ✓ | Item frequency baseline |
| MF-BPR | ✓ | Dot-product embedding + BPR |
| LightGCN | ✓ | 3-layer sparse propagation, mean combination |
| SASRec | ✓ | 2-layer causal Transformer, positional embeddings |
| SASGCN-style | ✓ | Vanilla GCN + SASRec + trainable concat+MLP fusion |
| Frozen concat ablation | ✓ | Frozen LightGCN + SASRec encoders + concat+MLP |
| AdaFuse | ✓ | LightGCN + SASRec + scalar attention gate α |
| **BERT4Rec** | ✗ | Not implemented — was in plan but skipped |

### 2.3 Evaluation Protocol

**Implemented correctly ✓**
- Full eligible-item ranking (not sampled negatives)
- Interacted-item masking during ranking
- 5 seeds [42, 123, 456, 789, 1024]
- Paired significance tests (averaging user scores over seeds before testing across users — correct, avoids pseudoreplication)
- Metrics: Recall@10/20, NDCG@10/20, HR@10/20

**Not implemented ✗**
- No hyperparameter grid search (single fixed config used for all models)
- No sparsity robustness experiment (100/75/50/25%)
- No t-SNE embedding visualization
- No full ablation suite (6 variants) — only frozen concat run

---

## 3. Actual Results vs. Plan

### 3.1 Main Results (5-seed mean ± std, Amazon All Beauty 5-core, 253 users)

| Model | Recall@10 | NDCG@10 | NDCG@20 | vs Plan |
|-------|-----------|---------|---------|---------|
| Popularity | 0.0119 ± 0.00 | 0.0070 ± 0.00 | 0.0090 ± 0.00 | Plan: ~0.015 |
| MF-BPR | 0.0285 ± 0.012 | 0.0135 ± 0.004 | 0.0188 ± 0.004 | Plan: ~0.030 |
| LightGCN | 0.0213 ± 0.005 | 0.0083 ± 0.002 | 0.0158 ± 0.002 | Plan: ~0.042 |
| SASRec | 0.0206 ± 0.014 | 0.0092 ± 0.005 | 0.0140 ± 0.004 | Plan: ~0.045 |
| Frozen concat | 0.0126 ± 0.006 | 0.0056 ± 0.003 | 0.0108 ± 0.005 | Ablation |
| SASGCN-style | **0.0308 ± 0.009** | **0.0135 ± 0.005** | **0.0188 ± 0.004** | Plan: ~0.050 |
| **AdaFuse** | 0.0213 ± 0.009 | 0.0087 ± 0.005 | 0.0162 ± 0.002 | Plan: **~0.054** |

**Key finding:** AdaFuse did NOT outperform SASGCN-style. It roughly matched LightGCN. The hypothesis that adaptive attention fusion beats static fusion was not confirmed on this dataset.

### 3.2 Statistical Significance

AdaFuse vs. SASGCN: all four metrics show p > 0.12, not significant.
AdaFuse vs. frozen concat: Recall@20 (p=0.022) and NDCG@20 (p=0.045) are significant — AdaFuse does beat the frozen encoder ablation, showing that end-to-end training of the fusion helps.

### 3.3 Alpha Analysis

| User Group | Count | Mean α | Std |
|------------|-------|--------|-----|
| 1–5 interactions | 129 | 0.470 | 0.022 |
| 6–10 | 67 | 0.513 | 0.043 |
| 11–20 | 39 | 0.532 | 0.025 |
| 21–50 | 16 | 0.523 | 0.032 |
| 50+ | 2 | 0.559 | 0.002 |

Pearson r = 0.434, p = 4.6e-13 | Spearman ρ = 0.487, p = 1.8e-16

**Partially confirmed:** The correlation is statistically significant and in the right direction (active users → higher α → more weight on SASRec). However the effect size is small — alpha ranges only 0.47 to 0.56, not 0.3–0.7 as hypothesised. The gate is learning a meaningful signal but not differentiating as strongly as expected.

---

## 4. Root Cause Analysis — What Went Wrong

### Issue 1: Dataset Too Small (Critical)
**Problem:** 253 users is too few for training complex hybrid models. With one test item per user, evaluation variance is very high. Any model that correctly ranks 1 more user changes Recall@10 by 0.004 (1/253). This is the primary reason no model shows statistically significant improvement.
**Evidence:** Standard deviations are 30–70% of the mean values. Seed variance dominates signal.
**Fix:** Obtain the Amazon Reviews 2018 Beauty dataset (5-core version: ~22K users, ~12K items, ~198K interactions) which is the standard benchmark used by SASRec, LightGCN, and SASGCN papers.

### Issue 2: No Hyperparameter Tuning
**Problem:** AdaFuse and all baselines ran with a single fixed configuration (LR=1e-3, dropout=0.2, 3 GCN layers, 2 SASRec layers). The gate may need lower LR for stable training. SASGCN's trainable concat+MLP may converge faster at LR=1e-3 because it has more capacity in the fusion layer.
**Evidence:** In seed 42, AdaFuse early-stopped at epoch 20 (val NDCG@10=0.011 at best checkpoint) while SASGCN ran to epoch 30 (val NDCG@10=0.024). AdaFuse appears under-tuned.
**Fix:** Grid search over at minimum LR ∈ {1e-3, 5e-4, 1e-4} and dropout ∈ {0.2, 0.5} on validation NDCG@10.

### Issue 3: AdaFuse Training Instability
**Problem:** The gate network receives gradients through both the BPR loss and the embedding updates simultaneously. On a tiny dataset, this can cause the gate to converge to a trivial solution (α ≈ 0.5 for all users).
**Evidence:** Alpha range 0.47–0.56 shows the gate has not learned meaningfully different fusion strategies across users. The gate effectively collapsed to near-constant behavior.
**Fix:** (a) Add gate entropy regularization to encourage diverse α values; (b) Try two-stage training: pretrain LightGCN and SASRec branches separately, then fine-tune with the gate; (c) Lower LR for gate parameters while keeping higher LR for encoder parameters.

### Issue 4: run_all.py Was Invalidated
**Problem:** The original `src/run_all.py` (Amazon-Book experiments) had multiple bugs:
- Detached the graph embeddings during training (graph model could not receive gradients through BPR loss)
- Amazon-Book has no timestamps — treating file line order as an interaction sequence is questionable
- These results must not be used in the thesis
**Status:** Correctly identified and invalidated in `EXPERIMENTS.md`. The `run_beauty.py` pipeline is the valid one.

### Issue 5: Amazon-Book Results Incomplete
**Problem:** The Amazon-Book training run (run_all.py) was started but interrupted. The detached-graph bug was present for all GNN-based models in that script.
**Status:** Results partially captured: MF-BPR completed (Recall@10≈0.497, NDCG@10≈0.494 — inflated because of training data leakage in val split). LightGCN completed (Recall@10≈0.458). SASRec was training when session ended.
**Fix:** Invalidated — do not use. Re-run on valid dataset (Beauty 2018) with correct pipeline.

### Issue 6: Missing BERT4Rec and Ablation Variants
**Problem:** BERT4Rec (bidirectional Transformer) was in the plan but not implemented. The 6-variant ablation (no-GNN, no-Transformer, no-gate, no-element-wise, multi-head gate) was reduced to just one frozen-concat ablation.
**Fix:** These are lower priority. With 253 users, adding more baselines doesn't strengthen the paper. Priority is getting a larger dataset first.

---

## 5. What IS Valid and Useful

Despite the limitations, the following contributions are real:

1. **AdaFuse architecture is correctly implemented** — The scalar attention gate α(u) = σ(w^T[e_seq; e_col; e_seq⊙e_col]) is correctly coded, end-to-end trainable, and interpretable.

2. **Alpha-activity correlation is confirmed** — r=0.434, p<1e-12 across 253 users and 5 seeds. The direction is correct: active users put more weight on sequential representation. This is a genuine finding even if the effect is smaller than expected.

3. **AdaFuse beats frozen concat ablation** — Recall@20 (+0.018, p=0.022) and NDCG@20 (+0.005, p=0.045) are significant. This shows that end-to-end joint training of the fusion mechanism matters — you can't just freeze encoders and concatenate.

4. **Full reproducible pipeline** — `src/run_beauty.py` + `src/aggregate_beauty_results.py` is clean, reproducible, uses correct temporal split, full-item ranking, and proper significance testing. Ready to scale to a larger dataset.

5. **The core research question is still open** — The negative result on 253 users is not a disproof; it's a sample-size problem. The hypothesis remains testable on a proper-sized dataset.

---

## 6. Recommended Next Steps (Priority Order)

### Priority 1 — Get the Right Dataset (1 day)
Download Amazon Reviews 2018 Beauty (5-core version) from the McAuley lab. This is the version used in SASRec, LightGCN, and SASGCN papers. Expected size: ~22K users, ~12K items, ~198K interactions. This immediately makes results comparable to published numbers.

```
URL: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
Alternative: https://jmcauley.ucsd.edu/data/amazon/
```

### Priority 2 — Hyperparameter Tuning on New Dataset (2 days on GPU)
Run grid search on Amazon Beauty 2018:
- LR ∈ {1e-3, 5e-4, 1e-4}
- Dropout ∈ {0.2, 0.5}
- LightGCN layers ∈ {2, 3, 4}
- Select by val NDCG@10 on seed 42 only
- Then fix best config and run all 5 seeds

### Priority 3 — Two-Stage AdaFuse Training (optional improvement)
If AdaFuse still underperforms on the larger dataset:
- Stage 1: Pretrain SASRec alone (20 epochs)
- Stage 2: Pretrain LightGCN alone (20 epochs)
- Stage 3: Initialize AdaFuse from pretrained weights, train gate and fine-tune (40 epochs)
This avoids the instability of training all three components simultaneously from random init.

### Priority 4 — Alpha Analysis at Scale
With ~22K users (vs 253 now), alpha-activity correlation becomes much more convincing. Plot α distribution per user group and scatter plot α vs. |interactions|.

---

## 7. Summary Verdict

| Item | Status | Severity |
|------|--------|----------|
| AdaFuse architecture | Correctly implemented | — |
| Dataset sizing | Wrong version (200× too small) | **Critical** |
| Hyperparameter tuning | Not done | High |
| Main hypothesis (AdaFuse > SASGCN) | Not confirmed | Downstream of dataset issue |
| Alpha correlation | Partially confirmed | Positive finding |
| Reproducibility | Good (clean pipeline) | — |
| Comparison with paper numbers | Cannot compare (different dataset) | High |
| BERT4Rec baseline | Missing | Medium |
| Ablation suite | Minimal (1 of 6 variants) | Medium |
| Thesis/report | Not written | High |

**Bottom line:** The code is correct and the pipeline is sound. The experiment failed to confirm the main hypothesis primarily because the dataset collapsed to 253 users — a version mismatch between the plan and the actual file on disk. Switching to Amazon Reviews 2018 Beauty (the standard benchmark) and adding hyperparameter tuning are the two actions that would make the results publishable.
