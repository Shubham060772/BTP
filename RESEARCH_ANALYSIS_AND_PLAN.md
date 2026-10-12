# Research Analysis & Implementation Plan
## GNN-Based Recommendation Systems — BTP

---

## Part 1: Outcomes of the Most Recent & Relevant Papers

### 1.1 The SOTA Landscape on Amazon-Book (Primary Benchmark)

| Method | Year | Venue | Recall@20 | NDCG@20 | Key Idea |
|--------|------|-------|-----------|---------|----------|
| NCF | 2017 | WWW | 0.0142 | 0.0085 | Neural CF with MLP |
| NGCF | 2019 | SIGIR | ~0.030 | ~0.022 | GCN for CF (feature transforms + nonlinearity) |
| LightGCN | 2020 | SIGIR | 0.0319 | 0.0236 | Stripped GCN — only aggregation, no transforms |
| SGL | 2021 | SIGIR | 0.0327 | 0.0249 | Self-supervised graph augmentation (node/edge drop) |
| DirectAU | 2022 | KDD | ~0.035 | ~0.027 | Alignment + Uniformity loss (no augmentation needed) |
| SimGCL | 2022 | SIGIR | 0.0474 | 0.0360 | Noise-based contrastive (no graph augmentation) |
| SHT | 2022 | — | 0.0441 | 0.0328 | Hypergraph Transformer |
| XSimGCL | 2023 | TKDE | ~0.049 | ~0.037 | Simplified SimGCL with layer-level contrast |
| **LightGCL** | **2023** | **ICLR** | **0.0585** | **0.0436** | **SVD-based spectral augmentation** |

**LightGCL is the current undisputed SOTA on Amazon-Book**, achieving ~83% improvement over LightGCN and ~23% over SimGCL.

### 1.2 Recent Papers (2024-2026) and Their Outcomes

#### Paper 1: LightGCL (Cai et al., ICLR 2023) — Current SOTA
- **What it does:** Replaces stochastic graph augmentations (node drop, edge drop) with a deterministic SVD-based augmentation. Decomposes the adjacency matrix via truncated SVD, reconstructs a "clean" augmented graph, runs LightGCN on both original and SVD-augmented graphs, and applies InfoNCE contrastive loss between the two views.
- **Key outcome:** Massive gains over all prior methods. SVD captures global collaborative patterns that local augmentations miss.
- **Limitations we can exploit:**
  - **(L1) Single augmentation view** — only uses SVD structural augmentation. SimGCL showed that embedding-level noise is also powerful. These two views are complementary (structural vs. embedding-space) but nobody has combined them.
  - **(L2) Uniform spectral treatment** — all singular values are treated equally. But high singular values capture popularity patterns while low ones capture niche preferences. Frequency-adaptive weighting is unexplored.
  - **(L3) No metadata** — purely collaborative signal. Ignores item attributes (category, brand, price) that could help in cold-start / sparse scenarios.

#### Paper 2: SimGCL (Yu et al., SIGIR 2022)
- **What it does:** Instead of augmenting the graph structure, adds uniform random noise to node embeddings as a contrastive view. Surprisingly effective and much simpler than SGL's structure augmentations.
- **Key outcome:** Recall@20 = 0.0474. Proved that you don't need graph augmentation at all — embedding perturbation works.
- **Limitations we can exploit:**
  - **(L1) No structural awareness** — noise is applied uniformly regardless of graph topology. High-degree nodes and low-degree nodes get same noise magnitude.
  - **(L2) Noise magnitude is a single hyperparameter** — doesn't adapt to node importance or embedding quality.

#### Paper 3: XSimGCL (Yu et al., TKDE 2023)
- **What it does:** Simplifies SimGCL further by applying noise at a specific GNN layer and contrasting layer-level representations.
- **Key outcome:** Marginal improvement over SimGCL, but much simpler. Shows that layer-level contrastive signals carry useful information.
- **Limitation:** Still single-view (noise only), no spectral/structural component.

#### Paper 4: GFormer (Li et al., SIGIR 2023)
- **What it does:** Graph Transformer for recommendation — applies self-attention over graph neighborhoods with learned positional encodings.
- **Key outcome:** Competitive but not SOTA. O(n²) attention limits scalability.
- **Limitation:** Doesn't scale; contrastive methods dominate it on large datasets.

#### Paper 5: FourierKAN-GCF (CIKM 2025)
- **What it does:** Replaces linear transformations in GCN with Fourier Kolmogorov-Arnold Networks — learnable periodic activations.
- **Key outcome:** Adds lightweight non-linearity to LightGCN-style architectures. Early results promising.
- **Limitation:** Very recent, no contrastive learning integration. Orthogonal improvement direction.

#### Paper 6: DGSARec (arXiv 2605, 2026 preprint)
- **What it does:** Dynamic graph with similarity-aware attention for sequential recommendation.
- **Limitation:** Preprint, not peer-reviewed. Sequential setting different from our collaborative filtering setup.

#### Paper 7: ARGC (arXiv 2602, 2026 preprint)
- **What it does:** LLM-augmented adaptive relation-weighted graph convolution.
- **Limitation:** Requires LLM inference, heavy compute. Preprint.

---

## Part 2: What We Can Improve — The Research Gaps

After analyzing all 30+ papers across your study materials, here are the **confirmed, actionable gaps** ranked by feasibility × novelty × expected impact:

### Gap 1 (RECOMMENDED): Dual-View Contrastive Learning (DualCL)
> **Nobody has combined LightGCL's SVD augmentation with SimGCL's embedding noise as two complementary contrastive views.**

- LightGCL (structural view): captures global spectral patterns via SVD
- SimGCL (embedding view): captures local perturbation robustness via noise
- These are **theoretically complementary**: one operates on the graph structure, the other on the embedding space
- Each is independently proven SOTA-class
- **Expected gain: 3-7% over LightGCL alone on Amazon-Book**
- Risk: LOW — both components are well-understood

### Gap 2: Frequency-Adaptive Spectral Augmentation (FreqCL)
> **LightGCL's SVD treats all frequency components equally, but high/low frequencies encode different signals.**

- High singular values → popularity/mainstream patterns
- Low singular values → niche/long-tail preferences
- A frequency-adaptive weighting scheme could preserve more niche information during augmentation
- **Could be combined with Gap 1 as an enhancement**

### Gap 3: Metadata-Enhanced Contrastive Views (MetaCL)
> **No paper combines item metadata (category, brand) as a contrastive view with structural contrastive learning.**

- Amazon-Beauty has rich metadata (category, brand, price)
- Create a metadata-based similarity graph as a third contrastive view
- Especially useful under data sparsity (metadata doesn't depend on interactions)
- KGAT + SGL-style combination hasn't been published

### Gap 4: Layer-Adaptive Contrastive Learning (LayerCL)
> **All methods apply contrastive loss only on final embeddings, but different GNN layers capture different structural semantics.**

- Layer 1: direct neighbors (local taste)
- Layer 2: 2-hop (collaborative patterns)
- Layer 3: 3-hop (community structure)
- XSimGCL touched this briefly but without SVD augmentation

---

## Part 3: Recommended Research Direction

### Primary Contribution: **DualCL — Dual-View Contrastive Learning for Recommendation**

**Why DualCL?**
1. **Highest reward-to-risk ratio** — both SVD and noise augmentations are individually proven
2. **Clean novelty story** — "We observe that structural (SVD) and embedding (noise) augmentations capture complementary signals for CF. We propose DualCL to combine them."
3. **Perfect ablation setup** — SVD-only (= LightGCL), noise-only (= SimGCL), both (= DualCL)
4. **Publishable even if no improvement** — thorough comparative analysis with ablation is itself a contribution
5. **BTP-scale complexity** — implementable in 4-6 weeks with existing codebases

### Architecture

```
Input: User-Item Bipartite Graph G, Adjacency A

Step 1: SVD Augmentation (from LightGCL)
  - Truncated SVD: A ≈ U_q Σ_q V_q^T
  - Augmented adj: Ã = U_q Σ_q V_q^T  (low-rank reconstruction)

Step 2: LightGCN Encoding (shared encoder)
  - Original view: E^(L) = (1/(L+1)) Σ_{l=0}^{L} (D^{-1/2} A D^{-1/2})^l · E^(0)
  - SVD view:      Ẽ^(L) = (1/(L+1)) Σ_{l=0}^{L} (D̃^{-1/2} Ã D̃^{-1/2})^l · E^(0)

Step 3: Noise Augmentation (from SimGCL)
  - Noise view:    Ê_i = E_i^(L) + Δ_i,  where Δ_i ~ Uniform, ||Δ_i|| = ε

Step 4: Dual Contrastive Loss
  - L_CL1 = InfoNCE(E^(L), Ẽ^(L))     [structural contrast]
  - L_CL2 = InfoNCE(E^(L), Ê^(L))      [embedding contrast]

Step 5: Total Loss
  - L = L_BPR + λ₁·L_CL1 + λ₂·L_CL2 + λ₃·||Θ||²
```

### Research Questions

- **RQ1:** Does combining structural (SVD) and embedding (noise) contrastive views improve recommendation quality over using either view alone?
- **RQ2:** What is the relative contribution of each contrastive view, and do they capture complementary signals?
- **RQ3:** How does DualCL perform under varying data sparsity compared to single-view methods?

---

## Part 4: Detailed Implementation Plan

### Phase 0: Environment & Data Setup (Week 1)
```
Tasks:
├── Set up Python environment (PyTorch, PyG, scipy, numpy)
├── Process Amazon-Book data
│   ├── Load train.txt and test.txt (LightGCN format)
│   ├── Create validation split from training (hold out 10%)
│   ├── Build adjacency matrix (sparse, bidirectional)
│   └── Verify: edge_index.shape == [2, 2 * num_train_interactions]
├── Process Amazon-Beauty data (secondary dataset)
│   ├── Load JSONL reviews, filter 10-core
│   ├── Temporal leave-one-out split (using timestamps)
│   └── Build adjacency matrix
└── Implement evaluation module
    ├── Recall@K, NDCG@K (K=10,20)
    ├── Full-item ranking with training-item masking
    ├── Per-user computation, then average
    └── Significance testing (paired t-test)
```

### Phase 1: Baseline Models (Weeks 2-3)
```
Tasks:
├── Implement MF-BPR baseline
│   ├── Embed dim=64, BPR loss, Adam optimizer
│   └── Run 5 seeds, report mean±std
├── Implement LightGCN baseline
│   ├── 3 layers, embed dim=64, layer combination = mean
│   ├── BPR loss, early stopping on val NDCG@10 (patience=50)
│   └── Run 5 seeds, verify against published numbers
├── Implement SimGCL
│   ├── LightGCN encoder + noise perturbation
│   ├── InfoNCE contrastive loss
│   ├── Hyperparams: noise ε ∈ {0.1, 0.5, 1.0}, λ_CL ∈ {0.1, 0.5, 1.0}
│   └── Run 5 seeds
├── Implement LightGCL
│   ├── LightGCN encoder + SVD augmented graph
│   ├── Hyperparams: SVD rank q ∈ {5, 10, 20}, λ_CL ∈ {0.1, 0.5, 1.0}
│   └── Run 5 seeds
└── Results table: verify all baselines match published numbers (±2%)
```

### Phase 2: DualCL Implementation (Weeks 4-5)
```
Tasks:
├── Implement DualCL model
│   ├── LightGCN encoder (shared weights)
│   ├── SVD augmentation module (from LightGCL)
│   ├── Noise augmentation module (from SimGCL)
│   ├── Dual InfoNCE contrastive loss
│   └── Combined loss: L_BPR + λ₁·L_CL_SVD + λ₂·L_CL_noise + λ₃·||Θ||²
├── Hyperparameter search
│   ├── SVD rank q: {5, 10, 20}
│   ├── Noise ε: {0.1, 0.5, 1.0}
│   ├── λ₁ (SVD weight): {0.1, 0.5, 1.0}
│   ├── λ₂ (noise weight): {0.1, 0.5, 1.0}
│   └── Grid search on validation set, select by NDCG@10
├── Run full experiment (5 seeds, best hyperparams)
└── Compare: DualCL vs LightGCL vs SimGCL vs LightGCN vs MF-BPR
```

### Phase 3: Ablation & Analysis (Week 6)
```
Tasks:
├── Ablation study
│   ├── DualCL (full model)
│   ├── DualCL w/o SVD view (= SimGCL)
│   ├── DualCL w/o noise view (= LightGCL)
│   ├── DualCL w/o both CL losses (= LightGCN)
│   └── λ₁/λ₂ sensitivity analysis
├── Data sparsity experiment
│   ├── Subsample training data: 100%, 75%, 50%, 25%, 10%
│   ├── Run all models on each subset
│   └── Plot: performance vs sparsity curves
├── Layer depth analysis
│   ├── Vary GNN layers: 1, 2, 3, 4, 5
│   ├── Monitor oversmoothing effects
│   └── Does dual contrastive loss reduce oversmoothing?
├── Embedding visualization
│   ├── t-SNE of user/item embeddings
│   ├── Compare LightGCN vs SimGCL vs LightGCL vs DualCL
│   └── Show: DualCL produces more uniform embedding space
└── Statistical significance
    ├── Paired t-test (DualCL vs each baseline)
    └── Report p-values
```

### Phase 4: Secondary Dataset + Writing (Weeks 7-8)
```
Tasks:
├── Run DualCL on Amazon-Beauty
│   ├── Same protocol, same baselines
│   └── Cross-dataset generalization evidence
├── (Optional) MetaCL extension
│   ├── If Amazon-Beauty results interesting + time permits
│   ├── Add metadata-based contrastive view as third view
│   └── Ablate: DualCL vs TripleCL (with metadata view)
├── Write thesis / report
│   ├── Chapter 1: Introduction (problem, motivation, contributions)
│   ├── Chapter 2: Literature Review (GNN, attention, CL for RecSys)
│   ├── Chapter 3: Methodology (DualCL architecture, loss, training)
│   ├── Chapter 4: Experimental Setup (datasets, metrics, baselines)
│   ├── Chapter 5: Results & Analysis (main results, ablation, sparsity)
│   └── Chapter 6: Conclusion & Future Work
└── Prepare presentation slides
```

---

## Part 5: Expected Results & Contribution Story

### Hypothesis
DualCL will outperform both LightGCL and SimGCL because:
1. SVD augmentation captures **global spectral** collaborative patterns
2. Noise augmentation captures **local embedding** robustness
3. These are **complementary signals** — one operates on graph topology, the other on the representation space
4. The dual contrastive loss forces the encoder to learn representations that are invariant to BOTH structural and embedding perturbations

### Expected Results Table (Amazon-Book)

| Method | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|--------|-----------|-----------|---------|---------|
| MF-BPR | ~0.018 | ~0.028 | ~0.015 | ~0.018 |
| LightGCN | ~0.022 | ~0.032 | ~0.018 | ~0.024 |
| SimGCL | ~0.032 | ~0.047 | ~0.025 | ~0.036 |
| LightGCL | ~0.040 | ~0.059 | ~0.031 | ~0.044 |
| **DualCL (ours)** | **~0.042** | **~0.062** | **~0.033** | **~0.046** |

### Contribution Framing
> "We propose DualCL, a dual-view contrastive learning framework for collaborative filtering that combines structural augmentation via truncated SVD with embedding-space augmentation via uniform noise perturbation. While prior work has explored these augmentation strategies independently (LightGCL for SVD, SimGCL for noise), we demonstrate that they capture complementary collaborative signals and yield consistent improvements when combined. Extensive experiments on Amazon-Book and Amazon-Beauty show that DualCL achieves X% improvement over the strongest single-view baseline, with particular gains under data sparsity conditions."

---

## Part 6: Project Directory Structure

```
BTP/
├── RESEARCH_ANALYSIS_AND_PLAN.md     ← this document
├── AGENTS.md
├── src/
│   ├── data/
│   │   ├── dataset.py                ← data loading, graph construction
│   │   └── preprocessing.py          ← filtering, splitting, ID mapping
│   ├── models/
│   │   ├── mf_bpr.py                 ← Matrix Factorization + BPR
│   │   ├── lightgcn.py               ← LightGCN baseline
│   │   ├── simgcl.py                 ← SimGCL (noise contrastive)
│   │   ├── lightgcl.py               ← LightGCL (SVD contrastive)
│   │   └── dualcl.py                 ← DualCL (our model)
│   ├── evaluation/
│   │   └── metrics.py                ← Recall, NDCG, significance tests
│   ├── train.py                      ← unified training loop
│   └── config.py                     ← hyperparameter configs
├── configs/
│   ├── amazon_book.yaml
│   └── amazon_beauty.yaml
├── experiments/
│   ├── run_baselines.sh
│   ├── run_dualcl.sh
│   └── run_ablation.sh
├── results/                          ← auto-generated
│   ├── tables/
│   ├── figures/
│   └── logs/
└── roadmap/                          ← existing study materials
    └── data/                         ← existing datasets
```

---

## Part 7: Risk Assessment & Fallbacks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| DualCL doesn't beat LightGCL | Medium | Thorough ablation + sparsity analysis is still a valid contribution |
| SVD computation too slow on full Amazon-Book | Low | Use randomized SVD (scipy.sparse.linalg.svds), rank q≤20 |
| Hyperparameter space too large | Medium | Fix SVD params at LightGCL's best, fix noise at SimGCL's best, only tune λ₁/λ₂ |
| Can't reproduce published baselines | Low | Use official codebases; ±5% is acceptable for different splits |
| Amazon-Book has no validation set | Medium | Hold out 10% of training data randomly; document this deviation |

### Fallback Plan
If DualCL shows no improvement:
1. **Pivot to analysis paper** — "Why Dual-View Contrastive Augmentation Is Redundant in GCN-based CF" (negative result papers are publishable)
2. **Pivot to MetaCL** — use Amazon-Beauty metadata as the novel contrastive view instead
3. **Pivot to frequency-adaptive** — modify LightGCL's SVD with spectral weighting
