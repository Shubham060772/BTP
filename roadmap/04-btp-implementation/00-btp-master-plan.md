# BTP Master Plan: Surpassing Recent GNN-Based Recommendation Systems

> **Goal:** Build a GNN-based product recommendation model that achieves competitive or superior results to recent state-of-the-art methods on the Amazon-Book dataset.

---

## 1. Literature Landscape

### 1.1 Current SOTA Results on Amazon-Book

The Amazon-Book dataset (52,643 users, 91,599 items, ~3M interactions, 10-core) is a standard benchmark. Here are published results (Recall@20 / NDCG@20):

| Method | Year | Recall@20 | NDCG@20 | Key Technique |
|--------|------|-----------|---------|---------------|
| NCF | 2017 | 0.0142 | 0.0085 | Neural collaborative filtering |
| GCCF | 2019 | 0.0317 | 0.0243 | Graph co-filtering (no nonlinearity) |
| LightGCN | 2020 | 0.0319 | 0.0236 | Simplified GCN, no feature transform |
| DGCF | 2020 | 0.0211 | 0.0154 | Disentangled graph CF |
| SGL | 2021 | 0.0327 | 0.0249 | Self-supervised graph learning (augment) |
| NCL | 2022 | ~0.033 | ~0.025 | Neighborhood-enriched contrastive |
| DirectAU | 2022 | ~0.035 | ~0.027 | Alignment + uniformity loss |
| SimGCL | 2022 | 0.0474 | 0.0360 | Noise-based contrastive (no augment) |
| SHT | 2022 | 0.0441 | 0.0328 | Hypergraph transformer |
| HCCF | 2022 | 0.0322 | 0.0247 | Hypergraph contrastive CF |
| XSimGCL | 2023 | ~0.049 | ~0.037 | Extremely simple contrastive |
| **LightGCL** | **2023** | **0.0585** | **0.0436** | SVD-based contrastive augmentation |

### 1.2 Key Observations

1. **Contrastive learning dominates.** Every method that significantly beats LightGCN uses some form of contrastive/self-supervised learning. The gap between LightGCN (0.0319) and LightGCL (0.0585) is **83% improvement** in Recall@20.

2. **Graph augmentation is losing favor.** SGL used graph structure augmentation (node/edge dropout). SimGCL showed augmentation is unnecessary — noise in embedding space works better. LightGCL uses SVD for augmentation. The trend is toward simpler, more principled augmentation.

3. **Representation quality matters more than architecture complexity.** LightGCN beat the more complex NGCF. SimGCL/XSimGCL beat SGL with simpler methods. The winning insight is learning more uniform, well-distributed representations rather than building complex message-passing.

4. **Attention mechanisms are underexplored in this line of work.** GAT exists but most SOTA graph CF methods use uniform aggregation (LightGCN-style) + contrastive loss rather than attention-weighted aggregation. SHT (hypergraph transformer) showed promise but didn't beat SimGCL/LightGCL.

5. **Multi-interest / disentangled approaches underperform.** DGCF (disentangled) scored only 0.0211 — worse than LightGCN. Disentanglement adds complexity without clear benefit in the standard CF setting.

### 1.3 What Recent Papers Are Doing (2023-2025 Trends)

| Trend | Examples | Status |
|-------|----------|--------|
| SVD-based augmentation | LightGCL | Current SOTA on Amazon-Book |
| Noise-based contrastive | SimGCL, XSimGCL | Strong, simple, proven |
| Graph transformers | SHT, NodeFormer | Promising but not yet beating CL methods |
| LLM-enhanced features | LLM4Rec papers | Emerging, mostly on text-rich datasets |
| Adaptive augmentation | AutoCF, AdaGCL | Moderate gains, added complexity |
| Multi-behavior modeling | MBGCN, CML | Uses multi-type interactions |
| Knowledge graph integration | KGAT, KGIN | Rich but needs side information |

---

## 2. Identified Gaps

### Gap 1: Attention + Contrastive Learning Synergy

Current SOTA (LightGCL, SimGCL) uses **uniform aggregation** (all neighbors weighted equally) with contrastive learning. GAT-style attention exists but hasn't been combined effectively with modern contrastive objectives. The hypothesis: **adaptive, attention-weighted aggregation could provide better-quality contrastive views** because:
- Not all neighbors are equally informative
- Attention can suppress noisy interactions (mis-clicks, gifts)
- Contrastive loss could benefit from more semantically coherent views

### Gap 2: Layer-Adaptive Contrastive Learning

All current methods apply contrastive loss on the **final** embeddings. But GNN layers capture different levels of information:
- Layer 1: direct neighbors (local taste)
- Layer 2: 2-hop connections (collaborative signal)
- Layer 3: broader community patterns

No paper systematically applies contrastive objectives at **each layer** with layer-specific augmentation strategies.

### Gap 3: Frequency-Aware Augmentation

LightGCL uses SVD to decompose the interaction graph. But SVD treats all singular components uniformly. In practice:
- High-frequency components capture popularity patterns
- Low-frequency components capture niche/personalized taste
- A **frequency-aware** contrastive strategy could weight these differently

### Gap 4: Hybrid Structural + Embedding Augmentation

SimGCL showed embedding noise works. LightGCL showed SVD-based structural augmentation works. **Nobody has combined both** — using structural augmentation for one contrastive view and embedding augmentation for the other could capture complementary information.

---

## 3. Proposed Approaches

### Option A: Attention-Enhanced Contrastive Graph CF (AttCL)

**Core Idea:** Replace LightGCN's uniform aggregation with a lightweight attention mechanism, and use the attention weights to generate more informed contrastive views. Instead of random noise (SimGCL) or SVD (LightGCL), use attention-derived importance scores to create augmented views that preserve important structural patterns while perturbing less important ones.

**Why it could work:**
- Attention identifies which neighbors matter for each user/item
- Augmentation guided by attention is more semantically meaningful than random perturbation
- LightGCN's simplicity (no feature transform) can be preserved while adding only a lightweight attention layer

**Difficulty:** Medium

**Novelty:** Attention has been used in GNNs (GAT), and contrastive learning has been used separately (SimGCL). The novelty is in using attention to guide contrastive augmentation rather than just message aggregation.

**Expected improvement:** 3-8% over LightGCN baseline, competitive with SimGCL/LightGCL.

---

### Option B: Layer-Wise Contrastive Learning (LayerCL)

**Core Idea:** Apply contrastive objectives at each GNN layer independently rather than only at the final layer. Each layer captures a different neighborhood scope, so layer-specific contrastive learning can regularize representations at multiple granularities simultaneously.

**Why it could work:**
- Early layers capture local patterns, later layers capture global ones
- Contrastive loss at each layer prevents information loss during propagation
- Similar to deep supervision in CNNs, which improves gradient flow

**Difficulty:** Easy-Medium

**Novelty:** Layer-wise contrastive learning has been explored in vision (BYOL-like methods) but not systematically in graph CF. Most graph CF methods apply CL only on final embeddings.

**Expected improvement:** 2-5% over single-layer CL baseline.

---

### Option C: Dual-View Contrastive Learning (DualCL)

**Core Idea:** Generate two contrastive views using fundamentally different augmentation strategies — one structural (SVD-based, like LightGCL) and one embedding-based (noise, like SimGCL). The views capture complementary information: structural augmentation preserves global graph topology while embedding noise captures local semantic perturbations.

**Why it could work:**
- SimGCL and LightGCL each work well independently
- Different augmentation types capture different aspects of the data
- Multi-view learning theory suggests diverse views improve representation quality

**Difficulty:** Easy

**Novelty:** Straightforward but not yet published. The simplicity is a feature — it's a natural next step that surprisingly hasn't been done.

**Expected improvement:** 3-7% over either SimGCL or LightGCL alone.

---

### Option D: Frequency-Adaptive Graph Contrastive Learning (FreqGCL)

**Core Idea:** Decompose the user-item interaction graph using SVD (like LightGCL), then apply different contrastive strategies to different frequency bands. High-frequency components (popular items, common patterns) get stronger augmentation, while low-frequency components (niche taste, long-tail items) are augmented more gently to preserve personalization signal.

**Why it could work:**
- Popularity bias is a known problem in CF — popular items dominate embeddings
- Gentle augmentation on low-frequency components preserves long-tail signal
- Aggressive augmentation on high-frequency components prevents overfitting to popularity

**Difficulty:** Medium-Hard

**Novelty:** Frequency-aware augmentation is novel in graph CF. Connects to spectral graph theory and signal processing, which is theoretically grounded.

**Expected improvement:** 5-10% on long-tail items, 2-5% overall.

---

## 4. Recommended Approach: Option C — Dual-View Contrastive Learning (DualCL)

### 4.1 Why This One

- **Highest reward-to-risk ratio.** Both component techniques (SimGCL's noise, LightGCL's SVD) are proven SOTA. Combining them is low-risk.
- **Simple implementation.** No new architectural components — just two augmentation strategies feeding into one contrastive loss.
- **Clear ablation story.** You can show: (1) SVD-only (LightGCL), (2) noise-only (SimGCL), (3) both (DualCL), and the combination wins.
- **Fallback safety.** Even if the combination doesn't improve over the best individual method, you still have a thorough comparison study as your BTP contribution.

### 4.2 Architecture

```
Input: User-Item Bipartite Graph (Amazon-Book)
         |
         v
┌─────────────────────────────────────┐
│       LightGCN Encoder              │
│  (K layers, no feature transform)   │
│                                     │
│  e_u^(k) = Σ (1/√|N_u|·1/√|N_i|)  │
│             · e_i^(k-1)             │
│                                     │
│  Final: e_u = Σ α_k · e_u^(k)      │
│               k=0..K                │
└───────────┬─────────────────────────┘
            |
            v
    ┌───────┴───────┐
    v               v
┌────────┐    ┌────────────┐
│ View 1 │    │   View 2   │
│ SVD    │    │   Noise    │
│ Augment│    │   Augment  │
└───┬────┘    └─────┬──────┘
    |               |
    v               v
┌────────┐    ┌────────────┐
│ LightGCN│   │  LightGCN  │
│ on Ã₁  │    │  + ε noise │
└───┬────┘    └─────┬──────┘
    |               |
    v               v
    z₁_u, z₁_i     z₂_u, z₂_i
    |               |
    └───────┬───────┘
            v
┌─────────────────────────────────────┐
│     Contrastive Loss (InfoNCE)      │
│                                     │
│  L_cl = -log( exp(z₁·z₂/τ) /       │
│              Σ exp(z₁·z_neg/τ) )    │
│                                     │
│  Applied to both users and items    │
└───────────┬─────────────────────────┘
            |
            v
┌─────────────────────────────────────┐
│     BPR Recommendation Loss         │
│                                     │
│  L_bpr = -Σ log σ(e_u·e_pos -      │
│                    e_u·e_neg)       │
└───────────┬─────────────────────────┘
            |
            v
┌─────────────────────────────────────┐
│     Total Loss                      │
│                                     │
│  L = L_bpr + λ₁·L_cl + λ₂·‖Θ‖²   │
└─────────────────────────────────────┘
```

### 4.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base encoder | LightGCN (3 layers) | Standard, proven, matches baselines |
| SVD augmentation | Top-q singular vectors, q ≈ rank/5 | Following LightGCL's approach |
| Noise augmentation | Uniform noise, ε ~ U(-δ, δ) | Following SimGCL |
| Contrastive loss | InfoNCE with temperature τ | Standard in CL literature |
| Training | Joint BPR + CL, end-to-end | Following SimGCL/LightGCL pattern |
| Embedding dim | 64 | Standard for Amazon-Book |
| Batch size | 2048 | Standard for Amazon-Book |
| Optimizer | Adam, lr=1e-3 | Standard |

### 4.4 Baselines to Compare Against

**Must-have baselines (implement or reproduce):**

| Baseline | Why | Source |
|----------|-----|--------|
| MF-BPR | Simplest CF baseline | Implement yourself |
| LightGCN | Your base architecture | LightGCN-PyTorch repo |
| SGL | Graph augmentation CL | SELFRec repo |
| SimGCL | Noise-based CL | SELFRec repo |
| LightGCL | SVD-based CL (current SOTA) | Official repo |

**Nice-to-have baselines (cite from papers):**
- NGCF, DGCF, NCL, DirectAU, XSimGCL

### 4.5 Ablation Studies

| Ablation | What it shows |
|----------|--------------|
| DualCL (full) vs SVD-only | Value of adding noise view |
| DualCL (full) vs noise-only | Value of adding SVD view |
| DualCL vs simple concatenation of views | Whether dual-view interaction matters |
| Varying λ₁ (CL weight) | Sensitivity to CL strength |
| Varying temperature τ | Sensitivity to contrastive sharpness |
| Varying SVD rank q | How much structural information to keep |
| Varying noise magnitude δ | How much embedding perturbation helps |
| 1-layer vs 2-layer vs 3-layer | Optimal propagation depth |
| With BPR only vs With CL only vs Both | Contribution of each loss |

---

## 5. Timeline (12 Weeks)

### Phase 1: Foundation (Weeks 1-3)

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Study GNN fundamentals (Phases 01-05 in gnn-study/) | Notes, exercises done |
| 2 | Study GraphSAGE, GAT, PyTorch Geometric (Phases 06-07) | Working PyG code |
| 3 | Study NGCF, LightGCN, BPR (Phases 08-09) | Understand LightGCN paper fully |

**Milestone 1:** Can explain LightGCN architecture and BPR loss from memory.

### Phase 2: Baseline Implementation (Weeks 4-5)

| Week | Task | Deliverable |
|------|------|-------------|
| 4 | Implement MF-BPR and LightGCN on Amazon-Book | Reproducible baselines |
| 5 | Reproduce LightGCN published numbers, implement evaluation | Recall@20 ≈ 0.032, NDCG@20 ≈ 0.024 |

**Milestone 2:** LightGCN baseline matches published results (±5%).

### Phase 3: Contrastive Learning (Weeks 6-7)

| Week | Task | Deliverable |
|------|------|-------------|
| 6 | Implement SimGCL (noise-based CL) on top of LightGCN | SimGCL numbers on Amazon-Book |
| 7 | Implement LightGCL (SVD-based CL) | LightGCL numbers on Amazon-Book |

**Milestone 3:** Both CL baselines working, results match published (±10%).

### Phase 4: Proposed Method (Weeks 8-9)

| Week | Task | Deliverable |
|------|------|-------------|
| 8 | Implement DualCL: combine SVD + noise views | First results |
| 9 | Hyperparameter tuning (λ₁, τ, q, δ) | Best configuration found |

**Milestone 4:** DualCL results on Amazon-Book, initial comparison table.

### Phase 5: Analysis & Ablations (Weeks 10-11)

| Week | Task | Deliverable |
|------|------|-------------|
| 10 | Run all ablation studies from Section 4.5 | Ablation tables |
| 11 | Error analysis, per-group analysis (active vs cold users, popular vs long-tail items) | Analysis plots |

**Milestone 5:** Complete experimental results with ablations.

### Phase 6: Writing (Week 12)

| Week | Task | Deliverable |
|------|------|-------------|
| 12 | Write BTP report: problem, method, experiments, analysis | Final BTP document |

**Milestone 6:** Submitted BTP.

---

## 6. Risk Mitigation

### Risk 1: DualCL doesn't beat individual methods

**Probability:** Medium (30-40%)

**Mitigation:**
- Even if DualCL doesn't win overall, analyze WHERE it wins (cold users? long-tail items?) — partial improvement is still a contribution
- Pivot to Option B (LayerCL) or Option A (AttCL) — they're quick to implement on top of your existing codebase
- Worst case: your BTP becomes a thorough comparative study of contrastive augmentation strategies, which is still a valid contribution

### Risk 2: Can't reproduce baseline numbers

**Probability:** Low-Medium (20%)

**Mitigation:**
- Use official code repos (LightGCN-PyTorch, SELFRec) as reference implementations
- Start with exact same hyperparameters as published papers
- Allow ±10% variance — exact reproduction is hard due to random seeds and hardware
- Document any deviations clearly

### Risk 3: Training is too slow on laptop

**Probability:** Medium (Amazon-Book has 3M interactions)

**Mitigation:**
- LightGCN is computationally light — should be feasible
- Use Google Colab free tier (T4 GPU) if needed
- Reduce batch size or embedding dim temporarily during development
- Develop and debug on Amazon-Beauty (smaller), then run final experiments on Amazon-Book

### Risk 4: Running out of time

**Probability:** Low-Medium

**Mitigation:**
- Weeks 1-5 (baselines) are the foundation — don't skip
- If behind schedule by week 7, simplify: implement only one CL method + your proposed improvement
- The minimum viable BTP is: LightGCN baseline + one CL method (SimGCL) + your DualCL variant + basic ablations

### Minimum Viable Contribution

Even in the worst case, your BTP should contain:
1. LightGCN baseline on Amazon-Book (reproduced)
2. At least one contrastive method (SimGCL or LightGCL)
3. Your proposed DualCL variant
4. Ablation showing contribution of each component
5. Analysis of where methods succeed/fail

This is a solid BTP regardless of whether DualCL beats all baselines.

---

## 7. Repository Structure

```
BTP/
├── roadmap/                    # Study materials (this repo)
│   ├── 01-gnn-study/
│   ├── 02-attention-study/
│   ├── 03-combined-approaches/
│   ├── 04-btp-implementation/
│   └── data/
│       ├── amazon-book/
│       └── amazon-beauty/
├── src/                        # Implementation (create when starting Phase 2)
│   ├── models/
│   │   ├── mf_bpr.py
│   │   ├── lightgcn.py
│   │   ├── simgcl.py
│   │   ├── lightgcl.py
│   │   └── dualcl.py          # Your proposed method
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── utils/
│   │   ├── metrics.py         # Recall@K, NDCG@K
│   │   └── training.py
│   ├── configs/
│   │   └── amazon_book.yaml
│   └── train.py
├── experiments/                # Results and logs
│   ├── baselines/
│   ├── ablations/
│   └── analysis/
└── report/                     # BTP thesis
    └── main.tex
```

---

## 8. Key Papers to Read

**Priority order (read these thoroughly):**

1. He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)
2. Yu et al., "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation" (SimGCL, SIGIR 2022)
3. Cai et al., "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation" (ICLR 2023)
4. Wang et al., "Neural Graph Collaborative Filtering" (NGCF, SIGIR 2019)
5. Lin et al., "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning" (NCL, WWW 2022)

**Skim for context:**
- Wu et al., "Self-supervised Graph Learning for Recommendation" (SGL, SIGIR 2021)
- Yu et al., "XSimGCL: Towards Extremely Simple Graph Contrastive Learning" (TKDE 2023)
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)
