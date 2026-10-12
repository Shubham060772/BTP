# Revised Research Analysis & Implementation Plan
## Customer Product Recommendation System Using Transformers and GNNs — BTP, IIT Patna

**Date:** October 2026
**Status:** Literature survey complete. Ready for implementation.

---

## Part 1: Literature Landscape — What Exists and What Doesn't

### 1.1 Core Papers (Verified)

#### Foundational — Sequential Recommendation (Transformer-based)

| Paper | Venue | Year | IEEE? | Key Idea | Limitation |
|-------|-------|------|-------|----------|------------|
| **SASRec** | IEEE ICDM | 2018 | YES | Causal self-attention for next-item prediction | No collaborative signal; each user encoded independently |
| **BERT4Rec** | ACM CIKM | 2019 | No | Bidirectional Transformer with Cloze objective | Train-test mismatch; needs 30x training time; no collaborative signal |

**Key insight:** Transformer-based sequential models capture *what the user did recently* but are blind to *what similar users do*. They process each user in isolation.

#### Foundational — Graph Collaborative Filtering (GNN-based)

| Paper | Venue | Year | IEEE? | Key Idea | Limitation |
|-------|-------|------|-------|----------|------------|
| **LightGCN** | ACM SIGIR | 2020 | No | Stripped GCN — only neighborhood aggregation | No sequential modeling; uniform layer weights; no metadata |
| **SGL** | ACM SIGIR | 2021 | No | Self-supervised graph augmentation (node/edge drop) | Stochastic augmentations lose global structure |
| **LightGCL** | ICLR | 2023 | No | SVD-based deterministic augmentation | Single view; no sequential; no metadata |

**Key insight:** GNN-based models capture *collaborative patterns from similar users* but are blind to *the order and recency of interactions*. They treat the interaction history as a set, not a sequence.

#### The Hybrid Gap — Transformer + GNN

| Paper | Venue | Year | IEEE? | Key Idea | Limitation |
|-------|-------|------|-------|----------|------------|
| **SASGCN** | IEEE Access | 2024 | YES | SASRec + vanilla GCN, simple fusion | Uses basic GCN (not LightGCN); concatenation/MLP fusion (not adaptive); no sparsity analysis; only 5 citations |
| **TGT** | IEEE TKDE | 2022 | YES | Temporal graph Transformer for multi-behavior | Requires multi-behavior data; complex |
| **TGSRec** | ACM CIKM | 2021 | No | Temporal collaborative Transformer on continuous-time graph | Complex temporal encoding |
| **EA-GPS** | ACM TOIS | 2025 | No | External attention + GCN + positional prompts | Lightweight but limited fusion analysis |
| **WPGRec** | ACM SIGIR | 2026 | No | Wavelet + graph + adaptive gated fusion | Very recent; complex wavelet computation |
| **MAERec** | ACM SIGIR | 2023 | No | Masked autoencoder on item-transition graph | Self-supervised; no explicit Transformer+GNN fusion |
| **MVCrec** | arXiv preprint | 2026 | No | Multi-view attention + contrastive (ID + graph views) | Unpublished preprint; contrastive-heavy |

#### Multi-Behavior Recommendation (Context)

| Paper | Venue | Year | IEEE? | Key Idea |
|-------|-------|------|-------|----------|
| **MBHT** | ACM KDD | 2022 | No | Multi-behavior hypergraph + Transformer |
| **SG-MST** | IEEE ICDM | 2024 | YES | Hypergraph + contrastive Transformer for e-commerce |
| **MGUI** | IEEE TCSS | 2026 | YES | Multi-grained interests + item-transition graph + MoE |
| **SpectraMB** | ACM KDD | 2026 | No | Spectral denoising + attention for multi-behavior |
| **GCIB** | ICML | 2026 | No | Graph information bottleneck for multi-behavior |
| **SWGCN** | Info. Sciences | 2026 | No | Synergy-weighted GCN for cross-behavioral signals |
| **MB-DGT** | arXiv preprint | 2026 | No | Boundary-aware dynamic graph Transformer |

#### Surveys

| Paper | Year | arXiv |
|-------|------|-------|
| GNN in RecSys: A Survey | 2020/2022 | 2011.02260 |
| Graph Foundation Models for Recommendation | 2025 | 2502.08346 |
| Session-based GNN+Sequential Survey | 2024 | 2408.14851 |

### 1.2 IEEE Paper Count for Your BTP

You have strong IEEE coverage:
- **SASRec** — IEEE ICDM 2018 (foundational, sequential)
- **SASGCN** — IEEE Access 2024 (direct comparator, hybrid)
- **TGT** — IEEE TKDE 2022 (temporal graph Transformer)
- **SG-MST** — IEEE ICDM 2024 (multi-behavior e-commerce)
- **MGUI** — IEEE TCSS 2026 (multi-grained interests)

Plus essential non-IEEE papers: LightGCN (SIGIR), BERT4Rec (CIKM), LightGCL (ICLR).

---

## Part 2: The Research Gap We Will Exploit

### 2.1 The Core Observation

**Transformer-based sequential models** (SASRec) and **GNN-based collaborative models** (LightGCN) capture fundamentally different signals:

| Signal Type | Model | Captures | Misses |
|-------------|-------|----------|--------|
| Sequential | SASRec | Recent preferences, temporal patterns, intent drift | What similar users like; cold-start collaborative info |
| Collaborative | LightGCN | Multi-hop user-item similarities, popularity, community patterns | Order of interactions; recency; user intent evolution |

**These signals are complementary.** A user who bought `Laptop → Mouse → Keyboard` has sequential intent (peripherals). But a user with only 2 interactions has no meaningful sequence — they need collaborative signal from similar users.

### 2.2 What SASGCN (IEEE 2024) Did — And What It Missed

SASGCN is the most directly relevant prior work. It combines SASRec + GCN. But it has clear limitations:

| What SASGCN Does | What It Misses |
|-------------------|----------------|
| Uses vanilla GCN | LightGCN is proven superior for CF (21% better on Amazon-Book) |
| Concatenation + MLP fusion | No adaptive mechanism — every user gets same fusion strategy |
| Single evaluation | No analysis of WHEN sequential vs. collaborative helps |
| No sparsity analysis | Doesn't show if the benefit varies by user activity level |

### 2.3 Our Specific Gap

> **No published paper adaptively fuses SASRec (Transformer) and LightGCN (GNN) representations with a learned per-user attention gate, and analyzes how the relative importance of sequential vs. collaborative signal varies with user interaction density.**

This is:
- **Novel** — SASGCN uses static fusion; MVCrec (unpublished preprint) uses contrastive learning not attention gating; WPGRec uses wavelets
- **Directly extends an IEEE paper** (SASGCN) — clean lineage
- **Experimentally testable** — the attention weight α is interpretable
- **Practically meaningful** — different users genuinely need different signal types

---

## Part 3: Proposed Model — AdaFuse

### 3.1 Full Name
**AdaFuse: Adaptive Attention Fusion of Sequential and Collaborative Representations for Recommendation**

### 3.2 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    AdaFuse Architecture                    │
│                                                           │
│  User Interaction Sequence: [i₁, i₂, ..., iₜ]           │
│  User-Item Bipartite Graph: G = (U ∪ I, E)               │
│                                                           │
│  ┌─────────────┐         ┌──────────────┐                │
│  │   SASRec     │         │   LightGCN    │                │
│  │  (Transformer)│         │    (GNN)      │                │
│  │             │         │              │                │
│  │  Causal     │         │  Layer 0: E⁰ │                │
│  │  Self-Attn  │         │  Layer 1: ÃE⁰│                │
│  │  + FFN      │         │  Layer 2: Ã²E⁰│               │
│  │  + PosEmbed │         │  ...          │                │
│  │             │         │  Output: mean │                │
│  └──────┬──────┘         └──────┬───────┘                │
│         │                       │                         │
│         ▼                       ▼                         │
│      e_seq(u)               e_col(u)                      │
│     (sequential             (collaborative                │
│      representation)         representation)               │
│         │                       │                         │
│         └───────────┬───────────┘                         │
│                     ▼                                     │
│         ┌───────────────────────┐                         │
│         │  Attention Fusion Gate │                         │
│         │                       │                         │
│         │  α(u) = σ(w^T · [e_seq; e_col; e_seq ⊙ e_col]) │
│         │                       │                         │
│         │  e_final(u) = α(u)·e_seq + (1-α(u))·e_col      │
│         └───────────┬───────────┘                         │
│                     ▼                                     │
│              e_final(u)^T · e_item → score                │
│                                                           │
│  Loss: L_BPR + λ·||Θ||²                                  │
└──────────────────────────────────────────────────────────┘
```

### 3.3 Mathematical Formulation

**Step 1 — Sequential Encoding (SASRec):**
```
Given user u's interaction sequence S_u = [i₁, i₂, ..., iₜ]:

Input embeddings:    H⁰ = [e_{i₁} + p₁, e_{i₂} + p₂, ..., e_{iₜ} + pₜ]
Self-attention:      H^l = FFN(MultiHead-CausalAttn(H^{l-1}))   for l = 1..L_s
Sequential repr:     e_seq(u) = H^{L_s}[t]   (last position output)
```

**Step 2 — Collaborative Encoding (LightGCN):**
```
Given user-item bipartite graph with normalized adjacency Ã = D^{-1/2} A D^{-1/2}:

Layer propagation:   E^{(k)} = Ã · E^{(k-1)}     for k = 1..K
Layer combination:   E_final = (1/(K+1)) · Σ_{k=0}^{K} E^{(k)}
Collaborative repr:  e_col(u) = E_final[u]
```

**Step 3 — Adaptive Attention Fusion:**
```
Fusion input:        z(u) = [e_seq(u); e_col(u); e_seq(u) ⊙ e_col(u)]   ∈ R^{3d}
Attention weight:    α(u) = σ(w^T z(u) + b)                              ∈ (0, 1)
Fused representation: e_final(u) = α(u) · e_seq(u) + (1 - α(u)) · e_col(u)
```

**Step 4 — Scoring and Loss:**
```
Score:    ŷ(u, i) = e_final(u)^T · e_item(i)
Loss:     L = Σ_{(u,i⁺,i⁻)} -ln σ(ŷ(u,i⁺) - ŷ(u,i⁻)) + λ||Θ||²    (BPR)
```

### 3.4 Why This Design

| Design Choice | Rationale |
|---------------|-----------|
| LightGCN over vanilla GCN | LightGCN is proven 16-22% better than NGCF/GCN for CF (He et al. 2020). SASGCN uses vanilla GCN — this is a direct upgrade. |
| SASRec over BERT4Rec | SASRec is more stable (BERT4Rec needs 30x training — Petrov & Macdonald 2022), causal attention matches the sequential recommendation task. |
| Scalar attention gate α(u) | Interpretable — we can literally plot α vs. user interaction count to verify our hypothesis. More complex gates (multi-head, vector-valued) can be ablated. |
| Element-wise product e_seq ⊙ e_col in gate input | Captures interaction between the two signals, not just their independent features. Standard practice in gating mechanisms (cf. highway networks, GRU). |
| Shared item embeddings | SASRec and LightGCN share the same item embedding table — this keeps the model parameter-efficient and ensures the two views operate in the same embedding space. |

### 3.5 Research Questions

- **RQ1 (Main):** Does AdaFuse's adaptive attention fusion outperform SASGCN's static fusion and the individual SASRec and LightGCN baselines?
- **RQ2 (Analysis):** How does the learned fusion weight α(u) correlate with user interaction density? Do sparse users rely more on collaborative (GNN) signal?
- **RQ3 (Ablation):** What is the individual contribution of each component — the Transformer branch, the GNN branch, the attention gate, and the element-wise interaction term?
- **RQ4 (Robustness):** How does AdaFuse perform under varying data sparsity (100%/75%/50%/25% of training interactions)?

---

## Part 4: Experimental Design

### 4.1 Datasets

| Dataset | Users | Items | Interactions | Timestamps? | Metadata? | Source |
|---------|-------|-------|-------------|-------------|-----------|--------|
| **Amazon-Beauty** | ~52K | ~57K | ~400K | YES | YES (category, brand, price) | `roadmap/data/amazon-beauty/` |
| **Amazon-Book** | 52,643 | 91,599 | 2,984,108 | NO (pre-split) | NO | `roadmap/data/amazon-book/` |

**Primary dataset: Amazon-Beauty** — has timestamps (needed for SASRec sequence ordering) and metadata (useful for cold-start analysis).

**Secondary dataset: Amazon-Book** — larger scale, standard benchmark. The pre-split train/test files work for LightGCN; for SASRec, we need to reconstruct sequences from the training set's user-item lists (treating the list order as implicit sequence, which is standard when timestamps are unavailable).

### 4.2 Data Preprocessing

**Amazon-Beauty:**
1. Load JSONL reviews → filter 10-core (users and items with ≥10 interactions)
2. Sort each user's interactions by timestamp
3. Split: last item → test, second-to-last → validation, rest → training (leave-one-out)
4. Build user-item bipartite graph from training interactions only
5. Construct interaction sequences from training data (max length 50)

**Amazon-Book:**
1. Use existing train.txt / test.txt split
2. Create validation set: hold out 10% of training users' last items
3. Build graph from remaining training interactions
4. Construct sequences from user-item lists in training file

### 4.3 Baselines (7 models)

| # | Model | Type | Why Include |
|---|-------|------|-------------|
| 1 | **Popularity** | Non-personalized | Sanity check |
| 2 | **MF-BPR** | Matrix Factorization | Classic collaborative filtering |
| 3 | **LightGCN** | GNN-only | Graph collaborative baseline |
| 4 | **SASRec** | Transformer-only | Sequential baseline |
| 5 | **BERT4Rec** | Transformer-only (bidirectional) | Alternative sequential baseline |
| 6 | **SASGCN** | Hybrid (GCN + SASRec, static fusion) | Direct comparator — the paper we extend |
| 7 | **AdaFuse (ours)** | Hybrid (LightGCN + SASRec, attention fusion) | Proposed model |

### 4.4 Evaluation Protocol

- **Metrics:** Recall@10, Recall@20, NDCG@10, NDCG@20, HR@10
- **Ranking protocol:** Full-item ranking with training-item masking (preferred); document if using 100 sampled negatives
- **Seeds:** [42, 123, 456, 789, 1024] — report mean ± std
- **Embedding dimension:** 64 (standard)
- **Hyperparameter selection:** Grid search, select by validation NDCG@10
- **Early stopping:** Patience 50 epochs on validation NDCG@10
- **Significance:** Paired t-test, p < 0.05

### 4.5 Hyperparameter Space

| Parameter | Values |
|-----------|--------|
| Embedding dim | 64 |
| LightGCN layers | {2, 3, 4} |
| SASRec layers | {1, 2} |
| SASRec heads | {1, 2} |
| Max sequence length | {50} |
| Learning rate | {1e-3, 5e-4, 1e-4} |
| Batch size | {256, 512, 1024} |
| BPR weight decay λ | {1e-4, 1e-5} |
| Dropout (SASRec) | {0.2, 0.5} |

### 4.6 Ablation Studies

| Variant | Description | Tests |
|---------|-------------|-------|
| AdaFuse-full | Full model | — |
| w/o GNN branch | Only SASRec + trivial gate (α=1) | = SASRec baseline |
| w/o Transformer branch | Only LightGCN + trivial gate (α=0) | = LightGCN baseline |
| w/o attention gate | Concatenation + MLP instead of attention gate | ≈ SASGCN approach |
| w/o element-wise product | Gate input = [e_seq; e_col] only | Tests interaction term |
| Multi-head gate | α ∈ R^d (vector-valued, per-dimension fusion) | Tests gate complexity |

### 4.7 Analysis Experiments

**A. Sparsity-Stratified Evaluation:**
- Group users by interaction count: [1-5], [6-10], [11-20], [21-50], [50+]
- Report metrics per group for all models
- Plot: performance gain of AdaFuse over baselines vs. user group
- **Hypothesis:** AdaFuse's advantage is largest for medium-sparsity users (very sparse users lack both signals; very active users have strong sequences)

**B. Attention Weight Analysis:**
- Plot distribution of learned α across all users
- Scatter plot: α(u) vs. |interactions(u)|
- **Hypothesis:** α correlates negatively with interaction count (sparse users → lower α → more GNN reliance)

**C. Data Sparsity Robustness:**
- Subsample training interactions: 100%, 75%, 50%, 25%
- Run all models on each subset
- Plot: performance vs. sparsity level
- **Hypothesis:** AdaFuse degrades more gracefully because it can shift weight to whichever signal is less affected

**D. Embedding Visualization:**
- t-SNE of user embeddings from LightGCN, SASRec, and AdaFuse
- Show that AdaFuse produces a more structured embedding space

---

## Part 5: Implementation Plan (8 Weeks)

### Week 1: Environment & Data

```
Tasks:
├── Set up Python environment
│   ├── PyTorch 2.x, PyTorch Geometric
│   ├── scipy, numpy, pandas, matplotlib, scikit-learn
│   └── Verify GPU availability (CUDA)
├── Implement data loading
│   ├── Amazon-Beauty: JSONL → DataFrame → 10-core filter → leave-one-out split
│   ├── Amazon-Book: load train.txt/test.txt → create validation split
│   ├── Build sparse adjacency matrices (bidirectional, offset item IDs by num_users)
│   └── Build interaction sequences (sorted by timestamp for Beauty)
├── Implement evaluation module
│   ├── Recall@K, NDCG@K, HR@K (K=10,20)
│   ├── Full-item ranking with training-item masking
│   ├── Per-user computation → average across users
│   └── Paired t-test utility
└── Verify: edge_index.shape == [2, 2 * num_train_interactions]
```

**Deliverable:** Working data pipeline + evaluation module with unit tests.

### Week 2: LightGCN Baseline

```
Tasks:
├── Implement LightGCN
│   ├── Sparse matrix propagation: E^(k) = Ã · E^(k-1)
│   ├── Layer combination: mean of layers 0..K
│   ├── BPR loss with uniform negative sampling
│   └── Training loop with early stopping (val NDCG@10, patience 50)
├── Run on Amazon-Beauty (5 seeds)
├── Run on Amazon-Book (5 seeds)
└── Verify against published numbers (±5% acceptable)
```

**Deliverable:** LightGCN results table. Sanity check: Amazon-Book Recall@20 ≈ 0.032-0.041.

### Week 3: SASRec Baseline

```
Tasks:
├── Implement SASRec
│   ├── Item embedding + positional embedding
│   ├── Causal multi-head self-attention (L layers, H heads)
│   ├── Point-wise FFN + LayerNorm + Dropout
│   ├── Binary cross-entropy loss with negative sampling
│   └── Training loop with early stopping
├── Run on Amazon-Beauty (5 seeds)
├── Run on Amazon-Book (5 seeds)
├── Also implement: Popularity baseline, MF-BPR baseline
└── Optional: BERT4Rec baseline
```

**Deliverable:** SASRec + MF-BPR + Popularity results. Full baseline table taking shape.

### Week 4: SASGCN Comparator + AdaFuse Core

```
Tasks:
├── Implement SASGCN-style model
│   ├── Vanilla GCN encoder (for fair comparison with the paper)
│   ├── SASRec encoder (shared from Week 3)
│   ├── Concatenation + MLP fusion
│   └── Run on both datasets (5 seeds)
├── Implement AdaFuse
│   ├── LightGCN branch (shared from Week 2)
│   ├── SASRec branch (shared from Week 3)
│   ├── Shared item embedding table
│   ├── Attention fusion gate: α(u) = σ(w^T [e_seq; e_col; e_seq ⊙ e_col])
│   ├── Fused representation: e_final = α·e_seq + (1-α)·e_col
│   └── BPR loss on fused scores
└── Initial training run to verify convergence
```

**Deliverable:** Working AdaFuse model. Initial comparison with baselines.

### Week 5: Hyperparameter Tuning + Full Experiments

```
Tasks:
├── Grid search for AdaFuse hyperparameters
│   ├── LightGCN layers: {2, 3, 4}
│   ├── SASRec layers: {1, 2}, heads: {1, 2}
│   ├── Learning rate: {1e-3, 5e-4, 1e-4}
│   ├── Select best config by validation NDCG@10
│   └── Record all configurations and results
├── Full experiment: run best AdaFuse config (5 seeds, both datasets)
├── Full experiment: run all baselines with tuned hyperparameters
└── Compile main results table
```

**Deliverable:** Complete main results table (Table 1 of the thesis).

### Week 6: Ablation Studies

```
Tasks:
├── Run all ablation variants (5 seeds each):
│   ├── AdaFuse-full
│   ├── w/o GNN branch (= SASRec)
│   ├── w/o Transformer branch (= LightGCN)
│   ├── w/o attention gate (concat + MLP, ≈ SASGCN)
│   ├── w/o element-wise product
│   └── Multi-head gate variant
├── Compile ablation table
└── Analysis: which component contributes most?
```

**Deliverable:** Ablation table (Table 2) answering RQ3.

### Week 7: Analysis Experiments + Visualizations

```
Tasks:
├── Sparsity-stratified evaluation
│   ├── Group users by interaction count
│   ├── Run all models, report per-group metrics
│   └── Plot: performance vs. user group
├── Attention weight analysis
│   ├── Extract α(u) for all users from trained AdaFuse
│   ├── Scatter plot: α vs. |interactions|
│   ├── Compute correlation coefficient
│   └── Box plot: α distribution per user group
├── Data sparsity robustness experiment
│   ├── Subsample training: 100%, 75%, 50%, 25%
│   ├── Run key models on each subset
│   └── Plot: performance degradation curves
├── t-SNE embedding visualization
└── Statistical significance tests (paired t-test, all pairs)
```

**Deliverable:** All analysis figures and tables. Answers to RQ2 and RQ4.

### Week 8: Writing + Presentation

```
Tasks:
├── Thesis / Report (~35-45 pages)
│   ├── Ch 1: Introduction (2-3 pages)
│   │   └── Problem, motivation, contributions (3 bullet points)
│   ├── Ch 2: Literature Review (8-10 pages)
│   │   ├── Sequential recommendation (SASRec, BERT4Rec)
│   │   ├── Graph collaborative filtering (LightGCN, SGL, LightGCL)
│   │   ├── Hybrid approaches (SASGCN, TGT, TGSRec)
│   │   ├── Multi-behavior models (context, not our focus)
│   │   └── Research gap identification
│   ├── Ch 3: Methodology (5-7 pages)
│   │   ├── Problem formulation
│   │   ├── SASRec encoder
│   │   ├── LightGCN encoder
│   │   ├── Attention fusion gate
│   │   ├── Training objective
│   │   └── Complexity analysis
│   ├── Ch 4: Experimental Setup (4-5 pages)
│   │   ├── Datasets, preprocessing, splits
│   │   ├── Baselines and hyperparameters
│   │   └── Evaluation protocol
│   ├── Ch 5: Results & Analysis (8-10 pages)
│   │   ├── Main results (RQ1)
│   │   ├── Attention weight analysis (RQ2)
│   │   ├── Ablation study (RQ3)
│   │   ├── Sparsity analysis (RQ4)
│   │   └── Visualization
│   └── Ch 6: Conclusion & Future Work (2 pages)
├── Prepare presentation slides (15-20 slides)
└── Final code cleanup and documentation
```

---

## Part 6: Expected Results

### 6.1 Main Results Table (Hypothesis)

**Amazon-Beauty:**

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HR@10 |
|-------|-----------|-----------|---------|---------|-------|
| Popularity | ~0.015 | ~0.025 | ~0.008 | ~0.012 | ~0.025 |
| MF-BPR | ~0.030 | ~0.048 | ~0.018 | ~0.024 | ~0.050 |
| LightGCN | ~0.042 | ~0.065 | ~0.025 | ~0.033 | ~0.068 |
| SASRec | ~0.045 | ~0.070 | ~0.028 | ~0.036 | ~0.072 |
| BERT4Rec | ~0.043 | ~0.067 | ~0.026 | ~0.034 | ~0.069 |
| SASGCN | ~0.050 | ~0.076 | ~0.031 | ~0.040 | ~0.079 |
| **AdaFuse** | **~0.054** | **~0.082** | **~0.034** | **~0.043** | **~0.085** |

*Expected improvement: 5-8% over SASGCN, 15-20% over individual SASRec/LightGCN.*

### 6.2 Attention Weight Hypothesis

We expect to observe:
- Users with <10 interactions: α ≈ 0.3-0.4 (rely more on GNN collaborative signal)
- Users with 10-30 interactions: α ≈ 0.5 (balanced)
- Users with >30 interactions: α ≈ 0.6-0.7 (rely more on Transformer sequential signal)

This would confirm that the adaptive gate learns a meaningful fusion strategy.

---

## Part 7: Contribution Summary

### For the thesis:

> "We propose AdaFuse, a hybrid recommendation model that combines a Transformer encoder (SASRec) for sequential user behavior modeling with a Graph Neural Network (LightGCN) for collaborative filtering, connected through a learned per-user attention fusion gate. Unlike SASGCN (IEEE 2024) which uses static concatenation fusion, AdaFuse adaptively determines how much sequential vs. collaborative signal each user needs. Our experiments on Amazon-Beauty and Amazon-Book demonstrate that: (1) AdaFuse outperforms both individual components and static fusion baselines, (2) the learned fusion weight correlates with user interaction density — sparse users benefit more from GNN collaborative signal while active users benefit more from Transformer sequential signal, and (3) AdaFuse is more robust under data sparsity conditions."

### Three concrete contributions:
1. **AdaFuse model** — attention-based adaptive fusion of sequential (Transformer) and collaborative (GNN) representations, improving over SASGCN's static fusion
2. **Empirical analysis** — systematic study of when sequential vs. collaborative signal dominates, stratified by user activity level
3. **Robustness study** — sparsity-stratified evaluation showing the practical benefits of adaptive fusion

---

## Part 8: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AdaFuse doesn't beat SASGCN | Medium | High | The analysis contribution (RQ2: when does each signal help?) is independent of overall performance. Even if the gate learns α≈0.5 for everyone, understanding WHY is valuable. |
| Can't reproduce SASRec on Amazon-Book (no timestamps) | Medium | Medium | Use item list order as implicit sequence (standard practice). Primary experiments on Amazon-Beauty which has timestamps. |
| Attention gate converges to constant α | Low | Medium | Ablation already covers this case. Add regularization to encourage diverse α values. |
| LightGCN training is slow on Amazon-Book (3M interactions) | Low | Low | Use sparse matrix operations; 3-layer LightGCN is efficient. |
| Professor wants multi-behavior modeling | Medium | Medium | Frame as future work: "AdaFuse's gate mechanism naturally extends to behavior-specific fusion weights." Can pivot to Retailrocket dataset if needed. |

### Fallback Plans

**If AdaFuse shows no improvement:**
1. Pivot to analysis paper — "When Do Sequential and Collaborative Signals Help in Recommendation? An Empirical Study" — the stratified analysis is valuable regardless
2. Add contrastive auxiliary loss (L_CL between e_seq and e_col) — this creates a self-supervised training signal that may help

**If professor wants more ambition:**
1. Extend to multi-behavior: replace single gate with behavior-specific gates (one α per interaction type)
2. Extend to dynamic graph: update graph structure over time windows
3. Add item metadata (Amazon-Beauty has category/brand) as a third representation source with a three-way gate

---

## Part 9: Key References (Ordered by Relevance)

### Must-Cite (Core)
1. SASRec — Kang & McAuley, IEEE ICDM 2018
2. LightGCN — He et al., SIGIR 2020
3. **SASGCN** — Seng et al., IEEE Access 2024 ← **direct comparator**
4. BERT4Rec — Sun et al., CIKM 2019
5. BPR — Rendle et al., UAI 2009

### Should-Cite (Context)
6. NGCF — Wang et al., SIGIR 2019
7. TGT — Xia et al., IEEE TKDE 2022
8. MBHT — Yang et al., KDD 2022
9. SG-MST — Liao & Mok, IEEE ICDM 2024
10. MGUI — IEEE TCSS 2026
11. LightGCL — Cai et al., ICLR 2023
12. SimGCL — Yu et al., SIGIR 2022
13. SGL — Wu et al., SIGIR 2021

### Surveys
14. Graph Neural Networks in Recommender Systems: A Survey (Wu et al., 2020/2022)
15. Deep Learning based Recommender System Survey (Zhang et al., 2019)

---

## Part 10: Directory Structure

```
BTP/
├── REVISED_RESEARCH_PLAN.md       ← this document
├── RESEARCH_ANALYSIS_AND_PLAN.md  ← earlier draft (superseded)
├── AGENTS.md
├── src/
│   ├── data/
│   │   ├── dataset.py             ← data loading, graph construction, sequence building
│   │   └── preprocessing.py       ← filtering, splitting, ID mapping
│   ├── models/
│   │   ├── popularity.py          ← popularity baseline
│   │   ├── mf_bpr.py              ← MF + BPR baseline
│   │   ├── lightgcn.py            ← LightGCN baseline
│   │   ├── sasrec.py              ← SASRec baseline
│   │   ├── sasgcn.py              ← SASGCN comparator (GCN + SASRec + concat)
│   │   └── adafuse.py             ← AdaFuse (LightGCN + SASRec + attention gate)
│   ├── evaluation/
│   │   └── metrics.py             ← Recall, NDCG, HR, significance tests
│   ├── train.py                   ← unified training loop
│   ├── evaluate.py                ← unified evaluation script
│   └── config.py                  ← hyperparameter configs
├── configs/
│   ├── amazon_beauty.yaml
│   └── amazon_book.yaml
├── scripts/
│   ├── run_baselines.sh
│   ├── run_adafuse.sh
│   ├── run_ablation.sh
│   └── run_analysis.sh
├── results/                       ← auto-generated
│   ├── tables/
│   ├── figures/
│   └── logs/
└── roadmap/                       ← existing study materials
    └── data/                      ← existing datasets
```
