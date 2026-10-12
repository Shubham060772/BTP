# State of the Art: Graph + Attention for Recommendation (2024–2026)

> **Goal:** Survey the latest papers combining GNNs and attention for collaborative filtering. Identify gaps your BTP can target.

---

## 1. Where We Stand: LightGCN as the Baseline

LightGCN (2020) remains the dominant baseline in graph-based recommendation. Nearly every new paper compares against it. Key numbers on Amazon-Book (from the original paper):

```
LightGCN on Amazon-Book:
  Recall@20:  0.0411
  NDCG@20:    0.0315

Any new method needs to beat these convincingly.
```

The main directions for improvement over LightGCN:

```
Direction 1: Self-supervised / Contrastive Learning
  SGL (2021), SimGCL (2022), LightGCL (2023), GFormer (2023)

Direction 2: Disentangled / Multi-Intent Representations
  DGCF (2020), DCCF (2023), MacGCN

Direction 3: Graph Transformers
  GFormer (2023), PGTR (2024), DGSARec (2026)

Direction 4: Knowledge / Side Information
  KGAT (2019), ARGC (2026), multimodal methods

Direction 5: Frequency / Spectral Methods
  FourierKAN (2025), WaveHDNN (2025), Spectral reasoning
```

---

## 2. Recent Papers — Detailed Analysis

### 2.1 PGTR — Position-aware Graph Transformer for Recommendation (2024)

**Paper:** arXiv 2412.18731

**Problem:** LightGCN's fixed-weight aggregation ignores relative node positions in the graph. Graph Transformers capture global patterns but lose local structure.

**Method:**
```
Architecture:

Step 1: GCN layers produce local neighborhood embeddings
Step 2: Position encoding via graph distance / random walk
Step 3: Transformer attends globally with position-aware bias

  Local GCN    ──┐
                  ├── Position-aware Transformer ──> Final embeddings
  Position PE  ──┘

Attention(i,j) = (Q_i · K_j)/√d + bias(position(i,j))
```

**Key Innovation:** Combines local GCN features with global Transformer attention using position-aware attention bias. The position encoding tells the transformer *how far apart* nodes are in the graph.

**Results vs LightGCN:** Reported improvements on standard benchmarks, stronger on datasets with long-range dependencies.

**Limitations/Gaps:**
- Position encoding computation adds overhead
- Not clear how it handles very sparse regions
- No disentanglement of user intents

---

### 2.2 FourierKAN-GCF — Fourier KAN for Graph CF (CIKM 2025)

**Paper:** arXiv 2406.01034

**Problem:** LightGCN uses no feature transformation at all (by design). But some non-linearity could help without the over-parameterization of NGCF.

**Method:**
```
LightGCN:     aggregate → no transform → average layers
NGCF:         aggregate → heavy MLP transform → concatenate layers
FourierKAN:   aggregate → lightweight Fourier transform → average layers

The Fourier KAN (Kolmogorov-Arnold Network) transform:
  f(x) = Σ a_k · cos(k·x) + b_k · sin(k·x)

This adds learnable non-linearity with very few parameters.
Only a handful of Fourier coefficients per layer.
```

**Key Innovation:** Found the sweet spot between LightGCN (no transform) and NGCF (heavy transform). Fourier basis is parameter-efficient and naturally captures periodic patterns in embeddings.

**Results:** Accepted at CIKM 2025. Improvements over LightGCN while adding minimal parameters.

**Limitations/Gaps:**
- Only uses Fourier basis — other spectral bases unexplored
- No contrastive learning component
- Not combined with attention mechanisms

---

### 2.3 DGSARec — Dynamic Graph with Similarity-Aware Attention (2026)

**Paper:** arXiv 2605.05238

**Problem:** Static graphs miss temporal dynamics. User preferences evolve over time.

**Method:**
```
Step 1: Construct dynamic graph snapshots
  t=1: User1──ProdA, User2──ProdB
  t=2: User1──ProdC, User2──ProdA  (preferences shift)

Step 2: Similarity-aware attention
  Attention weights consider node similarity, not just connectivity
  Similar users/products get higher attention even without direct edges

Step 3: Graph Transformer fusion
  Combine dynamic graph signals with transformer for temporal modeling
```

**Key Innovation:** Attention is based on embedding similarity, not just graph adjacency. This allows information flow between structurally disconnected but semantically similar nodes.

**Limitations/Gaps:**
- Temporal snapshots add complexity
- Not benchmarked against all recent baselines yet (2026 preprint)
- Dynamic graph construction is expensive

---

### 2.4 ARGC — Adaptive Relation-weighted Graph Convolution (2026)

**Paper:** arXiv 2602.21099

**Problem:** LLM-generated semantic signals are underutilized in graph recommendation.

**Method:**
```
Step 1: Use LLM to generate attribute descriptions for products
  Product A → "affordable daily skincare moisturizer for sensitive skin"

Step 2: Build heterogeneous graph with semantic edges
  Product A ──semantic_sim──> Product B  (if LLM embeddings are close)
  Product A ──category──> Skincare
  Product A ──purchased_by──> User 1

Step 3: Adaptive relation-weighted GCN
  Different edge types get different learned weights
  w_semantic, w_category, w_purchase — all learned
```

**Key Innovation:** Bridges LLMs and GNNs. LLM-generated semantics create new edges that complement collaborative signals.

**Limitations/Gaps:**
- Depends on LLM quality and cost
- Semantic similarity can be noisy
- Limited to items with textual descriptions

---

### 2.5 WaveHDNN — Wavelet Hypergraph Diffusion (2025)

**Paper:** arXiv 2501.16722

**Problem:** Standard GNNs operate in spatial domain only. Spectral methods can capture multi-scale patterns.

**Method:**
```
Step 1: Build hypergraph from user-item interactions
  Hyperedge: group of users with similar purchase patterns

Step 2: Wavelet transform on hypergraph
  Decompose signals into frequency bands
  Low frequency: broad community patterns
  High frequency: individual-level differences

Step 3: Diffusion with attention across frequency bands
  Learn which frequency bands matter for each user
```

**Key Innovation:** Multi-scale analysis via wavelets. Different users may be best modeled at different frequency scales.

**Limitations/Gaps:**
- Wavelet computation is expensive on large graphs
- Hypergraph construction choices affect results significantly
- Complex pipeline — hard to reproduce and ablate

---

### 2.6 LightGCL — Light Graph Contrastive Learning (WWW 2023)

**Paper:** A strong recent baseline worth knowing.

**Problem:** SGL's random augmentation can corrupt important graph structures.

**Method:**
```
SGL augmentation: randomly drop edges → can break important connections
LightGCL augmentation: SVD-based augmentation → preserves global structure

Steps:
  1. Compute truncated SVD of adjacency: A ≈ U_k Σ_k V_k^T
  2. Use SVD reconstruction as augmented view
  3. Contrastive learning between original and SVD view

The SVD view smooths noise while preserving dominant collaborative signals.
```

**Results on Amazon-Book:** Consistent 3-6% improvement over SGL.

**Limitations/Gaps:**
- SVD recomputation is expensive (but only needed once)
- Fixed SVD rank — not adaptive per user
- No attention mechanism

---

## 3. Summary Table

| Paper | Year | Venue | Base Model | Enhancement | Key Idea | Over LightGCN |
|-------|------|-------|-----------|-------------|----------|---------------|
| SGL | 2021 | SIGIR | LightGCN | Contrastive | Edge/node dropout augmentation | +5-8% Recall |
| DGCF | 2020 | SIGIR | GCN | Attention | Disentangled intent modeling | +3-7% Recall |
| LightGCL | 2023 | WWW | LightGCN | Contrastive | SVD-based augmentation | +3-6% Recall |
| GFormer | 2023 | SIGIR | Graph Transformer | Contrastive | Learned augmentation | +4-8% Recall |
| PGTR | 2024 | — | GCN + Transformer | Attention | Position-aware graph transformer | Reported gains |
| FourierKAN | 2025 | CIKM | LightGCN | Spectral | Fourier KAN feature transform | Reported gains |
| WaveHDNN | 2025 | — | Hypergraph | Spectral | Wavelet multi-scale diffusion | Reported gains |
| DGSARec | 2026 | — | Dynamic Graph | Attention | Similarity-aware temporal attention | Preprint |
| ARGC | 2026 | — | Het. GCN | LLM + GNN | LLM-augmented semantic edges | Preprint |

---

## 4. Identifying Gaps for Your BTP

### Gap 1: Attention + Contrastive (underexplored combination)

```
What exists:
  SGL: contrastive + random augmentation (no attention)
  GAT: attention (no contrastive)
  GFormer: transformer + learned augmentation (complex)

Gap: Simple attention-weighted contrastive learning
  Use GAT attention to weight which edges to augment
  High-attention edges = important → don't drop them
  Low-attention edges = noisy → safe to drop

  This is a principled augmentation strategy:
  the model itself decides what to augment.
```

**Feasibility:** Medium. Requires GAT + contrastive pipeline. Reasonable for a BTP.

### Gap 2: Adaptive Layer Attention + Spectral Transform

```
What exists:
  LightGCN: uniform layer averaging
  FourierKAN: spectral transform but fixed layer combination
  Various: attention across layers but no spectral component

Gap: Learn per-user layer importance with lightweight spectral transform
  Different users benefit from different propagation depths.
  A user with many purchases → deeper layers help (community patterns)
  A user with few purchases → shallow layers help (direct neighbors)

  Combine: Fourier transform per layer + attention across layers
```

**Feasibility:** High. Extension of LightGCN with two small additions. Clean ablation story.

### Gap 3: Disentangled Intents + Contrastive Learning

```
What exists:
  DGCF: disentangled intents (no contrastive)
  SGL: contrastive (no disentanglement)

Gap: Contrastive learning at the intent level
  Instead of contrasting whole user embeddings,
  contrast each intent channel separately.

  Intent 1 (skincare): augmented views should agree
  Intent 2 (gifting): augmented views should agree
  → Forces each intent to be individually robust
```

**Feasibility:** Medium-High. Clear improvement story, clean experiments.

### Gap 4: Knowledge Graph + Contrastive (with Amazon metadata)

```
What exists:
  KGAT: knowledge graph attention (no contrastive)
  SGL: contrastive (no knowledge graph)

Gap: Contrastive views from KG vs collaborative graph
  View 1: User-Item collaborative graph (who bought what)
  View 2: Item-Attribute knowledge graph (product metadata)

  Contrastive alignment: product embeddings from both views
  should agree, even though the signals are different.

  Amazon-Beauty has rich metadata (category, brand, price)
  → natural fit for this approach.
```

**Feasibility:** High. You have the metadata in Amazon-Beauty. Clear contribution.

---

## 5. Recommended BTP Directions (Ranked)

Based on feasibility, novelty, and clarity of contribution:

```
Rank 1: Gap 4 — Knowledge Graph + Contrastive
  Why: You have Amazon-Beauty metadata ready.
       KGAT + SGL haven't been properly combined.
       Clean story: "collaborative + semantic contrastive views."
       Easy to ablate: remove KG → baseline, remove contrastive → baseline.

Rank 2: Gap 2 — Adaptive Layer Attention + Spectral
  Why: Minimal added complexity over LightGCN.
       Strong ablation story.
       Novel angle: user-adaptive propagation depth.

Rank 3: Gap 3 — Disentangled + Contrastive
  Why: Clean combination of two proven ideas.
       Intent-level contrastive is underexplored.
       But: DGCF implementation is complex.

Rank 4: Gap 1 — Attention-guided Augmentation
  Why: Elegant idea but hard to show large gains.
       May be marginal over random augmentation.
```

---

## 6. Positioning Your BTP

### What reviewers / supervisors look for:

```
1. Clear problem statement
   "LightGCN ignores [specific limitation]"

2. Principled solution
   "We address this by combining [technique A] + [technique B]"

3. Strong baselines
   LightGCN, SGL, NGCF — compare against all of them

4. Clean ablations
   Remove each component → show it matters

5. Analysis beyond numbers
   Attention visualization, per-group performance, cold-start analysis
```

### Template for your BTP abstract:

```
"Graph-based collaborative filtering methods like LightGCN have shown
strong performance on product recommendation. However, they [GAP].
We propose [METHOD NAME], which combines [TECHNIQUE 1] with
[TECHNIQUE 2] to address this limitation. Specifically, we [HOW].
Experiments on Amazon-Book and Amazon-Beauty show that our method
achieves [X]% improvement in Recall@20 over LightGCN and outperforms
recent baselines including SGL and [OTHER]. We further provide
[ANALYSIS] to demonstrate [INSIGHT]."
```

---

## 7. Exercises

1. **Reproduce LightGCN baseline.** Run LightGCN on Amazon-Book and verify you get Recall@20 ≈ 0.0411. This is your ground truth.

2. **Reproduce SGL.** Add edge dropout contrastive learning. Verify you beat LightGCN. This is your second baseline.

3. **Pick a gap.** Choose one of the four gaps above. Write a 1-page proposal: problem, method, expected experiments, timeline.

4. **Literature check.** For your chosen gap, search arXiv for any paper that already fills it. If someone published it in 2025-2026, pivot to a different gap or find a further refinement.

5. **Prototype.** Implement the simplest possible version of your proposed method. Even if it doesn't beat baselines yet, verify it trains and converges. Debug the pipeline before tuning.
