# Phase 8 — Recommendation-Specific GNNs (Week 5–6)

> This is the **most important phase** of your BTP learning. Everything before this was building blocks. Everything after this builds on what you learn here.

---

## Why This Phase Matters

You've learned GCN, GraphSAGE, and GAT — general-purpose GNN architectures designed for tasks like node classification and link prediction on generic graphs. But recommendation has a **specific structure** that these general models don't exploit:

```
General graph:            Recommendation graph:

  A ── B ── C             User ── Item
  │         │             (bipartite)
  D ── E ── F
  (arbitrary topology)    All edges cross between two node types.
                          No user-user or item-item edges.
```

Recommendation-specific GNNs exploit this structure. The two models you must understand are **NGCF** and **LightGCN**.

---

## 1. Why Generic GNNs Aren't Optimal for Recommendation

### Problem 1: Unnecessary Complexity

Standard GCN applies feature transformation and nonlinear activation at every layer:

$$
H^{(l+1)} = \sigma\!\left(\hat{A}\,H^{(l)}\,W^{(l)}\right)
$$

But in collaborative filtering, nodes have **no input features** — only learnable embedding IDs. The feature transformation $W$ and nonlinearity $\sigma$ are operating on **learned embeddings**, not rich features.

LightGCN's authors showed that these operations **don't help** and actually **hurt** performance in CF.

### Problem 2: Wrong Inductive Bias

GCN was designed for graphs where nodes have meaningful features (e.g., paper abstracts in citation networks). In CF:

```
Citation network:                Recommendation:
  Node = paper                   Node = user or item
  Features = word counts (1433D) Features = NONE (just an ID)
  Task = classify paper topic    Task = predict interactions

The feature transformation W makes sense    W is transforming a random
when features carry information.            learned vector — it just adds
                                            parameters without benefit.
```

### Problem 3: Task Mismatch

Generic GNNs output node embeddings for classification. Recommendation needs **user-item compatibility scores** for ranking.

---

## 2. NGCF — Neural Graph Collaborative Filtering

**Paper:** Wang et al. (2019) — "Neural Graph Collaborative Filtering"

NGCF was the first major paper to apply GNNs specifically to collaborative filtering. It's important to understand because LightGCN is defined as an improvement over it.

### 2.1 Core Idea

NGCF encodes the collaborative signal (who interacted with what) directly into the embedding process using the user-item interaction graph.

```
Traditional MF:                    NGCF:
  User embed = e_u                 User embed = GNN(graph, e_u)
  Item embed = e_i                 Item embed = GNN(graph, e_i)
  Score = e_u · e_i                Score = GNN_u · GNN_i

MF: embeddings are independent    NGCF: embeddings are influenced
    of graph structure                   by graph neighbors
```

### 2.2 NGCF Propagation Rule

At layer $l$, for a user $u$ with neighbor items $\mathcal{N}_u$:

$$
e_u^{(l+1)} = \sigma\!\left(W_1^{(l)}\,e_u^{(l)} + \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}\left(W_1^{(l)}\,e_i^{(l)} + W_2^{(l)}\,(e_i^{(l)} \odot e_u^{(l)})\right)\right)
$$

Let's break this down:

```
NGCF update for user u at layer l:

  e_u^(l+1) = σ(
      W₁ · e_u^(l)                              ← self-connection
    + Σ  1/√(|N_u|·|N_i|) · (                   ← normalized neighbor sum
          W₁ · e_i^(l)                           ← neighbor transform
        + W₂ · (e_i^(l) ⊙ e_u^(l))              ← feature INTERACTION
      )
  )

Key components:
  W₁: weight matrix for feature transformation
  W₂: weight matrix for feature interaction
  ⊙:  element-wise product (feature interaction)
  σ:  nonlinear activation (LeakyReLU)
```

### 2.3 Feature Interaction

The term $W_2\,(e_i \odot e_u)$ is the **feature interaction** between user and item embeddings. This is what makes NGCF different from just applying GCN — it encodes **how** the user and item are related, not just that they are connected.

```
User embedding:  [0.3, 0.8, 0.1, 0.5]
Item embedding:  [0.7, 0.2, 0.9, 0.4]
                   ↓
Element-wise ⊙:  [0.21, 0.16, 0.09, 0.20]

This captures: "which dimensions of user and item BOTH activate"
Then W₂ transforms this interaction into a useful signal.
```

### 2.4 Multi-Layer Aggregation in NGCF

After $L$ layers, NGCF concatenates embeddings from all layers:

$$
e_u^* = e_u^{(0)} \| e_u^{(1)} \| \ldots \| e_u^{(L)}
$$

$$
e_i^* = e_i^{(0)} \| e_i^{(1)} \| \ldots \| e_i^{(L)}
$$

Then the prediction:

$$
\hat{y}_{ui} = {e_u^*}^T \cdot e_i^*
$$

```
Layer 0: initial embeddings (direct preferences)
Layer 1: 1-hop neighbor info (items of user, users of item)
Layer 2: 2-hop neighbor info (users who liked similar items)
Layer 3: 3-hop neighbor info (broader collaborative signal)

Concatenation captures ALL levels of collaborative signal.

User u's final embedding:
  [e_u^(0) | e_u^(1) | e_u^(2) | e_u^(3)]
   direct    1-hop     2-hop     3-hop
```

### 2.5 NGCF Limitations

Despite being a landmark paper, NGCF has problems:

1. **Too many parameters:** Two weight matrices ($W_1$, $W_2$) per layer
2. **Feature interaction may not help:** The $\odot$ interaction adds complexity
3. **Nonlinearity hurts:** For CF, $\sigma$ (activation function) degrades performance
4. **Training difficulty:** More parameters → harder to train, more overfitting risk

These limitations directly motivated LightGCN.

---

## 3. LightGCN — The Central Model for Your BTP

**Paper:** He et al. (2020) — "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"

> **Read this paper.** It should be the central paper of your BTP if you're doing CF-based GNN recommendation.

### 3.1 The Key Insight

The LightGCN authors asked:

> "Which components of GCN actually matter for collaborative filtering?"

They systematically removed components and found:

```
Full GCN for CF:
  Feature transformation (W)  → REMOVE (hurts performance)
  Nonlinear activation (σ)    → REMOVE (hurts performance)
  Self-connection              → REMOVE (handled differently)
  Neighbor aggregation         → KEEP (this is what matters!)

Conclusion: For CF, ONLY neighborhood aggregation matters.
```

This is a **profound** result. It means that for collaborative filtering, the complex neural operations in GCN/NGCF are not just unnecessary — they actively make things worse.

### 3.2 The LightGCN Propagation Rule

Beautifully simple:

$$
e_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}}\;e_i^{(l)}
$$

$$
e_i^{(l+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i|}\sqrt{|\mathcal{N}_u|}}\;e_u^{(l)}
$$

In words:

```
To update a user's embedding:
  1. Take all items the user interacted with
  2. Normalize each item's embedding by both degrees
  3. Sum them up
  That's it. No W. No σ. No feature interaction.

To update an item's embedding:
  1. Take all users who interacted with the item
  2. Normalize each user's embedding by both degrees
  3. Sum them up
```

In matrix form (on the full bipartite graph):

$$
E^{(l+1)} = \tilde{A}\,E^{(l)}
$$

where $\tilde{A} = D^{-1/2} A\, D^{-1/2}$ is the symmetrically normalized adjacency matrix, and $E^{(l)}$ stacks all user and item embeddings at layer $l$.

### 3.3 Layer Combination

Instead of concatenation (NGCF), LightGCN uses a **weighted sum** of all layers:

$$
e_u = \sum_{l=0}^{L} \alpha_l\,e_u^{(l)}
$$

where $\alpha_l$ is typically set to $\frac{1}{L+1}$ (uniform weighting).

```
Why weighted sum instead of concatenation?

Concatenation: [e^(0) | e^(1) | e^(2)]
  → Embedding dimension grows: d × (L+1)
  → Need more parameters in prediction layer

Weighted sum: (e^(0) + e^(1) + e^(2)) / 3
  → Embedding dimension stays: d
  → Simpler, fewer parameters
  → Works equally well or better for CF
```

### 3.4 Full Architecture Walkthrough

Here is the complete LightGCN pipeline:

```
Step 1: Initialize learnable embeddings
  ┌─────────────────────────────────┐
  │  User embeddings: E_u  [N_u, d] │
  │  Item embeddings: E_i  [N_i, d] │
  │  (randomly initialized)          │
  └─────────────────────────────────┘
                  │
                  ↓
Step 2: Stack into full embedding matrix
  ┌─────────────────────────────────┐
  │  E^(0) = [E_u ; E_i]  [N, d]   │
  │  where N = N_u + N_i            │
  └─────────────────────────────────┘
                  │
                  ↓
Step 3: Graph propagation (L layers)
  ┌─────────────────────────────────┐
  │  E^(1) = Ã · E^(0)             │
  │  E^(2) = Ã · E^(1)             │
  │  E^(3) = Ã · E^(2)             │
  │  ...                             │
  │  (no W, no σ at any layer)       │
  └─────────────────────────────────┘
                  │
                  ↓
Step 4: Layer combination
  ┌─────────────────────────────────┐
  │  E_final = 1/(L+1) · Σ E^(l)   │
  │          = (E^(0)+E^(1)+...)/   │
  │            (L+1)                 │
  └─────────────────────────────────┘
                  │
                  ↓
Step 5: Split back into user/item
  ┌─────────────────────────────────┐
  │  E_u_final = E_final[:N_u]      │
  │  E_i_final = E_final[N_u:]      │
  └─────────────────────────────────┘
                  │
                  ↓
Step 6: Predict scores
  ┌─────────────────────────────────┐
  │  ŷ_ui = E_u_final[u] · E_i_final[i] │
  │  (dot product)                   │
  └─────────────────────────────────┘
                  │
                  ↓
Step 7: BPR loss + train
  ┌─────────────────────────────────┐
  │  L = -Σ log σ(ŷ_ui - ŷ_uj)    │
  │  + λ · regularization            │
  └─────────────────────────────────┘
```

### 3.5 What Each Layer Captures

```
Layer 0: e_u^(0) — the raw embedding of user u
         This captures the user's intrinsic preference.

Layer 1: e_u^(1) = Σ (normalized) e_i^(0)  for items i that u interacted with
         This captures: "What items has this user liked?"

Layer 2: e_u^(2) = Σ (normalized) e_i^(1)
                  = Σ (normalized) Σ (normalized) e_u'^(0)
         This captures: "What do OTHER USERS who liked the
         same items look like?"

Layer 3: e_u^(3) captures: "What items do those similar users like?"
```

Visually, with a concrete example:

```
         Layer 0          Layer 1          Layer 2

User A ─── Item 1         Item 1:          User A gets info about
User A ─── Item 2         agg(User A,      User B (through Item 1),
User B ─── Item 1              User B)     User C (through Item 2)
User C ─── Item 2         Item 2:
User C ─── Item 3         agg(User A,      → User A's layer-2 embedding
                               User C)       now knows about Item 3
                                              (via User C)!
                         User A:
                         agg(Item 1,       This is the collaborative signal.
                              Item 2)
```

### 3.6 Why Simpler Is Better for CF

The paper provides this comparison:

```
NGCF components:                Effect on CF performance:
  Feature transformation (W)     HURTS ↓
  Nonlinear activation (σ)       HURTS ↓
  Feature interaction (⊙)        HURTS ↓
  Neighbor aggregation            HELPS ↑

Removing ALL neural components → BETTER performance!
```

**Intuition:** In CF, embeddings are learned from scratch (not from input features). Applying $W$ and $\sigma$ to learned embeddings introduces unnecessary distortion. The embeddings can learn to be "GNN-ready" without needing transformation — the only useful operation is aggregating neighborhood structure.

### 3.7 Implementation in PyTorch

```python
import torch
import torch.nn as nn
from torch_geometric.utils import degree

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # Only learnable parameters: initial embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_norm_adj(self, edge_index, num_nodes):
        """Compute D^{-1/2} A D^{-1/2}"""
        row, col = edge_index
        deg = degree(row, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_index, norm

    def forward(self, edge_index):
        num_nodes = self.num_users + self.num_items

        # Step 1: Stack user and item embeddings
        e0_user = self.user_embedding.weight
        e0_item = self.item_embedding.weight
        all_emb = torch.cat([e0_user, e0_item], dim=0)  # [N, d]

        # Step 2: Compute normalized adjacency
        edge_index, norm = self.compute_norm_adj(edge_index, num_nodes)

        # Step 3: Graph propagation
        embs = [all_emb]  # layer 0
        x = all_emb
        for layer in range(self.num_layers):
            # Sparse message passing: x' = Ã · x
            row, col = edge_index
            x = torch.zeros_like(all_emb)
            x.index_add_(0, row, norm.unsqueeze(1) * embs[-1][col])
            embs.append(x)

        # Step 4: Layer combination (mean)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)  # [N, d]

        # Step 5: Split back
        user_emb = final_emb[:self.num_users]
        item_emb = final_emb[self.num_users:]

        return user_emb, item_emb

    def predict(self, user_emb, item_emb, users, items):
        """Predict scores for (user, item) pairs"""
        u = user_emb[users]  # [batch, d]
        i = item_emb[items]  # [batch, d]
        return (u * i).sum(dim=1)  # dot product → [batch]
```

### 3.8 Training LightGCN

```python
model = LightGCN(num_users=6040, num_items=3706,
                  embed_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    user_emb, item_emb = model(data.edge_index)

    # Sample BPR triplets
    users, pos_items, neg_items = sample_bpr_triplets(data)

    pos_scores = model.predict(user_emb, item_emb, users, pos_items)
    neg_scores = model.predict(user_emb, item_emb, users, neg_items)

    # BPR loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    # L2 regularization on embeddings
    reg = (model.user_embedding(users).norm(2).pow(2) +
           model.item_embedding(pos_items).norm(2).pow(2) +
           model.item_embedding(neg_items).norm(2).pow(2))
    loss += 1e-4 * reg / len(users)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 4. User-Item Bipartite Graph Construction

### 4.1 From Interactions to Graph

```
Raw data:
  User 1 → purchased Product A (5 stars)
  User 1 → purchased Product C (4 stars)
  User 2 → purchased Product A (3 stars)
  User 2 → purchased Product B (5 stars)

Bipartite graph (implicit feedback — ignore rating values):
  User 1 ─── Product A
  User 1 ─── Product C
  User 2 ─── Product A
  User 2 ─── Product B

Adjacency matrix:
              Product A  Product B  Product C
  User 1     [  1          0          1  ]
  User 2     [  1          1          0  ]

Full graph adjacency (users + items):
              U1  U2  PA  PB  PC
  User 1     [ 0   0   1   0   1 ]
  User 2     [ 0   0   1   1   0 ]
  Product A  [ 1   1   0   0   0 ]
  Product B  [ 0   1   0   0   0 ]
  Product C  [ 1   0   0   0   0 ]
```

### 4.2 Why Bipartite Matters

The bipartite structure means:

```
Odd layers:  information flows User → Item → User → ...
Even layers: information flows Item → User → Item → ...

Layer 1: Users aggregate from their Items
Layer 2: Items aggregate from their Users (who aggregated from Items)
Layer 3: Users aggregate from Items (who aggregated from Users who...)

This naturally captures collaborative filtering!
  "Users who liked similar items will get similar embeddings"
```

---

## 5. Embedding Propagation Mechanics

### 5.1 What Propagation Does Geometrically

```
Before propagation (layer 0):
  User embeddings and Item embeddings are RANDOM.
  No structure, no collaborative signal.

  Embedding space:
    u1 •      • i2
         • i1
    u2 •           • i3

After propagation (layer 1+):
  Users who interact with the same items → pulled CLOSER.
  Items interacted with by the same users → pulled CLOSER.

  Embedding space:
    u1 • • i1
    u2 •    • i2
             • i3

  u1 and u2 are closer because they share Item i1.
  i1 and i2 are closer because User u1 interacted with both.
```

### 5.2 Degree Normalization

The normalization factor $\frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}}$ prevents popular items and active users from dominating.

```
Without normalization:
  A bestseller (purchased by 100K users) would DOMINATE
  every user's embedding who bought it.

With symmetric normalization:
  The bestseller contributes LESS to each user (because it has high degree)
  A niche product contributes MORE (because it has low degree)

This is analogous to TF-IDF in text:
  Common words (the, a, is) → low weight
  Rare words (quantum, enzyme) → high weight
```

---

## 6. Layer Aggregation Strategies

### 6.1 Concatenation (NGCF style)

$$
e_u^* = e_u^{(0)} \| e_u^{(1)} \| \ldots \| e_u^{(L)}
$$

- Final dimension: $d \times (L+1)$
- Preserves information from each layer separately
- More parameters in downstream prediction

### 6.2 Weighted Sum (LightGCN style)

$$
e_u^* = \sum_{l=0}^{L} \alpha_l\,e_u^{(l)} \quad \text{where} \quad \alpha_l = \frac{1}{L+1}
$$

- Final dimension: $d$ (unchanged)
- Simpler, fewer parameters
- Uniform weighting works well in practice
- Can also learn $\alpha_l$ — but uniform is usually sufficient

### 6.3 Last-Layer Only

$$
e_u^* = e_u^{(L)}
$$

- Only uses the final layer
- Loses direct embedding signal from layer 0
- Generally worse than combining layers

```
Comparison:
  Method          | Recall@20 on Gowalla (from paper)
  ─────────────── | ───────────────────────────────────
  Concat (NGCF)   | 0.1547
  Weighted sum     | 0.1830  ← best
  Last layer only  | 0.1399
  Layer 0 only     | 0.1554
```

---

## 7. NGCF vs LightGCN — Head-to-Head

| Aspect | NGCF | LightGCN |
|---|---|---|
| **Weight matrices** | $W_1$, $W_2$ per layer | None |
| **Activation** | LeakyReLU | None |
| **Feature interaction** | $e_i \odot e_u$ | None |
| **Self-connection** | Via $W_1 e_u$ | Via layer combination |
| **Layer combination** | Concatenation | Weighted sum |
| **Parameters** | Many | Only embeddings |
| **Performance** | Good | Better (simpler!) |
| **Training speed** | Slower | Faster |
| **Overfitting risk** | Higher | Lower |

### Results from the LightGCN paper:

```
Dataset: Gowalla
  Model      Recall@20   NDCG@20
  MF         0.1291      0.1109
  NGCF       0.1547      0.1327
  LightGCN   0.1830      0.1554   ← +18% over NGCF

Dataset: Yelp2018
  Model      Recall@20   NDCG@20
  MF         0.0543      0.0442
  NGCF       0.0579      0.0477
  LightGCN   0.0649      0.0530   ← +12% over NGCF

Dataset: Amazon-Book
  Model      Recall@20   NDCG@20
  MF         0.0322      0.0250
  NGCF       0.0344      0.0263
  LightGCN   0.0411      0.0315   ← +20% over NGCF
```

**LightGCN beats NGCF on every benchmark while being simpler.**

---

## 8. Why LightGCN Should Be Your Central BTP Paper

### For Your Thesis

1. **It's the state-of-the-art baseline** for graph-based collaborative filtering
2. **It's simple enough to implement from scratch** — ideal for a BTP
3. **It clearly shows why design matters** — simpler can be better
4. **It provides clear comparison points** — MF, NGCF, GCN as baselines
5. **It opens multiple research directions** — you can propose improvements

### For Your Thesis Narrative

```
Your BTP story:

Chapter 2: Background
  "Collaborative filtering can be formulated as a graph problem..."
  "GCN can be applied to user-item graphs..."
  "NGCF was the first major GNN recommender..."

Chapter 3: Methodology
  "LightGCN showed that for CF, only neighborhood aggregation matters..."
  "We adopt LightGCN as our base model because..."

Chapter 4: Proposed Approach (your novelty)
  "We extend LightGCN by incorporating [your idea]..."

Chapter 5: Experiments
  "We compare against Popularity, MF, NGCF, and LightGCN..."
```

### For Your Defense

```
Q: "Why not use a standard GCN?"
A: "The LightGCN paper showed that feature transformation and
    nonlinear activation hurt CF performance. Our base model
    follows this finding."

Q: "Why not use GraphSAGE or GAT?"
A: "GraphSAGE's inductive capability isn't needed for our static
    dataset. GAT's attention mechanism adds complexity that
    LightGCN's authors showed doesn't help for pure CF."

Q: "What's your contribution beyond LightGCN?"
A: "We investigate [your modification] and show that it
    improves performance under [specific conditions]."
```

---

## 9. Beyond LightGCN — Other Models to Be Aware Of

You don't need to implement all of these, but knowing they exist strengthens your literature review:

### UltraGCN (2021)

Skips explicit message passing entirely — approximates infinite-layer GCN.

### GCCF (2020)

Similar insight to LightGCN — removes nonlinearities from GCN for CF.

### PinSage (2018, Pinterest)

Industrial-scale GNN recommendation. Uses GraphSAGE with random walks for neighborhood sampling on billion-node graphs.

### SGL (2021)

Self-supervised graph learning for recommendation — adds contrastive learning to LightGCN.

```
Evolution of GNN recommendation:

  GCN (generic)
    ↓
  NGCF (first GNN recommender) — 2019
    ↓
  LightGCN (simplified, better) — 2020  ← your starting point
    ↓
  UltraGCN, SGL, etc. (advanced) — 2021+
```

---

## Exercises

1. **Ablation study (conceptual):** Given the NGCF propagation rule, explain what happens if you (a) remove $W_2$ and the feature interaction, (b) then also remove $\sigma$, (c) then also remove $W_1$. What are you left with?

2. **Layer analysis:** For a user who purchased products A, B, C — and products A and B were also purchased by user 2, while product C was only purchased by user 1 — trace through 2 layers of LightGCN propagation. What information does the user's layer-2 embedding contain?

3. **Implementation:** Implement LightGCN from scratch using PyTorch (not PyG). Use sparse matrix multiplication for the propagation step. Test on a small synthetic graph.

4. **Paper reading:** Read the LightGCN paper (He et al., 2020). In particular, study Table 2 (ablation of GCN components) and explain why each removed component hurts performance.

5. **Comparison:** On the Amazon-Book dataset, compare the number of trainable parameters between NGCF (with hidden dim 64 and 3 layers) and LightGCN (with embed dim 64 and 3 layers). How many times more parameters does NGCF have?

---

## Key Takeaways

1. **Generic GNNs aren't optimal for recommendation** because they apply unnecessary feature transformations and nonlinearities to learned embeddings.

2. **NGCF** was the first major GNN recommender — it works, but it's overparameterized for the CF task.

3. **LightGCN** removes all neural components except neighborhood aggregation and achieves **better** performance with **fewer** parameters. This is the model you should build your BTP around.

4. **The only thing that matters for graph-based CF is neighborhood aggregation** — propagating information through the user-item bipartite graph to encode collaborative signals into embeddings.

5. **Layer combination** (weighted sum of all layers) is important — it combines direct embeddings with multi-hop collaborative signals.

6. **LightGCN is both simple to implement and state-of-the-art** — making it ideal as a BTP base model that you can then extend with your own contribution.
