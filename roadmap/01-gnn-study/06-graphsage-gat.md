# Phase 6 — GraphSAGE and GAT (Week 4)

## Why This Phase Matters

In the previous phase you learned GCN — a powerful but limited architecture. Before you commit to a GNN model for your BTP, you need to understand the **alternatives** and their trade-offs. During your thesis defense, you will almost certainly be asked:

> "Why did you choose this GNN architecture over others?"

This phase gives you the knowledge to answer that question confidently.

---

## 1. Limitations of GCN

Recall the GCN update rule:

$$
H^{(l+1)} = \sigma\!\left(\hat{D}^{-1/2}\,\hat{A}\,\hat{D}^{-1/2}\,H^{(l)}\,W^{(l)}\right)
$$

GCN has two key limitations for recommendation:

### 1.1 Transductive Learning

GCN operates on a **fixed** graph. The weight matrix $W$ and the normalized adjacency $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ are computed over the entire graph at training time.

```
Training graph:
  User A ── Item 1
  User A ── Item 2
  User B ── Item 1

New user at inference time:
  User C ── Item 3    ← GCN can't handle this without retraining!
```

**Why this matters for recommendations:** New users and new items appear constantly. A model that needs retraining for every new node is impractical.

### 1.2 Full-Graph Computation

GCN multiplies by the **full** adjacency matrix each layer. For a recommendation graph with millions of users and items, this is extremely expensive.

```
Layer computation:
  H' = Â · H · W

If the graph has 1M nodes:
  Â is 1M × 1M   ← doesn't fit in GPU memory
```

### 1.3 Equal Neighbor Weighting

GCN treats all neighbors equally (up to degree normalization). But intuitively:

```
User Alice:
  ├── watched "Inception"      (loved it, watched 3 times)
  ├── watched "Titanic"        (clicked by accident)
  └── watched "Interstellar"   (loved it, watched 5 times)

GCN: all three contribute roughly equally to Alice's embedding
Reality: Inception and Interstellar should matter much more
```

---

## 2. GraphSAGE — Inductive Representation Learning

**Paper:** Hamilton, Ying & Leskovec (2017) — "Inductive Representation Learning on Large Graphs"

GraphSAGE solves the transductive and scalability problems through **neighborhood sampling and aggregation**.

### 2.1 The Core Idea

Instead of operating on the full graph, GraphSAGE learns an **aggregation function** that can generate embeddings for unseen nodes.

```
                    ┌──────────────────────────┐
                    │   GraphSAGE Philosophy   │
                    │                          │
                    │   Don't memorize the     │
                    │   graph structure.        │
                    │                           │
                    │   Learn HOW to aggregate  │
                    │   neighbors, so you can   │
                    │   handle ANY graph.       │
                    └──────────────────────────┘
```

### 2.2 The Three Steps: Sample → Aggregate → Combine

For each target node, at each layer:

**Step 1 — Sample:** Select a fixed number of neighbors.

```
Target: User A (has 500 neighbors)

Sample K=3 neighbors:
  User A → {Item 12, Item 87, Item 203}

Next layer, for each sampled neighbor, sample again:
  Item 12  → {User A, User F}
  Item 87  → {User A, User C, User D}
  Item 203 → {User B}
```

**Step 2 — Aggregate:** Combine neighbor representations.

$$
h_{\mathcal{N}(v)}^{(l)} = \text{AGGREGATE}^{(l)}\!\left(\left\{h_u^{(l-1)} : u \in \mathcal{N}(v)\right\}\right)
$$

**Step 3 — Combine:** Merge the aggregated neighborhood with the node's own representation.

$$
h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot \text{CONCAT}\!\left(h_v^{(l-1)},\; h_{\mathcal{N}(v)}^{(l)}\right)\right)
$$

Visually for a 2-layer GraphSAGE:

```
Layer 2:        [User A embedding]
                    ↑
              COMBINE + σ
               ↑          ↑
          [self]     AGG(neighbors)
                      ↑
Layer 1:     [Item 12]  [Item 87]  [Item 203]
               ↑          ↑          ↑
          AGG(nbrs)   AGG(nbrs)  AGG(nbrs)
             ↑           ↑          ↑
Layer 0: {U_A,U_F}  {U_A,U_C,U_D} {U_B}
```

### 2.3 Aggregator Functions

GraphSAGE proposes three aggregators:

#### Mean Aggregator

$$
h_{\mathcal{N}(v)}^{(l)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(l-1)}
$$

Simple element-wise mean of neighbor features.

#### LSTM Aggregator

Feed neighbor embeddings through an LSTM. Since neighborhoods have no inherent order, the neighbors are randomly permuted before processing.

#### Pool Aggregator

$$
h_{\mathcal{N}(v)}^{(l)} = \max\!\left(\left\{\sigma\!\left(W_{\text{pool}}\,h_u^{(l-1)} + b\right) : u \in \mathcal{N}(v)\right\}\right)
$$

Apply a shared neural network to each neighbor, then take the element-wise max.

```
Aggregator Comparison:
┌─────────┬──────────┬────────────┬───────────────┐
│         │   Mean   │    LSTM    │     Pool      │
├─────────┼──────────┼────────────┼───────────────┤
│ Speed   │ Fastest  │ Slowest    │ Medium        │
│ Quality │ Good     │ Good       │ Often best    │
│ Params  │ None     │ Many       │ Medium        │
│ Order   │ Invariant│ Sensitive  │ Invariant     │
└─────────┴──────────┴────────────┴───────────────┘
```

### 2.4 Why GraphSAGE Matters for Recommendation

**Inductive capability:** GraphSAGE can embed a brand-new user the moment they interact with one item, without retraining.

```
Training:
  User A ── Item 1, Item 2
  User B ── Item 1, Item 3

After deployment, new user arrives:
  User C ── Item 1

GraphSAGE can compute User C's embedding by:
  1. Look up Item 1's embedding
  2. Aggregate Item 1's features
  3. Combine with User C's initial features
  → User C now has an embedding!

GCN would need to retrain on the expanded graph.
```

### 2.5 GraphSAGE in Code (Sketch)

```python
import torch
import torch.nn as nn

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Concat of self + aggregated neighbor → out_dim
        self.W = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index, sample_size=10):
        # For each node, sample neighbors
        neighbors = sample_neighbors(edge_index, sample_size)

        # Aggregate: mean of neighbor features
        neigh_agg = aggregate_mean(x, neighbors)

        # Combine: concat self with aggregated neighbors
        combined = torch.cat([x, neigh_agg], dim=-1)
        out = torch.relu(self.W(combined))

        # Normalize
        out = F.normalize(out, p=2, dim=-1)
        return out
```

---

## 3. GAT — Graph Attention Networks

**Paper:** Velickovic et al. (2018) — "Graph Attention Networks"

GAT solves the **equal neighbor weighting** problem by learning attention weights.

### 3.1 The Core Idea

Not all neighbors are equally important. GAT learns to assign different **attention scores** to different neighbors.

```
User Alice:
  ├── Item "Inception"      attention = 0.45  ← high
  ├── Item "Titanic"        attention = 0.10  ← low
  └── Item "Interstellar"   attention = 0.45  ← high

Alice's embedding is a WEIGHTED sum of neighbor features,
where the weights are learned, not fixed.
```

### 3.2 Computing Attention Coefficients

For a target node $i$ and neighbor $j$:

**Step 1:** Transform features with a shared weight matrix.

$$
z_i = W \cdot h_i, \quad z_j = W \cdot h_j
$$

**Step 2:** Compute raw attention score using a learned attention vector $\vec{a}$:

$$
e_{ij} = \text{LeakyReLU}\!\left(\vec{a}^T \cdot [z_i \| z_j]\right)
$$

where $\|$ denotes concatenation.

**Step 3:** Normalize with softmax over all neighbors of $i$:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

**Step 4:** Compute the new representation:

$$
h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\,z_j\right)
$$

Visually:

```
                    h_i' (updated embedding for node i)
                     ↑
                  σ(Σ αᵢⱼ · zⱼ)
                 ↗    ↑     ↖
         α₁·z₁    α₂·z₂    α₃·z₃
          ↑          ↑         ↑
    neighbor 1  neighbor 2  neighbor 3

Where α₁ + α₂ + α₃ = 1 (softmax normalized)
```

### 3.3 Multi-Head Attention

A single attention head might miss different aspects of neighbor importance. GAT uses multiple attention heads, each learning different attention patterns.

$$
h_i' = \Big\|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\,W^{(k)} h_j\right)
$$

where $\|$ denotes concatenation of $K$ heads.

```
Head 1: focuses on genre similarity
  "Inception" → high attention (same genre as target)

Head 2: focuses on rating pattern
  "Interstellar" → high attention (similar ratings)

Head 3: focuses on recency
  "Titanic" → low attention (watched long ago)

Final: CONCAT(head1, head2, head3)
```

For the **last layer**, concatenation is replaced by averaging:

$$
h_i' = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\,W^{(k)} h_j\right)
$$

### 3.4 When Attention Helps Recommendations

Attention is useful when **neighbor importance varies**:

```
Scenario 1: User with diverse taste
  ├── Haircare         (occasional)
  ├── Skincare         (primary interest)
  └── Kitchen gadgets  (gift purchases)

  → Attention should weight sci-fi neighbors higher

Scenario 2: Item with diverse audience
  ├── Watched by action fans
  ├── Watched by drama fans     (because of the actor)
  └── Watched by sci-fi fans    (primary audience)

  → Attention should weight sci-fi watchers higher
     when predicting for a sci-fi user
```

### 3.5 GAT in Code (Sketch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector for each head
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x, edge_index):
        # Transform features
        z = self.W(x)  # [N, out_dim]
        z = z.view(-1, self.num_heads, self.head_dim)  # [N, heads, head_dim]

        src, dst = edge_index  # source and target node indices

        # Compute attention for each edge
        z_src = z[src]  # [E, heads, head_dim]
        z_dst = z[dst]  # [E, heads, head_dim]

        # Concatenate source and target features
        edge_feat = torch.cat([z_src, z_dst], dim=-1)  # [E, heads, 2*head_dim]

        # Attention score
        e = (edge_feat * self.a).sum(dim=-1)  # [E, heads]
        e = F.leaky_relu(e, negative_slope=0.2)

        # Softmax over neighbors (using scatter)
        alpha = softmax_per_node(e, dst, num_nodes=x.size(0))

        # Weighted aggregation
        out = scatter_sum(alpha.unsqueeze(-1) * z_src, dst, dim=0)
        # [N, heads, head_dim]

        out = out.view(-1, self.num_heads * self.head_dim)  # [N, out_dim]
        return out
```

---

## 4. Architecture Comparison

| Property | GCN | GraphSAGE | GAT |
|---|---|---|---|
| **Neighbor weighting** | Degree-normalized (fixed) | Equal (mean) or learned (pool) | Learned attention weights |
| **Scalability** | Full graph needed | Mini-batch via sampling | Full graph (or sampling) |
| **Inductive?** | No (transductive) | Yes | Partially (needs feature init) |
| **New nodes** | Requires retraining | Can embed immediately | Needs initial features |
| **Complexity** | Low | Medium | Higher (attention computation) |
| **Parameters** | $W$ per layer | $W$ per layer + aggregator | $W$ + $\vec{a}$ per layer per head |
| **Best for** | Fixed, small-medium graphs | Large, evolving graphs | Heterogeneous importance |

### For Your BTP Specifically

```
If your graph is:
  ├── Static (fixed users/items)     → GCN or LightGCN is fine
  ├── Growing (new users/items)      → GraphSAGE is better
  └── Has varied importance          → GAT might help

Most BTP recommendation datasets are static
  → LightGCN (a simplified GCN variant) is usually the best choice
  → But you should KNOW GraphSAGE and GAT to justify your choice
```

---

## 5. Preparing for Your BTP Defense

When your thesis committee asks "Why did you choose LightGCN?", here is how you can frame your answer:

```
"I evaluated the three main GNN families:

1. GCN-based (including LightGCN):
   Best for static collaborative filtering because it
   operates directly on the full user-item graph structure.

2. GraphSAGE:
   Better for inductive settings with new nodes, but our
   dataset is fixed — we don't need inductive capability.

3. GAT:
   Could learn neighbor importance, but LightGCN's authors
   showed that for pure collaborative filtering, attention
   adds complexity without improving performance. The
   interaction signal itself is sufficient.

LightGCN was chosen because it is designed specifically for
the collaborative filtering setting and has been shown to
outperform both NGCF and standard GCN on recommendation
benchmarks."
```

---

## 6. Quick Reference

### When to Use What

```
GraphSAGE when:
  ✓ Graph changes (new users/items added frequently)
  ✓ Very large graph (millions of nodes)
  ✓ Need mini-batch training
  ✓ Rich node features available

GAT when:
  ✓ Neighbor importance varies significantly
  ✓ Heterogeneous graph (different edge types)
  ✓ Want interpretable attention weights
  ✓ Medium-sized graph (attention is expensive)

GCN / LightGCN when:
  ✓ Static graph
  ✓ Pure collaborative filtering (no rich features)
  ✓ Want simplicity and strong baseline
  ✓ Your dataset is a standard benchmark
```

---

## Exercises

1. **Conceptual:** Given a user who has purchased 100 products, explain how GraphSAGE's sampling strategy would handle this versus GCN's full aggregation.

2. **Attention intuition:** For a user who mostly buys skincare but also bought one haircare product, sketch what you'd expect the GAT attention weights to look like for recommending the next product.

3. **Trade-off analysis:** Your BTP uses the Amazon-Book dataset (52,643 users, 91,599 items). Which architecture would you choose and why? Consider the graph size, whether the graph is static, and whether you have rich node features.

4. **Code exercise:** Using PyTorch Geometric's `SAGEConv` and `GATConv`, build a 2-layer model for each and train them on Cora (a simple graph dataset). Compare their accuracy.

---

## Key Takeaways

1. **GCN is transductive** — it can't handle new nodes without retraining, which matters for real recommendation systems but usually not for a BTP with fixed datasets.

2. **GraphSAGE enables inductive learning** through its Sample-Aggregate-Combine framework. It's the go-to for production recommenders at companies like Pinterest (PinSage).

3. **GAT learns neighbor importance** through attention, which is intuitive but adds computational cost. For pure collaborative filtering, the added complexity may not pay off.

4. **For your BTP**, you'll likely use LightGCN (next phases), but understanding GraphSAGE and GAT lets you justify your choice and discuss alternatives in your thesis.

5. **The key defense question** is "Why this architecture?" — now you can answer it.
