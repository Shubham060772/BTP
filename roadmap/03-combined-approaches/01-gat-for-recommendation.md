# Graph Attention Networks for Recommendation

> **Goal:** Understand how attention replaces fixed aggregation in GNNs and why this matters for product recommendation.

---

## 1. The Problem with Fixed Aggregation

In GCN and LightGCN, every neighbor contributes to a node's representation equally (after normalization):

```
GCN aggregation (fixed weights):

User 1 purchased:
  Product A (daily essential)     ──┐
  Product B (gift for someone)    ──┤── equal weight ──> User 1 embedding
  Product C (impulse buy)         ──┘

All three products contribute equally to User 1's embedding.
But Product A (daily essential) says much more about User 1's preferences
than Product C (impulse buy).
```

**GAT's solution:** Learn attention weights that determine how much each neighbor matters.

```
GAT aggregation (learned weights):

User 1 purchased:
  Product A (daily essential)     ── α=0.6 ──┐
  Product B (gift for someone)    ── α=0.1 ──┤──> User 1 embedding
  Product C (impulse buy)         ── α=0.3 ──┘

Attention weights α are learned during training.
```

---

## 2. GAT Mechanism — Step by Step

### Step 1: Linear Transformation

Each node's features are projected into a new space:

$$
\mathbf{z}_i = \mathbf{W} \mathbf{h}_i
$$

where $\mathbf{W} \in \mathbb{R}^{d' \times d}$ is a learnable weight matrix and $\mathbf{h}_i$ is the input embedding.

### Step 2: Attention Coefficients

For each edge $(i, j)$, compute a raw attention score:

$$
e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{z}_i \| \mathbf{z}_j])
$$

where $\mathbf{a} \in \mathbb{R}^{2d'}$ is a learnable attention vector and $\|$ denotes concatenation.

```
Computing attention for User 1 → Product A:

z_user1  = W · h_user1     (project user embedding)
z_prodA  = W · h_prodA     (project product embedding)

e = LeakyReLU(a^T · [z_user1 || z_prodA])
                    ↑ concatenate both
```

### Step 3: Normalize with Softmax

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

This ensures attention weights across all neighbors of node $i$ sum to 1.

### Step 4: Weighted Aggregation

$$
\mathbf{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{z}_j\right)
$$

```
Full pipeline for one node:

h_i ──> W·h_i = z_i ──┐
                       ├── e_ij = LeakyReLU(a^T[z_i||z_j]) ──> softmax ──> α_ij
h_j ──> W·h_j = z_j ──┘                                                      |
                                                                               v
                                                           h'_i = σ(Σ α_ij · z_j)
```

---

## 3. Multi-Head Attention

A single attention head may focus on one aspect. Multiple heads capture diverse patterns:

$$
\mathbf{h}'_i = \Big\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)
$$

```
Head 1: focuses on product category similarity
Head 2: focuses on price range similarity
Head 3: focuses on brand affinity
Head 4: focuses on purchase recency

Each head has its own W^(k) and a^(k).
Final embedding = concatenation (or average) of all heads.
```

For the final layer, concatenation is replaced by averaging:

$$
\mathbf{h}'_i = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)
$$

---

## 4. GCN vs GAT vs GraphSAGE — Comparison

```
                GCN                    GAT                    GraphSAGE
                 |                      |                        |
Aggregation:   Fixed symmetric       Learned attention        Sampled + aggregated
               normalization          weights                  (mean/LSTM/pool)
               1/√(d_i · d_j)        α_ij (softmax)          1/|S(i)| for mean

Weights:       Same for all          Different per            Same aggregator
               neighbors             neighbor pair            for all nodes

Scalability:   O(|E| · d)           O(|E| · d + |E| · d')   O(k^L · d)
                                     (extra attention cost)   (sampling bounds cost)

Expressiveness: Low                  High                     Medium
               (can't distinguish    (learns which            (sampling adds
                neighbor importance)  neighbors matter)        stochasticity)

Parameters:    W only                W + a (per head)         W + aggregator params
```

### When to use what for recommendation:

| Scenario | Best choice | Why |
|----------|-------------|-----|
| Large-scale, need speed | LightGCN (no params in aggregation) | Fastest, competitive results |
| Neighbors have varying importance | GAT | Learned attention handles this |
| Very large graph, memory-constrained | GraphSAGE | Sampling controls memory |
| Rich side information (KG) | GAT / KGAT | Attention over heterogeneous edges |

---

## 5. GAT for User-Item Bipartite Graphs

Applying GAT to recommendation has a subtlety: in a bipartite graph, users only connect to items and vice versa.

```
Bipartite GAT:

Layer 1: Users attend over their purchased products
         Products attend over their purchasing users

         User 1 ──α=0.5──> Product A
         User 1 ──α=0.3──> Product B
         User 1 ──α=0.2──> Product C

Layer 2: Users attend over 2-hop neighbors (other users via shared products)
         Products attend over 2-hop neighbors (other products via shared users)

         User 1 ──(via Product A)──> User 2, User 5
```

The attention mechanism learns:
- Which purchased products best represent a user's preferences
- Which users best represent a product's audience

---

## 6. KGAT — Knowledge Graph Attention Network

KGAT extends GAT to knowledge graphs where items have rich attributes.

### The Knowledge Graph

```
Standard bipartite:              With knowledge graph:

User 1 ─── Product A            User 1 ─── Product A ─── Category: Skincare
User 2 ─── Product A                           |
User 2 ─── Product B            Product A ─── Brand: CeraVe
                                                  |
                                Product B ─── Brand: CeraVe
                                    |
                                User 2 ─── Product B ─── Category: Moisturizer
```

Products sharing a brand are now connected through the brand node, even if no user purchased both.

### KGAT Architecture

```
Step 1: Build CKG (Collaborative Knowledge Graph)
        = User-Item graph + Item-Attribute KG

Step 2: Attentive embedding propagation
        For each node, aggregate neighbors with attention:

        e(h,r,t) = (W_r · h)^T · (W_r · t)
                    ↑ relation-aware transformation

        α(h,r,t) = softmax(e(h,r,t))

Step 3: Stack L layers of propagation

Step 4: Predict: score(user, item) = user_embed^T · item_embed
```

### Key Innovation

KGAT uses **relation-aware attention**: different edge types (purchased, belongs_to, made_by) get different transformations.

$$
e_{(h,r,t)} = (\mathbf{W}_r \mathbf{e}_h)^T (\mathbf{W}_r \mathbf{e}_t)
$$

where $r$ is the relation type. This means "purchased" edges are scored differently from "belongs_to" edges.

---

## 7. Code — GAT for Recommendation in PyTorch Geometric

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATRecommender(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, heads=4, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else embed_dim * heads
            self.gat_layers.append(
                GATConv(in_dim, embed_dim, heads=heads, concat=(i < num_layers - 1))
            )

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)

        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)

        user_embeds = x[:self.num_users]
        item_embeds = x[self.num_users:]
        return user_embeds, item_embeds

    def bpr_loss(self, user_embeds, item_embeds, users, pos_items, neg_items):
        u = user_embeds[users]
        pos = item_embeds[pos_items]
        neg = item_embeds[neg_items]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss
```

### Comparison: LightGCN vs GAT forward pass

```python
# LightGCN: simple, no learnable aggregation
# Each layer: X^(l+1) = D^(-1/2) A D^(-1/2) X^(l)
# Final: average over all layers

# GAT: learned attention per edge
# Each layer: X^(l+1) = attention_aggregate(X^(l), edge_index)
# More parameters, potentially more expressive
```

---

## 8. Attention Weight Analysis

One advantage of GAT: you can inspect what the model learned.

```python
# After training, extract attention weights
with torch.no_grad():
    # GATConv stores attention weights if return_attention_weights=True
    out, (edge_index_att, alpha) = gat_layer(x, edge_index,
                                              return_attention_weights=True)

# alpha[i] = attention weight for edge_index[:, i]
# Find which products User 1 attends to most
user1_edges = (edge_index_att[0] == 0)
user1_neighbors = edge_index_att[1, user1_edges]
user1_attention = alpha[user1_edges]

# Sort by attention weight
sorted_idx = user1_attention.argsort(descending=True)
for idx in sorted_idx[:5]:
    print(f"Product {user1_neighbors[idx].item()}: attention = {user1_attention[idx].item():.4f}")
```

This gives you interpretability: "User 1's embedding is primarily influenced by Product A (α=0.42) and Product D (α=0.31)."

---

## 9. Practical Considerations

### GAT vs LightGCN for your BTP

```
LightGCN advantages:
  + Faster training (no attention computation)
  + Fewer parameters
  + Strong baseline — hard to beat

GAT advantages:
  + Learns neighbor importance
  + More expressive
  + Attention weights are interpretable
  + Natural fit for heterogeneous graphs (KGAT)

GAT disadvantages:
  - O(|E|) extra computation per layer for attention
  - More hyperparameters (number of heads, head dimension)
  - Can overfit on sparse data
  - Often marginal gains over LightGCN on standard benchmarks
```

### Memory comparison (Amazon-Book scale)

```
Amazon-Book: 52,643 users, 91,599 items, ~3M interactions

LightGCN (64-dim, 3 layers):
  Parameters: (52,643 + 91,599) × 64 = 9.2M params
  No aggregation params

GAT (64-dim, 4 heads, 2 layers):
  Embedding params: 9.2M (same)
  Attention params per layer: 64 × 64 × 4 (W) + 128 × 4 (a) = 16,896
  Total attention: ~34K extra params
  But attention computation on ~6M edges per layer adds memory
```

---

## 10. Exercises

1. **Implement GAT for recommendation.** Use the code template above with Amazon-Book. Compare Recall@20 and NDCG@20 against your LightGCN baseline. Is the improvement worth the extra training time?

2. **Visualize attention weights.** For 5 users, extract the attention weights over their purchased products. Do the high-attention products share any patterns (e.g., same category, similar price)?

3. **Multi-head analysis.** Train a 4-head GAT. For a single user, check if different heads focus on different aspects of their purchase history.

4. **Ablation study.** Compare: (a) 1-head GAT, (b) 4-head GAT, (c) 8-head GAT. Plot Recall@20 vs number of heads. Is more always better?

5. **Think about KGAT.** If you had Amazon-Beauty metadata (category, brand, price), sketch the knowledge graph. What node types and edge types would you define? How many extra edges would the KG add compared to the bipartite graph alone?
