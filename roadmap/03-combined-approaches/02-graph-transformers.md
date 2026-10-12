# Graph Transformers

> **Goal:** Understand how Transformer architecture is adapted for graph-structured data and what this means for recommendation.

---

## 1. Why Transformers on Graphs?

Standard GNNs (GCN, GAT, LightGCN) propagate information **locally** — each node only talks to its direct neighbors per layer. To reach a node 3 hops away, you need 3 layers.

Transformers use **global self-attention** — every node can attend to every other node in a single layer.

```
GNN (local message passing):           Transformer (global attention):

Layer 1: 1-hop neighbors               Layer 1: ALL nodes
Layer 2: 2-hop neighbors               (every node attends to every node)
Layer 3: 3-hop neighbors

User 1 → Products → Other Users        User 1 → ALL Users + ALL Products
(information flows through edges)       (information flows through attention)
```

**The promise:** Capture long-range dependencies in one layer instead of stacking many GNN layers (which causes over-smoothing).

**The cost:** $O(n^2)$ attention vs $O(|E|)$ message passing.

---

## 2. The Core Challenge: Encoding Graph Structure

A standard Transformer treats input as a set (or sequence with positional encoding). It has no notion of graph topology. If you just feed node features to a Transformer, it ignores which nodes are connected.

**Solution:** Encode structural information into the Transformer through:

### 2.1 Laplacian Positional Encodings

Use the eigenvectors of the graph Laplacian as positional encodings:

$$
L = D - A \quad \text{(unnormalized Laplacian)}
$$

$$
L = U \Lambda U^T \quad \text{(eigendecomposition)}
$$

Take the first $k$ eigenvectors as positional encoding for each node:

$$
\text{PE}(v_i) = [u_1(i), u_2(i), \ldots, u_k(i)]
$$

```
Graph Laplacian encodes structure:

Eigenvector 1 (Fiedler): rough partition of graph
  → Users on one side, popular products on another

Eigenvector 2: finer partition
  → Clusters of similar users

Eigenvectors capture global graph structure that local GNN layers miss.
```

### 2.2 Random Walk Positional Encodings

Compute the probability that a random walk starting at node $i$ returns to $i$ in $k$ steps:

$$
\text{RWPE}(v_i) = [\text{RW}_{ii}^1, \text{RW}_{ii}^2, \ldots, \text{RW}_{ii}^k]
$$

where $\text{RW}^k = (D^{-1}A)^k$.

```
Random walk encoding for User 1:

Step 1: P(return in 1 step) = 0       (bipartite: user→product, can't return)
Step 2: P(return in 2 steps) = 0.15   (user→product→same user via shared product)
Step 3: P(return in 3 steps) = 0       (odd steps can't return in bipartite)
Step 4: P(return in 4 steps) = 0.08

This encodes local graph density around User 1.
```

### 2.3 Edge Features as Attention Bias

Add a bias to the attention scores based on graph connectivity:

$$
\text{Attention}(i, j) = \frac{(\mathbf{W}_Q \mathbf{h}_i)^T (\mathbf{W}_K \mathbf{h}_j)}{\sqrt{d}} + b_{ij}
$$

where $b_{ij}$ encodes structural information (e.g., shortest path distance, edge type).

---

## 3. Key Graph Transformer Architectures

### 3.1 Graphormer (Ying et al., 2021 — Microsoft)

```
Input: node features + graph structure

Encoding:
  1. Centrality encoding: degree(v) → bias added to node features
  2. Spatial encoding: shortest_path(i,j) → bias in attention
  3. Edge encoding: edge features along shortest path → attention bias

Attention(i,j) = (Q_i · K_j) / √d + b_spatial(i,j) + b_edge(i,j)

Architecture:
  Standard Transformer blocks
  + structural encodings baked into attention
```

**Key insight:** Graph structure enters through attention biases, not through message passing.

### 3.2 SAN — Spectral Attention Network (Kreuzer et al., 2021)

```
Split attention into two streams:

Stream 1: Learned PE attention
  - Use Laplacian eigenvectors as positional encodings
  - Attend based on spectral similarity

Stream 2: Full graph attention
  - Standard self-attention over all nodes

Combine: weighted sum of both streams
```

### 3.3 NodeFormer (Wu et al., 2022)

```
Problem: O(n²) attention is too expensive for large graphs

Solution: Kernelized message passing
  - Approximate softmax attention with random features
  - Reduces to O(n) complexity
  - Adds "gumbel-softmax" for neighbor selection

Complexity: O(n · d · k) where k << n
```

### 3.4 Comparison Table

| Architecture | Structure encoding | Complexity | Key innovation |
|-------------|-------------------|------------|---------------|
| Graphormer | Centrality + spatial + edge bias | $O(n^2)$ | Structural bias in attention |
| SAN | Laplacian eigenvectors | $O(n^2)$ | Spectral positional encoding |
| NodeFormer | Kernelized attention | $O(n)$ | Efficient global attention |
| GPS (Rampášek et al.) | Combines local MPNN + global transformer | $O(n^2 + |E|)$ | Best of both worlds |

---

## 4. Graph Transformer vs Standard GNN

```
                        GNN                     Graph Transformer
                         |                           |
Receptive field:       Local (L-hop)             Global (all nodes)
                       after L layers            in 1 layer

Over-smoothing:        Yes (deep GNNs)           Less prone
                       embeddings converge        attention is selective

Complexity per layer:  O(|E| · d)               O(n² · d) or O(n · d)
                       scales with edges          scales with nodes

Inductive bias:        Strong (locality,          Weaker (needs structure
                       homophily)                 encoding)

When it works best:    Homophilous graphs         Heterophilous graphs
                       (similar nodes connect)     Long-range dependencies

Parameters:            Fewer                      More (Q, K, V per head)
```

### For recommendation specifically:

```
User-item graph properties:
  - Bipartite (strong structural pattern)
  - Sparse (~99.9% for Amazon-Book)
  - Nodes: 144K (52K users + 91K items)
  - Edges: ~6M (undirected)

GNN (LightGCN):
  O(|E| · d) = O(6M × 64) ≈ 384M ops per layer
  3 layers ≈ 1.2B ops

Graph Transformer (full attention):
  O(n² · d) = O(144K² × 64) ≈ 1.3T ops per layer
  Infeasible without approximation!

→ Pure graph transformers don't scale to recommendation graphs.
   You need efficient variants or hybrid approaches.
```

---

## 5. Practical Approaches for Recommendation

### 5.1 Subgraph Transformers

Instead of global attention over all nodes, apply transformer to local subgraphs:

```
For User 1:
  1. Sample a subgraph: User 1 + their products + 2-hop users
  2. Apply transformer within this subgraph
  3. Much smaller: ~100 nodes instead of 144K

  Subgraph:
  User 1 ── Product A ── User 2
     |          |
  Product B ── User 5
     |
  Product C
```

### 5.2 GNN + Transformer Hybrid (GPS-style)

```
Each layer combines:

  Local:  GNN message passing (captures graph structure)
  Global: Transformer attention (captures long-range patterns)

  h_i^{local}  = GNN_layer(h_i, neighbors)
  h_i^{global} = Transformer_layer(h_i, all_nodes)
  h_i^{new}    = MLP(h_i^{local} + h_i^{global})
```

### 5.3 Sequential Approach

```
Stage 1: GNN encodes graph structure
  LightGCN → user/item embeddings (captures collaborative signals)

Stage 2: Transformer refines embeddings
  Feed GNN embeddings into Transformer
  → captures patterns GNN missed (non-local similarities)
```

---

## 6. Code — Minimal Graph Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_bias=None):
        # x: [num_nodes, embed_dim]
        # attn_bias: [num_nodes, num_nodes] structural encoding
        residual = x
        x = self.norm1(x)
        x = x.unsqueeze(0)  # [1, num_nodes, embed_dim] for batch dim

        if attn_bias is not None:
            attn_mask = attn_bias.unsqueeze(0)
            x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        else:
            x, _ = self.attention(x, x, x)

        x = x.squeeze(0)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class SubgraphTransformerRecommender(nn.Module):
    """Apply transformer to sampled subgraphs for scalability."""

    def __init__(self, num_users, num_items, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Learnable node type embedding
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=user, 1=item

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def encode_subgraph(self, user_ids, item_ids):
        u_emb = self.user_embed(user_ids) + self.type_embed(
            torch.zeros_like(user_ids))
        i_emb = self.item_embed(item_ids) + self.type_embed(
            torch.ones_like(item_ids))

        x = torch.cat([u_emb, i_emb], dim=0)

        for layer in self.layers:
            x = layer(x)

        user_out = x[:len(user_ids)]
        item_out = x[len(user_ids):]
        return user_out, item_out
```

---

## 7. Structural Encodings — Implementation

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def laplacian_positional_encoding(adj_matrix, k=8):
    """Compute Laplacian PE for graph nodes."""
    n = adj_matrix.shape[0]
    deg = np.array(adj_matrix.sum(axis=1)).flatten()
    D = sp.diags(deg)
    L = D - adj_matrix  # unnormalized Laplacian

    # Smallest k+1 eigenvalues (skip first which is 0)
    eigenvalues, eigenvectors = eigsh(L.astype(float), k=k+1, which='SM')
    pe = eigenvectors[:, 1:k+1]  # skip trivial eigenvector
    return torch.tensor(pe, dtype=torch.float32)


def random_walk_positional_encoding(adj_matrix, walk_length=16):
    """Compute Random Walk PE for graph nodes."""
    n = adj_matrix.shape[0]
    deg = np.array(adj_matrix.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    D_inv = sp.diags(1.0 / deg)
    RW = D_inv @ adj_matrix  # random walk matrix

    pe = np.zeros((n, walk_length))
    RW_power = sp.eye(n)
    for step in range(walk_length):
        RW_power = RW_power @ RW
        pe[:, step] = RW_power.diagonal()

    return torch.tensor(pe, dtype=torch.float32)
```

---

## 8. When to Use Graph Transformers for Recommendation

```
Use Graph Transformers when:
  ✓ You need to capture long-range user-item patterns
  ✓ Your graph has heterophilous regions (dissimilar nodes connected)
  ✓ You have rich structural features to encode
  ✓ Subgraph sampling is feasible

Stick with GNN (LightGCN) when:
  ✓ Graph is large (>100K nodes) and you can't subsample well
  ✓ Local neighborhood signals are sufficient
  ✓ Training budget is limited
  ✓ You need a fast, reproducible baseline

Hybrid (GNN + Transformer) when:
  ✓ You want local structure (GNN) + global patterns (Transformer)
  ✓ You have compute budget for both
  ✓ This is often the sweet spot for a BTP contribution
```

---

## 9. Exercises

1. **Compute Laplacian PE** for a small user-item graph (10 users, 20 items). Visualize the first 3 eigenvectors. Do users with similar purchase patterns cluster together in eigenvector space?

2. **Scalability experiment.** Measure the time to compute full self-attention for subgraphs of size 50, 100, 500, 1000 nodes. At what size does it become impractical on your GPU?

3. **Compare attention patterns.** In a subgraph transformer, visualize the attention matrix. Do users mostly attend to products? Do products attend to other products in the same category?

4. **Hybrid architecture.** Implement a simple hybrid: (a) Run 2-layer LightGCN to get base embeddings, (b) Feed embeddings into a 1-layer Transformer for the top-100 candidate items per user. Compare vs pure LightGCN.

5. **Random Walk PE.** Compute RWPE for Amazon-Book. Plot the distribution of step-2 return probabilities. How does this correlate with node degree?
