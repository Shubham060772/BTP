# Hybrid GNN + Attention Models for Recommendation

> **Goal:** Study how GNN and attention are combined in state-of-the-art recommender systems and understand the architectural patterns.

---

## 1. Why Combine GNN + Attention?

Each approach has complementary strengths:

```
GNN (LightGCN):                        Attention (Transformer):
  + Captures collaborative signals        + Learns importance weighting
  + Respects graph topology               + Captures global patterns
  + Efficient on sparse graphs            + Handles heterogeneous signals
  - Fixed aggregation weights             - Ignores graph structure
  - Local receptive field                 - O(n²) scaling
  - Over-smoothing with depth             - Needs structural encoding
```

**The hybrid idea:** Use GNN to encode graph structure, then use attention to refine, weight, or augment the representations.

```
The spectrum of GNN-Attention integration:

Pure GNN ───── GAT ───── GNN+Attention ───── Graph Transformer ───── Pure Transformer
(LightGCN)   (learned    (two-stage or      (attention with         (ignores
              neighbor     parallel)          structure bias)         graph)
              weights)
```

---

## 2. Architecture Patterns

### Pattern 1: GNN Encoder + Attention Aggregator

```
Stage 1: GNN encodes structural embeddings
  User/Item ──> LightGCN (3 layers) ──> structural embeddings

Stage 2: Attention aggregates across layers or views
  layer_0_embed ──┐
  layer_1_embed ──┤── Attention ──> final embedding
  layer_2_embed ──┤
  layer_3_embed ──┘

Instead of simple averaging (as in LightGCN), learn which layer matters more.
```

**Why this works:** Different GNN layers capture different proximity levels. Layer 1 = direct neighbors, Layer 2 = 2-hop patterns, Layer 3 = community-level signals. Attention learns the optimal mix per user/item.

```python
class AttentiveLayerAggregation(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, layer_embeddings):
        # layer_embeddings: [num_layers, num_nodes, embed_dim]
        scores = self.attention(layer_embeddings)  # [num_layers, num_nodes, 1]
        weights = F.softmax(scores, dim=0)         # normalize across layers
        output = (weights * layer_embeddings).sum(dim=0)
        return output
```

### Pattern 2: Attention-Enhanced Message Passing

Replace the fixed aggregation in each GNN layer with attention:

```
Standard LightGCN layer:
  h_i^{l+1} = Σ (1/√(d_i·d_j)) · h_j^l     (fixed normalization)
                j∈N(i)

Attention-enhanced layer:
  h_i^{l+1} = Σ α_ij · h_j^l                 (learned attention)
                j∈N(i)

  where α_ij = softmax(score(h_i^l, h_j^l))
```

This is essentially GAT applied in a LightGCN-like framework (no feature transformation, just attention weighting).

### Pattern 3: Sequential GNN → Transformer

```
Step 1: LightGCN
  Collaborative filtering on user-item bipartite graph
  Output: user_cf_embed, item_cf_embed

Step 2: Transformer
  Feed CF embeddings + optional side features into Transformer
  Global attention captures patterns GNN missed

  Input: [user_cf_embed || user_features]
  Output: refined_user_embed

Step 3: Prediction
  score(u, i) = refined_user_embed^T · refined_item_embed
```

### Pattern 4: Parallel GNN + Transformer with Fusion

```
                    Input embeddings
                    /              \
                   v                v
            LightGCN branch    Transformer branch
            (graph structure)  (global attention)
                   \                /
                    v              v
                   Fusion layer
                   (concat + MLP, or gating)
                        |
                        v
                  Final embeddings
```

```python
class ParallelGNNTransformer(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # GNN branch
        self.lightgcn_layers = nn.ModuleList([
            LightGCNConv() for _ in range(3)
        ])

        # Transformer branch (on subgraphs)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )

        # Fusion
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, edge_index, subgraph_nodes):
        x0 = torch.cat([self.user_embed.weight, self.item_embed.weight])

        # GNN branch
        x_gnn = x0
        gnn_layers = [x0]
        for layer in self.lightgcn_layers:
            x_gnn = layer(x_gnn, edge_index)
            gnn_layers.append(x_gnn)
        x_gnn = torch.stack(gnn_layers).mean(dim=0)

        # Transformer branch (on subgraph for efficiency)
        x_sub = x0[subgraph_nodes].unsqueeze(0)
        x_transformer = self.transformer(x_sub).squeeze(0)

        # Gated fusion
        x_both = torch.cat([x_gnn[subgraph_nodes], x_transformer], dim=-1)
        gate = self.gate(x_both)
        fused = gate * x_gnn[subgraph_nodes] + (1 - gate) * x_transformer

        return fused
```

---

## 3. Key Papers

### 3.1 SGL — Self-supervised Graph Learning (SIGIR 2021)

**Problem:** LightGCN is biased toward high-degree nodes and vulnerable to noisy interactions.

**Core idea:** Add contrastive self-supervised learning to LightGCN.

```
Main task: BPR recommendation loss (same as LightGCN)

Auxiliary task: Contrastive learning
  Create two augmented views of the graph:
    View 1: randomly drop edges
    View 2: randomly drop edges (different random seed)

  Pull same node's embeddings in two views together:
    maximize sim(z1_user, z2_user)
  Push different nodes apart:
    minimize sim(z1_user, z2_other_user)

Total loss = L_BPR + λ · L_contrastive
```

```
Graph augmentation strategies:

Original:                Node Dropout:           Edge Dropout:
User1──ProdA            User1──ProdA            User1──ProdA
User1──ProdB            [User2 dropped]         User1  ProdB  (edge removed)
User2──ProdA                                    User2──ProdA
User2──ProdC            User2──ProdC            User2──ProdC
```

**Results on Amazon-Book:** ~5-8% improvement over LightGCN on Recall@20.

### 3.2 DGCF — Disentangled Graph CF (SIGIR 2020)

**Problem:** Standard GNNs produce entangled user/item representations — a single embedding vector captures all intents.

**Core idea:** Decompose user-item interactions into multiple intent-aware components.

```
Standard embedding:
  User 1 = [0.3, -0.1, 0.5, ...]   (one vector for all intents)

Disentangled embedding:
  User 1, intent 1 (skincare)  = [0.8, 0.2, ...]
  User 1, intent 2 (gifting)   = [-0.1, 0.7, ...]
  User 1, intent 3 (budget)    = [0.3, 0.5, ...]

Each purchase is explained by a specific intent:
  User 1 → CeraVe Moisturizer:  intent 1 (skincare), attention α=0.9
  User 1 → Gift Set:            intent 2 (gifting),  attention α=0.8
```

**Architecture:**

$$
\mathbf{e}_u = \sum_{k=1}^{K} \mathbf{e}_u^{(k)} \quad \text{where } K \text{ = number of intents}
$$

Each intent has its own attention over neighbors:

$$
\alpha_{ij}^{(k)} = \frac{\exp(\mathbf{e}_i^{(k)T} \mathbf{e}_j^{(k)})}{\sum_{j'} \exp(\mathbf{e}_i^{(k)T} \mathbf{e}_{j'}^{(k)})}
$$

### 3.3 GFormer — Graph Transformer for Recommendation (SIGIR 2023)

**Problem:** Graph augmentation in contrastive learning (like SGL) is hand-crafted and may distort important structures.

**Core idea:** Use a graph transformer to generate augmented views automatically.

```
GFormer pipeline:

Step 1: Rationale Discovery
  Learn which edges/nodes are important (collaborative rationales)
  via parameterized masking

Step 2: Graph Transformer
  Apply transformer with structure-aware attention
  Uses rationale-guided augmentation

Step 3: Contrastive Learning
  Pull augmented views together (like SGL)
  But views are generated by the model, not random dropout

Main innovation: Self-supervised augmentation is learned, not hand-crafted.
```

### 3.4 HCCF — Hypergraph Contrastive CF (2022)

**Problem:** Pairwise edges (user-item) miss higher-order group relationships.

**Core idea:** Use hypergraph (one edge can connect multiple nodes) + contrastive learning.

```
Standard graph:              Hypergraph:
User1 ── ProdA               Hyperedge 1: {User1, User2, User5}
User1 ── ProdB                 (users who all bought skincare)
User2 ── ProdA
User2 ── ProdC               Hyperedge 2: {ProdA, ProdB, ProdD}
                                (products often bought together)
```

**Architecture:**
- Local: LightGCN on bipartite graph
- Global: Hypergraph convolution on learned hyperedges
- Contrastive: Align local and global views

### 3.5 Comparison Table

| Model | Year | Base GNN | Attention type | Extra signal | Key improvement |
|-------|------|----------|---------------|-------------|----------------|
| **SGL** | 2021 | LightGCN | None (contrastive) | Self-supervised | Robustness, long-tail |
| **DGCF** | 2020 | GCN | Intent-aware attention | Disentanglement | Multi-intent modeling |
| **GFormer** | 2023 | Graph Transformer | Global self-attention | Learned augmentation | Auto SSL augmentation |
| **HCCF** | 2022 | LightGCN + Hypergraph | Hypergraph attention | Contrastive | Higher-order patterns |
| **KGAT** | 2019 | GAT | Relation-aware attention | Knowledge graph | Side information |

---

## 4. Common Theme: Contrastive Learning

Most recent improvements over LightGCN add some form of self-supervised / contrastive learning:

```
Supervised only (LightGCN):
  Loss = L_BPR

Contrastive augmented:
  Loss = L_BPR + λ · L_contrastive

Why this helps:
  1. Regularization — prevents overfitting on sparse data
  2. Better representations — embeddings capture more than just collaborative signals
  3. Robustness — less sensitive to noisy interactions
  4. Long-tail — better performance on users/items with few interactions
```

### Contrastive Learning Framework

```python
def contrastive_loss(z1, z2, temperature=0.2):
    """InfoNCE loss between two views of the same nodes."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Positive pairs: same node in two views
    pos_score = (z1 * z2).sum(dim=1) / temperature

    # Negative pairs: different nodes
    neg_score = torch.mm(z1, z2.t()) / temperature

    # InfoNCE
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(neg_score, labels)
    return loss
```

---

## 5. Implementation: LightGCN + Contrastive Learning (SGL-style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN_SGL(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3,
                 drop_rate=0.1, temperature=0.2, ssl_weight=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.temperature = temperature
        self.ssl_weight = ssl_weight

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def graph_conv(self, x, edge_index, edge_weight):
        """One LightGCN propagation layer."""
        from torch_geometric.nn import MessagePassing
        # Simplified: use sparse matrix multiplication
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_weight is not None:
            norm = norm * edge_weight

        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(1).expand_as(x[row]),
                         norm.unsqueeze(1) * x[row])
        return out

    def encode(self, edge_index, edge_weight=None):
        x = torch.cat([self.user_embed.weight, self.item_embed.weight])
        all_layers = [x]
        for _ in range(self.num_layers):
            x = self.graph_conv(x, edge_index, edge_weight)
            all_layers.append(x)
        final = torch.stack(all_layers).mean(dim=0)
        return final[:self.num_users], final[self.num_users:]

    def edge_dropout(self, edge_index, drop_rate):
        """Randomly drop edges for augmentation."""
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
        return edge_index[:, mask]

    def forward(self, edge_index, users, pos_items, neg_items):
        # Main view
        user_emb, item_emb = self.encode(edge_index)

        # Augmented views for contrastive learning
        edge_index_1 = self.edge_dropout(edge_index, self.drop_rate)
        edge_index_2 = self.edge_dropout(edge_index, self.drop_rate)
        user_emb_1, item_emb_1 = self.encode(edge_index_1)
        user_emb_2, item_emb_2 = self.encode(edge_index_2)

        # BPR loss
        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]
        bpr_loss = -F.logsigmoid((u * pos).sum(1) - (u * neg).sum(1)).mean()

        # Contrastive loss (users)
        ssl_loss_user = contrastive_loss(
            user_emb_1[users], user_emb_2[users], self.temperature)

        # Contrastive loss (items)
        all_items = torch.cat([pos_items, neg_items]).unique()
        ssl_loss_item = contrastive_loss(
            item_emb_1[all_items], item_emb_2[all_items], self.temperature)

        total_loss = bpr_loss + self.ssl_weight * (ssl_loss_user + ssl_loss_item)
        return total_loss
```

---

## 6. Design Decision Framework

When designing your BTP model, choose your architecture pattern based on:

```
Question 1: Do you have side information (categories, brands)?
  Yes → Consider KGAT-style knowledge graph attention
  No  → Work with bipartite graph only

Question 2: Is your main bottleneck accuracy or robustness?
  Accuracy → Attention-enhanced aggregation (Pattern 2)
  Robustness → Contrastive learning (SGL/GFormer style)

Question 3: What's your compute budget?
  Low  → LightGCN + simple contrastive (SGL)
  Med  → GAT + contrastive OR disentangled (DGCF)
  High → Graph Transformer hybrid (GFormer)

Question 4: What's your novelty angle?
  Combine two ideas that haven't been combined:
    e.g., Disentangled intents + Contrastive learning
    e.g., Knowledge graph + Graph Transformer
    e.g., Attention layer aggregation + Edge dropout augmentation
```

---

## 7. Exercises

1. **Implement SGL.** Add edge dropout contrastive learning to your LightGCN baseline. Sweep λ ∈ {0.01, 0.05, 0.1, 0.5}. Report Recall@20 and NDCG@20 on Amazon-Book.

2. **Disentangled embeddings.** Split your 64-dim embeddings into 4 intent channels of 16-dim each. For each user-item interaction, compute attention over intents. Does this improve over standard LightGCN?

3. **Layer attention.** Replace LightGCN's uniform layer averaging with learned attention weights. Report per-layer weights after training — which layer gets the most attention?

4. **Augmentation comparison.** Implement three augmentation strategies: (a) edge dropout, (b) node dropout, (c) random walk. Which produces the best contrastive signal on Amazon-Book?

5. **Hybrid design.** Design (on paper first) a model that combines at least two ideas from this document. Sketch the architecture, write pseudocode, then estimate parameter count for Amazon-Book scale.
