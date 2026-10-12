# Phase 15 — Advanced Directions (Week 10–11)

> **Goal:** Choose ONE advanced direction to add depth and novelty to your BTP. Don't try all four.

---

## 1. Why Pick Only One

Your remaining BTP time is limited. Each direction below requires:
- Understanding new concepts
- Reading 2–3 papers
- Implementing new code
- Running experiments

Doing one well is better than doing four poorly. Pick the one that:
1. Aligns with your research question
2. Is feasible in 1–2 weeks
3. Interests you most

```
                    ┌──────────────┐
                    │ Your BTP so  │
                    │ far: LightGCN│
                    │ on bipartite │
                    │ user-item    │
                    │ graph        │
                    └──────┬───────┘
                           │
           Pick ONE:       │
     ┌─────────┬───────────┼───────────┬─────────┐
     v         v           v           v         v
  Option A  Option B   Option C   Option D   (or stay
  Hetero    Explain    Cold-Start Temporal   with
  GNN       ability                          baseline)
```

---

## 2. Option A — Heterogeneous GNN

### The Idea

Instead of treating your graph as only users and items, introduce additional node types like genres, actors, categories, or brands.

```
Homogeneous (what you have now):

User 1 ─── Product A
User 1 ─── Product B
User 2 ─── Product A
User 2 ─── Product C

Heterogeneous (what you'd build):

User 1 ─── Product A ─── Category: Skincare
User 1 ─── Product B ─── Category: Haircare
User 2 ─── Product A ─── Category: Skincare
User 2 ─── Product C ─── Category: Moisturizer
               |
            Brand: CeraVe
```

### Why It Helps

Products that share a category or brand become connected through those metadata nodes, even if no single user purchased both. This **densifies the graph** and provides additional collaborative signals.

$$
G = (\{U, I, C, B\}, \{E_{ui}, E_{ic}, E_{ib}\})
$$

where:
- $U$ = users, $I$ = items, $C$ = categories, $B$ = brands
- $E_{ui}$ = user-item edges, $E_{ic}$ = item-category edges, $E_{ib}$ = item-brand edges

### Implementation in PyG

```python
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv

data = HeteroData()

# Nodes
data['user'].x = torch.randn(num_users, 64)
data['product'].x = torch.randn(num_products, 64)
data['category'].x = torch.randn(num_categories, 64)

# Edges (each type is its own edge_index)
data['user', 'purchased', 'product'].edge_index = user_product_edges
data['product', 'belongs_to', 'category'].edge_index = product_category_edges

# Reverse edges (for undirected message passing)
data['product', 'purchased_by', 'user'].edge_index = user_product_edges.flip(0)
data['category', 'contains', 'product'].edge_index = product_category_edges.flip(0)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'purchased', 'product'): GCNConv(-1, hidden_dim),
            ('product', 'purchased_by', 'user'): GCNConv(-1, hidden_dim),
            ('product', 'belongs_to', 'category'): GCNConv(-1, hidden_dim),
            ('category', 'contains', 'product'): GCNConv(-1, hidden_dim),
        })

    def forward(self, x_dict, edge_index_dict):
        return self.conv1(x_dict, edge_index_dict)
```

### Difficulty: Medium-High

| Aspect | Details |
|--------|---------|
| **Concepts** | Heterogeneous graphs, typed message passing |
| **Papers** | HAN (2019), HGT (2020) |
| **Implementation** | PyG's `HeteroConv` handles most of the complexity |
| **Experiments** | Compare homogeneous LightGCN vs heterogeneous variant |
| **Time needed** | ~2 weeks |

### Pros and Cons for Recommendation

```
Pros:                                Cons:
+ Richer graph structure             - More complex to implement
+ Helps with sparse interactions     - More hyperparameters
+ Multiple paths between nodes       - Metadata may not always help
+ Natural fit for metadata           - Harder to train
```

---

## 3. Option B — Explainable Recommendation

### The Idea

Instead of just recommending items, explain **why**:

```
Standard recommendation:
  "We recommend Product A"

Explainable recommendation:
  "We recommend Product A because:
   - You purchased Product B, which is in the same Skincare category
   - Users similar to you also purchased it
   - It's from Brand X, which appears in 3 of your recent purchases"
```

### Approach 1: Attention Weights as Explanations

If you use GAT instead of GCN, the attention weights tell you which neighbors were most important:

```
User 1's representation is computed from:

Product A ──(0.45)──→ User 1
Product B ──(0.30)──→ User 1
Product C ──(0.25)──→ User 1

Interpretation:
"Product A was the most influential item in shaping
 User 1's preference profile."
```

$$
\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}
$$

The attention coefficient $\alpha_{ij}$ can serve as an explanation weight.

### Approach 2: Path-Based Explanations

On a graph, you can find **paths** between a user and a recommended item:

```
User 1 ──purchased──→ Product A ──category──→ Skincare ←──category── Product D
                                                                      ↑
                                                          Recommended item

Explanation: "Recommended because you purchased Product A,
             and Product D is also in the Skincare category."
```

```python
import networkx as nx

G = nx.from_edgelist(edges)
paths = nx.all_simple_paths(G, source=user_node, target=item_node, cutoff=3)
for path in paths:
    print(" → ".join(str(n) for n in path))
```

### Difficulty: Medium

| Aspect | Details |
|--------|---------|
| **Concepts** | Attention mechanisms, graph paths |
| **Papers** | KGAT (2019), RippleNet (2018) |
| **Implementation** | GAT is in PyG; path extraction uses NetworkX |
| **Experiments** | Qualitative examples + user study (optional) |
| **Time needed** | ~1.5 weeks |

### Pros and Cons

```
Pros:                                Cons:
+ Highly relevant in practice        - Hard to evaluate quantitatively
+ Impressive in thesis defense       - Attention ≠ true explanation
+ Multiple approaches available      - May not improve metrics
+ Good for qualitative analysis      - Path-based can be slow
```

---

## 4. Option C — Cold-Start Recommendation

### The Problem

```
Warm user:                    Cold user:
User A                        User B (new)
  ├── Product 1                 └── ??? 
  ├── Product 2                   
  ├── Product 3               No interaction history.
  └── Product 4               How do we recommend?

Warm item:                    Cold item:
Product X                     Product Y (new)
  ├── User 1                    └── ???
  ├── User 2                   
  └── User 3                  No one has interacted with it.
                               How do we recommend it?
```

### Why GNNs Can Help

Traditional collaborative filtering **cannot** handle cold-start — no interactions means no embeddings.

GNNs can propagate information from **node features**:

```
New User B:
  - Age: 25
  - Gender: M
  - Location: NYC
         |
         v
  GNN propagates features from similar users
         |
         v
  User B gets an embedding even without interactions
```

For cold items:

```
New Product Y:
  - Category: Skincare
  - Brand: CeraVe
  - Price: $18.99
         |
         v
  GNN propagates from similar products (same category/brand)
         |
         v
  Product Y gets an embedding even without interactions
```

### Implementation Sketch

```python
class ColdStartGNN(torch.nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, embed_dim):
        super().__init__()
        # Feature projection (for cold users/items)
        self.user_proj = nn.Linear(user_feat_dim, embed_dim)
        self.item_proj = nn.Linear(item_feat_dim, embed_dim)

        # Learnable embeddings (for warm users/items)
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # GNN layers
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

    def get_initial_embeddings(self, user_features, item_features, is_cold_user, is_cold_item):
        user_emb = torch.where(
            is_cold_user.unsqueeze(1),
            self.user_proj(user_features),  # cold: use features
            self.user_embed.weight           # warm: use learned embedding
        )
        # Similar for items
        return user_emb, item_emb
```

### Evaluation

```
Experiment: Cold-start performance

Split users into:
- Warm: users with ≥20 interactions (normal evaluation)
- Cool: users with 5–10 interactions
- Cold: users with 1–3 interactions

Report metrics per group:

User Group    MF Recall@10    LightGCN Recall@10    ColdStart-GNN Recall@10
─────────────────────────────────────────────────────────────────────────────
Warm (≥20)    0.0XXX          0.0XXX                0.0XXX
Cool (5-10)   0.0XXX          0.0XXX                0.0XXX
Cold (1-3)    0.0XXX          0.0XXX                0.0XXX
```

### Difficulty: Medium-High

| Aspect | Details |
|--------|---------|
| **Concepts** | Feature-based initialization, inductive learning |
| **Papers** | DropoutNet (2017), MetaHIN (2020) |
| **Implementation** | Extends your existing GNN with feature projections |
| **Experiments** | Split by user activity level, compare per group |
| **Time needed** | ~2 weeks |

### Pros and Cons

```
Pros:                                Cons:
+ Very practical problem             - Requires node features (metadata)
+ Clear experimental design          - Harder to beat baselines
+ Directly useful for real systems   - More complex model
+ Strong thesis narrative             - Feature engineering needed
```

---

## 5. Option D — Temporal/Sequential Recommendation

### The Idea

User preferences change over time. A user who bought skincare products in 2022 might shift to haircare in 2024.

```
User A's interactions over time:

t1: Skincare Product 1
t2: Skincare Product 2
t3: Skincare Product 3
t4: Haircare Product 1   ← Preferences shifting
t5: Haircare Product 2
t6: Haircare Product 3

Static model thinks:
  "User likes Skincare AND Haircare equally"

Temporal model knows:
  "User's recent preference is Haircare"
```

### Approaches

**Approach 1: Time-Weighted Edges**

Give recent interactions higher weight:

$$
w_{ui}(t) = e^{-\lambda (t_{now} - t_{interaction})}
$$

Recent interactions have weight close to 1; old interactions decay toward 0.

```python
import numpy as np

lambda_decay = 0.01
t_now = ratings['timestamp'].max()

ratings['weight'] = np.exp(-lambda_decay * (t_now - ratings['timestamp']))
```

**Approach 2: Graph Snapshots**

Build separate graphs for different time windows:

```
Graph at t1:          Graph at t2:          Graph at t3:
User─Product A        User─Product A        User─Product A
                      User─Product B        User─Product B
                                            User─Product C

Process each snapshot with GNN, combine representations over time.
```

**Approach 3: Sequential Models on Graphs**

Combine GNN with a sequence model (RNN/Transformer):

```
GNN ──→ Static user embedding
  +
RNN ──→ Sequential preference embedding
  =
Combined ──→ Recommendation
```

### Difficulty: High

| Aspect | Details |
|--------|---------|
| **Concepts** | Temporal graphs, sequence modeling, decay functions |
| **Papers** | TGN (2020), JODIE (2019), SASRec (2018) |
| **Implementation** | Requires temporal graph handling or hybrid GNN+RNN |
| **Experiments** | Compare static vs temporal, vary time window |
| **Time needed** | ~2–3 weeks |

### Pros and Cons

```
Pros:                                Cons:
+ Very realistic setting             - Most complex option
+ Rich experimental design           - Harder to implement correctly
+ Strong thesis narrative            - Needs timestamps in dataset
+ Active research area               - May not improve for short history
```

---

## 6. Decision Guide

```
┌─────────────────────────────────────────────────────┐
│            Which direction should I pick?            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Your RQ is about metadata/side info?               │
│     → Option A (Heterogeneous GNN)                  │
│                                                     │
│  Your RQ is about trust or transparency?            │
│     → Option B (Explainability)                     │
│                                                     │
│  Your RQ is about sparse data or new users?         │
│     → Option C (Cold-Start)                         │
│                                                     │
│  Your RQ is about changing preferences?             │
│     → Option D (Temporal)                           │
│                                                     │
│  Not sure / just want the safest option?            │
│     → Option A or C (most straightforward)          │
│                                                     │
│  Short on time?                                     │
│     → Option B (smallest implementation effort)     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Difficulty Ranking

```
Easiest ──────────────────────────────── Hardest

  Explainability  < Heterogeneous < Cold-Start < Temporal
    (Option B)      (Option A)      (Option C)   (Option D)
```

### Thesis Impact Ranking

```
Incremental ──────────────────────────── Novel

  Heterogeneous < Explainability < Cold-Start < Temporal
   (Option A)      (Option B)      (Option C)   (Option D)
```

---

## 7. Exercises

1. **Read the abstract and introduction** of one paper from your chosen direction. Write a one-paragraph summary of what they did and what they found.

2. **Sketch the graph structure** your advanced direction would create. How many new node types or edge types does it add?

3. **Design one experiment** specific to your chosen direction. What's the hypothesis, what's the independent variable, what's the metric?

4. **Estimate implementation time.** List the new components you need to build and estimate hours for each.

---

## Key Takeaways

- Pick **ONE** advanced direction — depth over breadth
- **Option A (Heterogeneous)** is the safest choice if your RQ involves metadata
- **Option B (Explainability)** is quickest to implement but hardest to evaluate quantitatively
- **Option C (Cold-Start)** is very practical and has clear experimental design
- **Option D (Temporal)** is the most complex but most novel
- Align your choice with your **research question** from Phase 13
- A well-executed baseline + one advanced direction is better than four half-baked extensions
