# Phase 10 — Build Your First Recommender (Week 7)

## Why This Phase Matters

This is where theory becomes code. You'll build **four models** — two baselines and two GNN models — and compare them. This comparison forms the backbone of your BTP experiments.

---

## 1. The Four Models

```
Model          | Type             | Purpose
───────────────|──────────────────|────────────────────────────
Popularity     | Non-personalized | Simplest baseline
MF             | Collaborative    | Classical baseline
GCN Recommender| GNN              | Generic GNN baseline
LightGCN       | GNN for CF       | Your main model
```

Each model answers: **"Given a user, which items should we recommend?"**

---

## 2. Dataset Preparation Pipeline

Before building any model, prepare your data.

### 2.1 Load and Clean

```python
import pandas as pd
import numpy as np
from collections import defaultdict

# Load Amazon-Book (LightGCN format)
train_interactions = []
with open('data/amazon-book/train.txt') as f:
    for line in f:
        parts = line.strip().split()
        user_id = int(parts[0])
        for item_id in parts[1:]:
            train_interactions.append({'user_id': user_id, 'item_id': int(item_id)})

ratings = pd.DataFrame(train_interactions)

print(f"Interactions: {len(ratings)}")
print(f"Users: {ratings['user_id'].nunique()}")
print(f"Items: {ratings['item_id'].nunique()}")
print(f"Sparsity: {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.4f}")
```

### 2.2 Convert to Implicit Feedback

```python
# For BPR-based training, we only need binary interactions
# Option 1: all ratings → positive
interactions = ratings[['user_id', 'item_id', 'timestamp']].copy()

# Option 2: only ratings >= 4 → positive (more conservative)
# interactions = ratings[ratings['rating'] >= 4][['user_id', 'item_id', 'timestamp']].copy()
```

### 2.3 Reindex Users and Items

```python
# Map to contiguous indices starting from 0
user_ids = interactions['user_id'].unique()
item_ids = interactions['item_id'].unique()

user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

interactions['user_idx'] = interactions['user_id'].map(user2idx)
interactions['item_idx'] = interactions['item_id'].map(item2idx)

num_users = len(user_ids)
num_items = len(item_ids)
```

### 2.4 Train/Validation/Test Split

For recommendation, **how you split matters a lot**.

#### Option A: Random Split (simpler)

```python
from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(interactions, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
```

#### Option B: Time-Based Split (more realistic)

```python
# Sort by timestamp
interactions = interactions.sort_values('timestamp')

# For each user, last interaction → test, second-to-last → val, rest → train
train_list, val_list, test_list = [], [], []

for user, group in interactions.groupby('user_idx'):
    items = group.sort_values('timestamp')
    if len(items) >= 3:
        test_list.append(items.iloc[-1])
        val_list.append(items.iloc[-2])
        train_list.extend(items.iloc[:-2].to_dict('records'))
    elif len(items) == 2:
        test_list.append(items.iloc[-1])
        train_list.append(items.iloc[0].to_dict())
    else:
        train_list.append(items.iloc[0].to_dict())

train_data = pd.DataFrame(train_list)
val_data = pd.DataFrame(val_list)
test_data = pd.DataFrame(test_list)
```

```
Why time-based is more realistic:

Random split:
  Train: User bought Product A (2024), Product C (2023)
  Test:  User bought Product B (2022)  ← BEFORE train data!
  → Predicting the past from the future (data leakage)

Time-based:
  Train: User bought Product B (2022), Product A (2023)
  Test:  User bought Product C (2024)  ← AFTER train data!
  → Predicting the future from the past (realistic)
```

### 2.5 Build User-Item Mappings

```python
# Build positive item sets per user (for negative sampling and evaluation)
train_user_items = defaultdict(set)
for _, row in train_data.iterrows():
    train_user_items[row['user_idx']].add(row['item_idx'])

val_user_items = defaultdict(set)
for _, row in val_data.iterrows():
    val_user_items[row['user_idx']].add(row['item_idx'])

test_user_items = defaultdict(set)
for _, row in test_data.iterrows():
    test_user_items[row['user_idx']].add(row['item_idx'])
```

### 2.6 Build PyG Graph from Training Data

```python
import torch
from torch_geometric.data import Data

# Only use TRAINING interactions for the graph
src = train_data['user_idx'].values
dst = train_data['item_idx'].values + num_users  # offset items

# Bidirectional edges
edge_index = torch.tensor(
    np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ]),
    dtype=torch.long
)

data = Data(edge_index=edge_index)
data.num_users = num_users
data.num_items = num_items
```

---

## 3. Baseline 1 — Popularity-Based Recommender

The simplest possible recommender: recommend the most popular items to everyone.

```python
class PopularityRecommender:
    def __init__(self):
        self.item_popularity = None

    def fit(self, train_user_items, num_items):
        # Count interactions per item
        counts = np.zeros(num_items)
        for items in train_user_items.values():
            for item in items:
                counts[item] += 1
        self.item_popularity = counts

    def recommend(self, user, train_items, k=10):
        """Recommend top-k popular items not already seen"""
        scores = self.item_popularity.copy()
        # Mask already-seen items
        for item in train_items:
            scores[item] = -1
        return np.argsort(scores)[::-1][:k]
```

```
Architecture:

  Count item interactions
         │
         ↓
  Sort by popularity
         │
         ↓
  Remove already-seen
         │
         ↓
  Return top-K

Simple but important: if your GNN can't beat this,
something is wrong with your GNN.
```

**What to log:** Item popularity distribution, Recall@K, NDCG@K.

---

## 4. Baseline 2 — Matrix Factorization

The classical collaborative filtering model.

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users, items):
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        return (u_emb * i_emb).sum(dim=1)

    def recommend(self, user_idx, train_items, k=10):
        """Recommend top-k items for a user"""
        with torch.no_grad():
            u_emb = self.user_embedding(torch.tensor([user_idx]))
            all_scores = (u_emb @ self.item_embedding.weight.T).squeeze()
            # Mask seen items
            all_scores[list(train_items)] = -float('inf')
            return torch.topk(all_scores, k).indices.tolist()
```

```
Architecture:

  R ≈ U · V^T

  User embedding: U [num_users, d]
  Item embedding: V [num_items, d]
  Score: ŷ_ui = u_i^T · v_j

  No graph structure used.
  Learns embeddings purely from (user, item) pairs.
```

### Training MF with BPR

```python
mf_model = MatrixFactorization(num_users, num_items, embed_dim=64)
optimizer = torch.optim.Adam(mf_model.parameters(), lr=0.001)

for epoch in range(100):
    mf_model.train()

    users, pos_items, neg_items = sample_bpr_triplets(
        train_user_items, num_items, batch_size=2048
    )

    pos_scores = mf_model(users, pos_items)
    neg_scores = mf_model(users, neg_items)

    loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

    # Regularization
    reg = (mf_model.user_embedding(users).norm(2).pow(2) +
           mf_model.item_embedding(pos_items).norm(2).pow(2) +
           mf_model.item_embedding(neg_items).norm(2).pow(2)) / len(users)
    loss += 1e-4 * reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**What to log:** Training loss curve, embedding dimension effect, Recall@K, NDCG@K.

---

## 5. Model 3 — Basic GCN Recommender

Apply a standard GCN to the user-item bipartite graph.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # Initial embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # GCN layers WITH weight matrices and activations
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(embed_dim, embed_dim))

    def forward(self, edge_index):
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)  # ← nonlinearity (LightGCN removes this)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        return user_emb, item_emb

    def predict(self, user_emb, item_emb, users, items):
        return (user_emb[users] * item_emb[items]).sum(dim=1)
```

```
Architecture:

  Initial embeddings
         │
         ↓
  GCNConv + ReLU    ← layer 1
         │
         ↓
  GCNConv + ReLU    ← layer 2
         │
         ↓
  Split user/item
         │
         ↓
  Dot product scoring

This is NGCF-like (simplified).
Compare with LightGCN to see the effect of removing W and σ.
```

**What to log:** Training loss, number of parameters (vs MF and LightGCN), Recall@K, NDCG@K.

---

## 6. Model 4 — LightGCN

Your main model (implementation from Phase 8):

```python
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        embs = [x]
        for _ in range(self.num_layers):
            x = self.propagate(x, edge_index)  # no W, no σ
            embs.append(x)

        # Layer combination: mean of all layers
        final = torch.stack(embs, dim=0).mean(dim=0)

        user_emb = final[:self.num_users]
        item_emb = final[self.num_users:]
        return user_emb, item_emb

    def propagate(self, x, edge_index):
        row, col = edge_index
        deg = degree(row, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.unsqueeze(1) * x[col])
        return out

    def predict(self, user_emb, item_emb, users, items):
        return (user_emb[users] * item_emb[items]).sum(dim=1)
```

```
Architecture:

  Initial embeddings
         │
         ↓
  Propagation (no W, no σ)    ← layer 1 → save
         │
         ↓
  Propagation (no W, no σ)    ← layer 2 → save
         │
         ↓
  Propagation (no W, no σ)    ← layer 3 → save
         │
         ↓
  Mean of all layer outputs
         │
         ↓
  Split user/item
         │
         ↓
  Dot product scoring
```

**What to log:** Training loss, convergence speed (vs GCN), Recall@K, NDCG@K.

---

## 7. Training Pipeline Structure

All models share the same evaluation and training infrastructure.

```python
import torch
from torch_geometric.utils import degree

def train_epoch(model, optimizer, data, train_user_items, num_items, batch_size=2048):
    model.train()

    if hasattr(model, 'forward') and 'edge_index' in str(model.forward.__code__.co_varnames):
        # GNN model
        user_emb, item_emb = model(data.edge_index)
    else:
        user_emb = None
        item_emb = None

    users, pos, neg = sample_bpr_triplets(train_user_items, num_items, batch_size)

    if user_emb is not None:
        pos_scores = (user_emb[users] * item_emb[pos]).sum(1)
        neg_scores = (user_emb[users] * item_emb[neg]).sum(1)
    else:
        pos_scores = model(users, pos)
        neg_scores = model(users, neg)

    loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

    # Regularization
    reg = (model.user_embedding(users).norm(2).pow(2) +
           model.item_embedding(pos).norm(2).pow(2) +
           model.item_embedding(neg).norm(2).pow(2)) / batch_size
    total_loss = loss + 1e-4 * reg

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def evaluate(model, data, train_user_items, test_user_items, num_items, k=10):
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward') and 'edge_index' in str(model.forward.__code__.co_varnames):
            user_emb, item_emb = model(data.edge_index)
        else:
            user_emb = model.user_embedding.weight
            item_emb = model.item_embedding.weight

        recalls, ndcgs = [], []

        for user in test_user_items:
            # Score all items
            scores = (user_emb[user] @ item_emb.T)

            # Mask training items
            for item in train_user_items.get(user, set()):
                scores[item] = -float('inf')

            # Top-K
            topk = torch.topk(scores, k).indices.tolist()

            # Metrics
            test_items = test_user_items[user]
            hits = [1 if item in test_items else 0 for item in topk]

            recalls.append(sum(hits) / min(len(test_items), k))
            ndcgs.append(ndcg_at_k(hits))

        return np.mean(recalls), np.mean(ndcgs)
```

---

## 8. Running the Comparison

```python
# Train all models and compare
results = {}

# Popularity
pop = PopularityRecommender()
pop.fit(train_user_items, num_items)
r, n = evaluate_popularity(pop, train_user_items, test_user_items, k=10)
results['Popularity'] = {'Recall@10': r, 'NDCG@10': n}

# MF
mf = MatrixFactorization(num_users, num_items, 64)
train_model(mf, optimizer, 100, ...)
r, n = evaluate(mf, data, train_user_items, test_user_items, num_items)
results['MF'] = {'Recall@10': r, 'NDCG@10': n}

# GCN
gcn = GCNRecommender(num_users, num_items, 64, num_layers=2)
train_model(gcn, optimizer, 100, ...)
r, n = evaluate(gcn, data, train_user_items, test_user_items, num_items)
results['GCN'] = {'Recall@10': r, 'NDCG@10': n}

# LightGCN
lgcn = LightGCN(num_users, num_items, 64, num_layers=3)
train_model(lgcn, optimizer, 100, ...)
r, n = evaluate(lgcn, data, train_user_items, test_user_items, num_items)
results['LightGCN'] = {'Recall@10': r, 'NDCG@10': n}

# Print comparison
print(f"{'Model':<15} {'Recall@10':<12} {'NDCG@10':<12}")
print("-" * 39)
for model_name, metrics in results.items():
    print(f"{model_name:<15} {metrics['Recall@10']:<12.4f} {metrics['NDCG@10']:<12.4f}")
```

Expected output pattern (actual values depend on dataset/hyperparameters):

```
Model           Recall@10    NDCG@10
───────────────────────────────────────
Popularity      0.05xx       0.03xx       ← weakest
MF              0.08xx       0.06xx       ← decent baseline
GCN             0.09xx       0.07xx       ← slightly better
LightGCN        0.11xx       0.08xx       ← best
```

---

## 9. What to Track and Log

For each model, record:

```
Training:
  ☐ Loss per epoch (plot the curve)
  ☐ Training time per epoch
  ☐ Total training time
  ☐ Number of trainable parameters

Hyperparameters:
  ☐ Embedding dimension
  ☐ Learning rate
  ☐ Number of layers (for GNN models)
  ☐ Batch size
  ☐ Regularization weight

Evaluation (every N epochs):
  ☐ Recall@10 on validation set
  ☐ NDCG@10 on validation set
  ☐ Use validation for early stopping / hyperparameter tuning

Final:
  ☐ Recall@10 on test set
  ☐ NDCG@10 on test set
  ☐ Recall@20, NDCG@20 (additional K values)
```

---

## 10. Organizing Your Code

Suggested project structure for your BTP:

```
BTP/
├── data/
│   └── amazon-book/
├── src/
│   ├── data_utils.py       # loading, splitting, sampling
│   ├── models/
│   │   ├── popularity.py   # popularity baseline
│   │   ├── mf.py           # matrix factorization
│   │   ├── gcn_rec.py      # GCN recommender
│   │   └── lightgcn.py     # LightGCN
│   ├── losses.py           # BPR loss
│   ├── metrics.py          # Recall@K, NDCG@K
│   ├── train.py            # training loop
│   └── evaluate.py         # evaluation pipeline
├── notebooks/
│   └── experiments.ipynb   # run and visualize experiments
└── results/
    └── comparison.csv      # model comparison results
```

---

## Exercises

1. **Baseline implementation:** Implement the popularity recommender and verify it works by checking that the most popular items appear in recommendations for all users.

2. **MF implementation:** Implement Matrix Factorization with BPR loss. Train on Amazon-Book and plot the training loss curve.

3. **Parameter counting:** Count the number of trainable parameters in MF, GCN recommender (2 layers), and LightGCN (3 layers), all with embedding dimension 64. Fill in a table.

4. **Early comparison:** Train all four models for 50 epochs each and produce a comparison table. Is LightGCN already winning? If not, why?

5. **Convergence analysis:** Plot Recall@10 on the validation set vs epoch for all models on the same plot. Which converges fastest?

---

## Key Takeaways

1. **Always start with baselines.** Popularity and MF give you a floor — your GNN must beat these to be useful.

2. **Data splitting matters.** Time-based splits are more realistic than random splits. Use leave-one-out (last interaction per user) for evaluation.

3. **All models share the same evaluation.** Same data, same metrics, same splits — the only difference is the model architecture.

4. **LightGCN should outperform the others** if everything is implemented correctly. If it doesn't, debug your graph construction, loss function, or hyperparameters first.

5. **This comparison IS your BTP experiment.** A table comparing Popularity, MF, GCN, and LightGCN with proper metrics is already thesis-worthy.

6. **Log everything.** Future experiments will build on this. Having clean records from day one saves enormous time later.
