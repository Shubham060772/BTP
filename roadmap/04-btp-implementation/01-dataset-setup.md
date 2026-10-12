# Phase 12 — Your BTP Dataset (Week 8)

> **Goal:** Choose, preprocess, and structure your dataset so it can be fed into a GNN-based recommender.

---

## 1. Why Dataset Choice Matters

Your dataset is the foundation of every experiment you run. A poor choice leads to:
- Results that don't generalize
- Graphs that are too sparse or too dense to learn from
- Missing baselines (no one else used this dataset, so you can't compare)

For a BTP, pick a **well-known, publicly available** dataset so your results are reproducible and comparable to published work.

---

## 2. Common Recommendation Datasets

| Dataset | Domain | Size | Interactions | Metadata | Sparsity |
|---------|--------|------|-------------|----------|----------|
| **Amazon-Book** | Products (Books) | 52,643 users, 91,599 items | 2,984,108 | IDs (pre-processed, 10-core) | 99.94% |
| **Amazon-Beauty** | Products (Beauty) | 631,986 users, 112,565 items | 701,528 | Title, category, brand, price | Very high |
| **MovieLens 1M** | Movies | 6,040 users, 3,706 items | 1,000,209 ratings | Genre, year, user demographics | 95.5% |
| **Yelp** | Businesses | ~1.5M users, 150K businesses | ~7M reviews | Business attributes, location | Very high |
| **Last.fm** | Music | 1,892 users, 17,632 artists | 92,834 interactions | Tags, artist metadata | 99.7% |

### Recommended Starting Point

**Amazon-Book (from LightGCN repo)** — it's the best fit:
- Used in the original LightGCN and NGCF papers — your results are directly comparable
- Pre-processed and 10-core filtered, ready to feed into your GNN
- Large enough (~3M interactions) to show meaningful differences between models
- Manageable on a laptop with LightGCN (which is lightweight by design)

**Amazon-Beauty** is a good secondary dataset:
- Smaller (~700K interactions), faster to iterate on
- Rich metadata (category, brand, price) for heterogeneous graph experiments in Phase 15
- 2023 version from McAuley Lab (HuggingFace)

```
Amazon-Book structure (LightGCN format):

train.txt:   UserID Item1 Item2 Item3 ...   (space-separated, one line per user)
test.txt:    UserID Item1 Item2 Item3 ...   (space-separated, one line per user)

Amazon-Beauty structure (JSONL):

{"rating": 5.0, "asin": "B00YQ6X8EO", "parent_asin": "B00YQ6X8EO",
 "user_id": "AGKH...", "timestamp": 1588687728923, "title": "...", ...}
```

---

## 3. Dataset Characteristics to Consider

### 3.1 Size

```
Too small          Just right         Too large
   |                   |                   |
   v                   v                   v
Can't learn        Good signal        Slow training
patterns           Reproducible       Hard to debug
                   Comparable         Needs sampling
```

For a BTP: aim for **100K–1M interactions**. You need enough data to train a GNN but not so much that each experiment takes hours.

### 3.2 Sparsity

Sparsity measures how many possible interactions are actually observed:

$$
\text{Sparsity} = 1 - \frac{|\text{observed interactions}|}{|\text{users}| \times |\text{items}|}
$$

Example for Amazon-Book:

$$
\text{Sparsity} = 1 - \frac{2{,}984{,}108}{52{,}643 \times 91{,}599} = 1 - \frac{2{,}984{,}108}{4{,}821{,}874{,}757} \approx 99.94\%
$$

Most real-world recommendation datasets are **95–99.9% sparse**. This is normal and is exactly the setting where GNNs can help — by propagating information through the graph.

### 3.3 Explicit vs Implicit Feedback

```
Explicit feedback:              Implicit feedback:
User rates an item              User interacts with an item
(1-5 stars)                     (click, purchase, view)
   |                               |
   v                               v
Direct preference signal        Binary signal: interacted or not
   |                               |
   v                               v
Predict: rating value           Predict: will user interact?
```

**For your BTP:** Even if your dataset has explicit ratings, you will likely **convert to implicit feedback** (see Section 4). Why?

1. BPR loss (Phase 9) works with implicit feedback
2. LightGCN was designed for implicit feedback
3. Real-world systems mostly deal with implicit signals

### 3.4 Available Metadata

Metadata enables richer graph structures later:

```
Interaction-only graph:          Metadata-enriched graph:

User 1 ─── Product A            User 1 ─── Product A ─── Skincare
User 1 ─── Product B            User 1 ─── Product B ─── Haircare
User 2 ─── Product A            User 2 ─── Product A ─── Skincare
                                 Product A ─── Brand: CeraVe
```

---

## 4. Data Preprocessing Pipeline

### Step 1: Load and Inspect

**For Amazon-Book (LightGCN format):**

```python
# Amazon-Book is already pre-processed — load train/test directly
train_interactions = []
with open('data/amazon-book/train.txt') as f:
    for line in f:
        parts = line.strip().split()
        user_id = int(parts[0])
        item_ids = [int(x) for x in parts[1:]]
        for item_id in item_ids:
            train_interactions.append((user_id, item_id))

import pandas as pd
train = pd.DataFrame(train_interactions, columns=['user_id', 'item_id'])
print(f"Users:        {train['user_id'].nunique()}")
print(f"Items:        {train['item_id'].nunique()}")
print(f"Interactions: {len(train)}")
```

**For Amazon-Beauty (raw JSONL):**

```python
import pandas as pd
import json

reviews = []
with open('data/amazon-beauty/raw/review_categories/All_Beauty.jsonl') as f:
    for line in f:
        r = json.loads(line)
        reviews.append({
            'user_id': r['user_id'],
            'item_id': r['parent_asin'],
            'rating': r['rating'],
            'timestamp': r['timestamp']
        })

ratings = pd.DataFrame(reviews)
print(f"Users:        {ratings['user_id'].nunique()}")
print(f"Items:        {ratings['item_id'].nunique()}")
print(f"Interactions: {len(ratings)}")
print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
print(f"Sparsity:     {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.4f}")
```

### Step 2: Filter Inactive Users and Items

Remove users/items with very few interactions. They add noise and make the graph sparser without adding useful signal.

```python
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# Count interactions per user and item
user_counts = ratings['user_id'].value_counts()
item_counts = ratings['item_id'].value_counts()

# Filter
active_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
active_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index

ratings = ratings[
    ratings['user_id'].isin(active_users) &
    ratings['item_id'].isin(active_items)
]

print(f"After filtering: {len(ratings)} interactions")
```

### Step 3: Convert Ratings to Implicit Feedback

```python
# Option A: Treat any rating as positive interaction
ratings['label'] = 1

# Option B: Threshold — only high ratings are positive
ratings['label'] = (ratings['rating'] >= 4).astype(int)
```

For most GNN recommender papers, **Option A** is standard: any interaction is positive.

### Step 4: Re-index Users and Items

GNN frameworks need contiguous integer IDs starting from 0:

```python
from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])
ratings['item_id'] = item_encoder.fit_transform(ratings['item_id'])

num_users = ratings['user_id'].nunique()
num_items = ratings['item_id'].nunique()

print(f"User IDs: 0 to {num_users - 1}")
print(f"Item IDs: 0 to {num_items - 1}")
```

### Step 5: Create Train / Validation / Test Splits

For recommendation, **never split randomly**. Use one of:

**Leave-one-out (most common in papers):**

```python
# Sort by timestamp per user
ratings = ratings.sort_values(['user_id', 'timestamp'])

# For each user: last item = test, second-to-last = val, rest = train
def split_by_user(group):
    n = len(group)
    group = group.copy()
    group['split'] = 'train'
    if n >= 3:
        group.iloc[-1, group.columns.get_loc('split')] = 'test'
        group.iloc[-2, group.columns.get_loc('split')] = 'val'
    elif n == 2:
        group.iloc[-1, group.columns.get_loc('split')] = 'test'
    return group

ratings = ratings.groupby('user_id', group_keys=False).apply(split_by_user)

train = ratings[ratings['split'] == 'train']
val   = ratings[ratings['split'] == 'val']
test  = ratings[ratings['split'] == 'test']
```

```
Timeline per user:

|--------- train ----------|-- val --|-- test --|
interaction 1, 2, ... n-2    n-1        n

This respects temporal order — you never train on future data.
```

**Ratio-based split (80/10/10):**

```python
# Per-user 80/10/10 split by timestamp
def ratio_split(group):
    n = len(group)
    group = group.copy()
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    group['split'] = 'train'
    group.iloc[train_end:val_end, group.columns.get_loc('split')] = 'val'
    group.iloc[val_end:, group.columns.get_loc('split')] = 'test'
    return group
```

### Step 6: Build the Graph

```python
import torch

# Create edge_index from training interactions only
user_ids = torch.tensor(train['user_id'].values, dtype=torch.long)
item_ids = torch.tensor(train['item_id'].values, dtype=torch.long)

# For bipartite graph: offset item IDs by num_users
# so user nodes are [0, num_users) and item nodes are [num_users, num_users + num_items)
item_ids_offset = item_ids + num_users

# Undirected edges: user -> item AND item -> user
edge_index = torch.stack([
    torch.cat([user_ids, item_ids_offset]),
    torch.cat([item_ids_offset, user_ids])
], dim=0)

print(f"edge_index shape: {edge_index.shape}")
# Expected: [2, 2 * num_train_interactions]
```

```
Graph structure in memory:

Node IDs:  [0, 1, ..., num_users-1, num_users, num_users+1, ..., num_users+num_items-1]
           |---- user nodes ----|  |------------- item nodes -------------------|

edge_index:
  [[user_0, user_0, ..., item_5, item_5, ...],   <-- source
   [item_3, item_7, ..., user_0, user_1, ...]]   <-- destination
```

---

## 5. From Simple to Rich Graphs

### Level 1: User-Item Bipartite Graph (Start Here)

```
User 1 ─── Product A
  |
User 1 ─── Product C

User 2 ─── Product A
  |
User 2 ─── Product B

User 3 ─── Product B
```

This is your **baseline graph**. Get this working first.

### Level 2: Add Item Metadata

```
               Category: Skincare
                  |
User 1 ─── Product A ─── Brand: CeraVe
  |            |
  |         Price: $15
  |
User 1 ─── Product C ─── Category: Haircare
```

This creates a **heterogeneous graph** — multiple node types and edge types.

### Level 3: Full Heterogeneous Graph

```
           Category: Skincare
               /        \
User 1 ── Product A ── Brand: CeraVe
  |           |              |
  |        Price: $15     Brand: CeraVe
  |                          |
User 2 ── Product B ── Category: Moisturizer
              |
           Brand: Neutrogena
```

Products that share brands, categories, or price ranges are now connected through those nodes — even if no single user purchased both.

---

## 6. Heterogeneous Graphs — Brief Introduction

A heterogeneous graph has multiple node types and/or edge types:

$$
G = (V, E, \phi, \psi)
$$

where:
- $V$ = set of nodes
- $E$ = set of edges
- $\phi: V \to \mathcal{A}$ maps each node to its type (user, product, category, ...)
- $\psi: E \to \mathcal{R}$ maps each edge to its type (purchased, belongs_to, ...)

In PyTorch Geometric, this is represented as `HeteroData`:

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features
data['user'].x = user_features         # [num_users, feature_dim]
data['product'].x = product_features   # [num_products, feature_dim]
data['category'].x = category_features # [num_categories, feature_dim]

# Edge indices
data['user', 'purchased', 'product'].edge_index = user_product_edges
data['product', 'belongs_to', 'category'].edge_index = product_category_edges
```

**Don't start with this.** Get the simple bipartite graph working first. Phase 15 covers heterogeneous GNNs as an advanced direction.

---

## 7. Practical Tips

### Tip 1: Save Preprocessed Data

```python
import pickle

preprocessed = {
    'num_users': num_users,
    'num_items': num_items,
    'edge_index': edge_index,
    'train': train,
    'val': val,
    'test': test,
    'user_encoder': user_encoder,
    'item_encoder': item_encoder,
}

with open('data/processed/amazon_book_processed.pkl', 'wb') as f:
    pickle.dump(preprocessed, f)
```

### Tip 2: Build a Dataset Class

```python
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    def __init__(self, interactions, num_items):
        self.users = torch.tensor(interactions['user_id'].values)
        self.pos_items = torch.tensor(interactions['item_id'].values)
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_item = torch.randint(0, self.num_items, (1,)).item()
        return user, pos_item, neg_item
```

### Tip 3: Verify Your Graph

Always sanity-check:

```python
print(f"Nodes: {num_users + num_items}")
print(f"Edges: {edge_index.shape[1]}")
print(f"Is undirected: {edge_index.shape[1] == 2 * len(train)}")

# Check no self-loops
assert (edge_index[0] != edge_index[1]).all()

# Check node IDs are in valid range
assert edge_index.max() < num_users + num_items
assert edge_index.min() >= 0
```

---

## 8. Exercises

1. **Load Amazon-Book** (already pre-processed from LightGCN repo) and compute:
   - Number of users, items, interactions
   - Sparsity percentage
   - Distribution of interactions per user (plot a histogram)

2. **Build the bipartite graph** from training interactions and verify:
   - The edge_index is undirected
   - Node IDs are contiguous
   - No data leakage (test interactions are not in the graph)

3. **Preprocess Amazon-Beauty** from raw JSONL: apply 5-core and 10-core filtering. How does this affect the number of nodes, edges, and sparsity?

4. **Create a leave-one-out split** for Amazon-Beauty (which has timestamps) and verify that for each user, the test item's timestamp is later than all training items.

---

## Key Takeaways

- **Start with Amazon-Book** — it's the standard benchmark in LightGCN/NGCF papers, making your results directly comparable
- **Use Amazon-Beauty as secondary** — smaller, faster iteration, rich metadata for heterogeneous graph experiments
- **Convert to implicit feedback** — your GNN models (LightGCN, BPR) expect binary signals
- **Never split randomly** — always respect temporal order to avoid data leakage
- **Build a clean preprocessing pipeline** — you'll run it dozens of times during experiments
- **Start with a simple bipartite graph** — add metadata only after your baseline works
- **Verify everything** — check edge counts, node ranges, and data leakage before training
