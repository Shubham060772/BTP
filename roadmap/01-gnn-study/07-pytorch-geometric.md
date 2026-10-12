# Phase 7 — PyTorch Geometric (Week 4–5)

## Why This Phase Matters

Up until now, you've been learning GNN concepts. Now it's time to **implement** them. PyTorch Geometric (PyG) is the library you'll use to build your BTP recommendation system. This phase teaches you the tools — think of it as learning the workshop before you build the furniture.

---

## 1. What Is PyTorch Geometric?

PyG is a library built on top of PyTorch specifically for deep learning on graphs. It provides:

- Efficient graph data structures
- Pre-built GNN layers (GCN, GraphSAGE, GAT, etc.)
- Standard graph datasets
- Mini-batch support for large graphs
- Utilities for graph manipulation

```
Your BTP stack:

  PyTorch              ← tensor operations, autograd
     │
  PyTorch Geometric    ← graph data structures, GNN layers
     │
  Your model           ← LightGCN for recommendation
     │
  Your dataset         ← user-item interaction graph
```

### Installation

```bash
pip install torch-geometric
# Also install these sparse tensor backends:
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
```

---

## 2. Core Data Structures

### 2.1 The `Data` Object

Everything in PyG revolves around the `Data` object. It holds a single graph.

```python
from torch_geometric.data import Data
import torch

# A simple graph:
#   0 ── 1
#   │    │
#   2 ── 3

edge_index = torch.tensor([
    [0, 1, 1, 0, 2, 3, 3, 2],  # source nodes
    [1, 0, 3, 2, 0, 1, 2, 3],  # target nodes
], dtype=torch.long)

# Node features: 4 nodes, 3 features each
x = torch.tensor([
    [1.0, 0.0, 0.0],  # node 0
    [0.0, 1.0, 0.0],  # node 1
    [0.0, 0.0, 1.0],  # node 2
    [1.0, 1.0, 0.0],  # node 3
])

data = Data(x=x, edge_index=edge_index)
```

Key attributes of `Data`:

| Attribute | Shape | Description |
|---|---|---|
| `x` | `[num_nodes, num_features]` | Node feature matrix |
| `edge_index` | `[2, num_edges]` | Graph connectivity (COO format) |
| `edge_attr` | `[num_edges, edge_features]` | Edge feature matrix |
| `y` | varies | Labels (node-level, edge-level, or graph-level) |

### 2.2 The `edge_index` Format (COO)

This is the most important thing to understand. PyG uses **COO (Coordinate)** format for edges.

```
edge_index = [[source_nodes],
              [target_nodes]]

Example — a triangle: 0 ── 1 ── 2 ── 0

edge_index = [[0, 1, 1, 2, 2, 0],   ← source
              [1, 0, 2, 1, 0, 2]]   ← target

Each column is one directed edge:
  Column 0: 0 → 1
  Column 1: 1 → 0
  Column 2: 1 → 2
  Column 3: 2 → 1
  Column 4: 2 → 0
  Column 5: 0 → 2
```

For **undirected** graphs, each edge appears **twice** (once in each direction).

### For Recommendation: Bipartite edge_index

```python
# Users: 0, 1, 2
# Items: 3, 4, 5  (offset by num_users)
#
# User 0 interacted with Item 3, Item 4
# User 1 interacted with Item 3
# User 2 interacted with Item 4, Item 5

edge_index = torch.tensor([
    [0, 3, 0, 4, 1, 3, 2, 4, 2, 5],  # source
    [3, 0, 4, 0, 3, 1, 4, 2, 5, 2],  # target
])

# Bidirectional: user→item AND item→user
# This lets GNN propagate information both ways
```

### 2.3 Inspecting Your Data

```python
print(data)
# Data(x=[4, 3], edge_index=[2, 8])

print(data.num_nodes)        # 4
print(data.num_edges)        # 8
print(data.num_node_features) # 3
print(data.is_directed())    # False (if edges go both ways)
print(data.has_self_loops()) # False
```

---

## 3. Key Classes

### 3.1 `Dataset` — Loading and Managing Datasets

PyG provides built-in datasets and a base class for custom ones.

```python
from torch_geometric.datasets import Planetoid

# Load the Cora citation dataset (for practice)
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

print(data)
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], ...)
```

### 3.2 `DataLoader` — Mini-Batching

For node-level tasks on a single large graph (like recommendation), you typically don't use `DataLoader` in the standard way. Instead you use **neighbor sampling**:

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # sample 10 neighbors at layer 1, 5 at layer 2
    batch_size=128,          # 128 target nodes per batch
    input_nodes=train_mask,
)

for batch in loader:
    # batch is a subgraph containing the target nodes
    # and their sampled neighborhoods
    out = model(batch.x, batch.edge_index)
```

For recommendation with LightGCN, you'll often train on the **full graph** and only batch the **BPR triplets** (user, positive item, negative item). More on this in Phase 10.

### 3.3 `MessagePassing` — The Heart of PyG

Every GNN layer in PyG inherits from `MessagePassing`. Understanding this class is crucial.

```python
from torch_geometric.nn import MessagePassing

class MyGNNLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # aggregation: 'add', 'mean', 'max'

    def forward(self, x, edge_index):
        # propagate triggers the message passing pipeline
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: features of source nodes for each edge
        # This is where you define WHAT information flows along edges
        return x_j

    def update(self, aggr_out):
        # aggr_out: aggregated messages for each node
        # This is where you define how to UPDATE node features
        return aggr_out
```

The pipeline:

```
propagate() calls:

  1. message(x_j, ...)
     "What information does each neighbor send?"
     x_j = features of the SOURCE node for each edge
     x_i = features of the TARGET node for each edge

  2. aggregate(messages)
     "How to combine all incoming messages?"
     Defined by aggr='add'/'mean'/'max'

  3. update(aggregated)
     "How to update the node's features?"
     Post-processing after aggregation
```

This is exactly the **message passing** framework from Phase 4, implemented as a class.

---

## 4. Built-in GNN Layers

### 4.1 GCNConv

```python
from torch_geometric.nn import GCNConv

conv = GCNConv(in_channels=16, out_channels=32)
out = conv(x, edge_index)
# out shape: [num_nodes, 32]
```

Implements: $h_i' = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i d_j}} W h_j$

### 4.2 SAGEConv

```python
from torch_geometric.nn import SAGEConv

conv = SAGEConv(in_channels=16, out_channels=32)
out = conv(x, edge_index)
```

Implements: $h_i' = W_1 h_i + W_2 \cdot \text{mean}_{j \in \mathcal{N}(i)} h_j$

### 4.3 GATConv

```python
from torch_geometric.nn import GATConv

conv = GATConv(in_channels=16, out_channels=32, heads=4)
out = conv(x, edge_index)
# out shape: [num_nodes, 32 * 4] (concatenated heads)
```

---

## 5. Building a GNN Model Step by Step

Let's build a complete 2-layer GCN for node classification (as a warm-up before recommendation).

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 1. Load dataset
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# 2. Define model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(
    in_channels=dataset.num_node_features,   # 1433
    hidden_channels=64,
    out_channels=dataset.num_classes,         # 7
)

# 3. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = correct / data.test_mask.sum()
    return acc.item()

# 4. Train
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch {epoch:3d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}')
```

This is **not** your recommendation model — it's a warm-up to make sure you understand the PyG workflow.

---

## 6. Creating a Custom Dataset for Recommendation

For your BTP, you'll need to convert interaction data into a PyG graph.

```python
import pandas as pd
import torch
from torch_geometric.data import Data

# Load interactions (e.g., from Amazon-Book or Amazon-Beauty)
df = pd.read_csv('ratings.csv')
# Columns: user_id, item_id, rating, timestamp

# Map IDs to contiguous indices
user_ids = df['user_id'].unique()
item_ids = df['item_id'].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
item_map = {iid: i + len(user_ids) for i, iid in enumerate(item_ids)}
# Items are offset by num_users so all node indices are unique

num_users = len(user_ids)
num_items = len(item_ids)

# Build edge_index (bidirectional)
src = [user_map[u] for u in df['user_id']]
dst = [item_map[i] for i in df['item_id']]

edge_index = torch.tensor([
    src + dst,  # user→item + item→user
    dst + src,
], dtype=torch.long)

# Create the graph
data = Data(edge_index=edge_index)
data.num_users = num_users
data.num_items = num_items

print(f'Users: {num_users}, Items: {num_items}')
print(f'Interactions: {len(df)}')
print(f'Graph edges: {data.num_edges} (2x interactions, bidirectional)')
```

**Important note:** For collaborative filtering (LightGCN), nodes typically have **no input features** — they only have **learnable embeddings**. This is different from GCN on citation networks where nodes have feature vectors.

```python
# In LightGCN, node features are learnable parameters:
user_embedding = torch.nn.Embedding(num_users, embed_dim)
item_embedding = torch.nn.Embedding(num_items, embed_dim)

# NOT from a feature matrix x
```

---

## 7. Loading Amazon-Book (Your BTP Dataset)

```python
# Amazon-Book is pre-processed in the LightGCN format:
# Each line: user_id item_id1 item_id2 item_id3 ...

import torch
import numpy as np

def load_amazon_book(data_dir='data/amazon-book'):
    train_interactions = []
    with open(f'{data_dir}/train.txt') as f:
        for line in f:
            parts = line.strip().split()
            user_id = int(parts[0])
            for item_id in parts[1:]:
                train_interactions.append((user_id, int(item_id)))

    test_interactions = []
    with open(f'{data_dir}/test.txt') as f:
        for line in f:
            parts = line.strip().split()
            user_id = int(parts[0])
            for item_id in parts[1:]:
                test_interactions.append((user_id, int(item_id)))

    return train_interactions, test_interactions

# Then build edge_index as shown above
```

---

## 8. Training Loop Template for Recommendation

This is the general structure you'll use for your BTP models:

```python
import torch

def train_one_epoch(model, optimizer, data, num_users, num_items):
    model.train()

    # Get all embeddings via GNN forward pass
    user_emb, item_emb = model(data.edge_index)

    # Sample BPR triplets
    users, pos_items, neg_items = sample_triplets(
        data.edge_index, num_users, num_items
    )

    # Look up embeddings
    u_emb = user_emb[users]
    pos_emb = item_emb[pos_items]
    neg_emb = item_emb[neg_items]

    # BPR loss
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    # Add regularization
    reg_loss = (u_emb.norm(2).pow(2) +
                pos_emb.norm(2).pow(2) +
                neg_emb.norm(2).pow(2)) / len(users)
    total_loss = loss + 1e-4 * reg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

This is a **preview** — you'll build this properly in Phase 10.

---

## 9. Important Tips

### Don't force recommendation into GCNConv blindly

```
WRONG approach:
  "I have a recommendation problem → let me use GCNConv"
  → Add node features → GCNConv → predict ratings

RIGHT approach:
  "I have a recommendation problem"
  → What is my graph structure? (bipartite user-item)
  → What is my task? (ranking, not classification)
  → What loss function? (BPR, not cross-entropy)
  → What model fits? (LightGCN, designed for this)
```

### Think about your graph FIRST

```
Step 1: What are my nodes?
  → Users and Items

Step 2: What are my edges?
  → User-Item interactions

Step 3: Do nodes have features?
  → For pure CF: no features, just learnable embeddings
  → For content-based: yes, item metadata

Step 4: What is my prediction task?
  → Rank items for each user (link prediction / ranking)

Step 5: NOW choose the model
  → LightGCN for pure CF
  → Or a feature-aware model if using metadata
```

### GPU Usage

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
data = data.to(device)
```

---

## Exercises

1. **Data creation:** Load the Amazon-Book dataset, convert it to a PyG `Data` object with a bipartite `edge_index`, and print the graph statistics (number of users, items, edges).

2. **Warm-up model:** Build a 2-layer GCN on the Cora dataset using PyG. Train for 200 epochs and report test accuracy.

3. **Custom MessagePassing:** Implement a simple message passing layer that averages neighbor features (without any weight matrix). Test it on a small toy graph.

4. **Neighbor loader:** Use `NeighborLoader` to create mini-batches from the Cora dataset. Print the subgraph size for different `num_neighbors` settings.

---

## Key Takeaways

1. **PyG's `Data` object** holds your graph: node features in `x`, connectivity in `edge_index` (COO format), and labels in `y`.

2. **`edge_index` is COO format** — a `[2, num_edges]` tensor where each column is a directed edge. Undirected edges appear twice.

3. **`MessagePassing`** is the base class for all GNN layers — it calls `message()`, `aggregate()`, `update()` in sequence.

4. **For recommendation**, your graph is bipartite (users + items), nodes typically have **no input features** (only learnable embeddings), and the task is **ranking** (not classification).

5. **Don't blindly apply GCNConv** to recommendation. Understand your graph structure and task first, then choose the appropriate model (LightGCN for pure CF).

6. **The training loop for recommendation** is different from classification — you sample BPR triplets and optimize a ranking loss, not cross-entropy.
