# Phase 5 — Graph Convolutional Networks (GCN)

**Week 3 | BTP Roadmap**

---

## 5.1 From Message Passing to GCN

In Phase 4, you learned the general message passing framework. GCN (Graph Convolutional Network) by Kipf & Welling (2017) is the first and most influential instantiation of that framework.

The name "convolutional" comes from an analogy with CNNs: just as a CNN aggregates information from a pixel's spatial neighborhood, a GCN aggregates information from a node's graph neighborhood.

```
CNN (image):                        GCN (graph):

  Pixel neighborhood                  Node neighborhood
  ┌─────────────┐                     
  │  1  2  3    │                     Node B
  │  4 [5] 6    │  → weighted sum     │
  │  7  8  9    │                     Node A ── Node C  → weighted sum → updated A
  └─────────────┘                     │
                                      Node D
```

---

## 5.2 The GCN Equation

The layer-wise propagation rule for GCN:

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

This looks intimidating, but every piece has a clear meaning. Let's break it down term by term.

---

## 5.3 Breaking Down Every Term

### $H^{(l)}$ — Node Feature Matrix at Layer $l$

Shape: $(n \times d_l)$ where $n$ = number of nodes, $d_l$ = feature dimension at layer $l$.

- $H^{(0)} = X$, the initial node features
- $H^{(l)}$ after layer $l$ contains the updated representations

Each row is one node's embedding:

```
H^(l) = [ h_1^(l) ]     ← Node 1's representation at layer l
        [ h_2^(l) ]     ← Node 2's representation at layer l
        [ h_3^(l) ]     ← Node 3's representation at layer l
        [   ...   ]
```

### $A$ — Adjacency Matrix

The original adjacency matrix of the graph. Shape: $(n \times n)$.

```
     1  2  3  4
1 [  0  1  1  0 ]       1 ── 2
2 [  1  0  0  1 ]       │    │
3 [  1  0  0  1 ]       3 ── 4
4 [  0  1  1  0 ]
```

### $\hat{A} = A + I$ — Adjacency with Self-Loops

We add the **identity matrix** $I$ to include self-loops. Why?

Without self-loops, when a node aggregates neighbor messages, it loses its own information:

```
Without self-loop:          With self-loop:
Node A aggregates from      Node A aggregates from
B and C only.               B, C, AND itself.

    B                           B
    │                           │
    A ── C                      A ── C
                                ↺
                           (self-loop)
```

$$\hat{A} = A + I = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$$

Now the diagonal is 1, meaning each node is its own neighbor.

### $\hat{D}$ — Degree Matrix of $\hat{A}$

$\hat{D}$ is the diagonal matrix where $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$ (row sum of $\hat{A}$).

From our example:

```
D̂ = diag(3, 3, 3, 3)

D̂ = [ 3  0  0  0 ]
     [ 0  3  0  0 ]
     [ 0  0  3  0 ]
     [ 0  0  0  3 ]
```

### $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ — Symmetric Normalization

This is the key transformation. Let's understand why we normalize.

**Problem without normalization:**

If we just compute $\hat{A}H^{(l)}$, we're summing over neighbors. A node with 100 neighbors gets a much larger aggregated signal than a node with 2 neighbors. The scale is inconsistent.

**Simple normalization** $\hat{D}^{-1}\hat{A}$ gives us row-normalized (mean) aggregation:

$$(\hat{D}^{-1}\hat{A})_{ij} = \frac{\hat{A}_{ij}}{\hat{D}_{ii}}$$

This divides each row by the degree — a mean over neighbors.

**Symmetric normalization** $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ goes further:

$$(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2})_{ij} = \frac{\hat{A}_{ij}}{\sqrt{\hat{D}_{ii}} \cdot \sqrt{\hat{D}_{jj}}}$$

This normalizes by BOTH the receiver's degree AND the sender's degree. The intuition:

- A message from a high-degree node (connected to many things) is **less specific** → weight it less
- A message to a high-degree node (receives many messages) should be **scaled down** → weight it less

```
Low-degree sender → High-specificity signal → Higher weight
High-degree sender → Low-specificity signal → Lower weight
```

### Worked Normalization Example

For our 4-node graph where $\hat{D}_{ii} = 3$ for all nodes:

$$(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2})_{ij} = \frac{\hat{A}_{ij}}{\sqrt{3}\sqrt{3}} = \frac{\hat{A}_{ij}}{3}$$

```
Normalized Â = (1/3) * [ 1  1  1  0 ]
                       [ 1  1  0  1 ]
                       [ 1  0  1  1 ]
                       [ 0  1  1  1 ]
```

Each non-zero entry becomes $\frac{1}{3}$. This means: each neighbor (including self) contributes equally, normalized by the geometric mean of degrees.

### $W^{(l)}$ — Learnable Weight Matrix

Shape: $(d_l \times d_{l+1})$. This is a standard linear transformation that projects features from one dimension to another.

$H^{(l)}W^{(l)}$ transforms the $d_l$-dimensional features into $d_{l+1}$-dimensional features before aggregation.

This is what makes the GCN a *neural* network — without $W$, you'd just be averaging neighbor features without learning anything new.

### $\sigma$ — Activation Function

Typically ReLU: $\sigma(x) = \max(0, x)$

This adds non-linearity, allowing the model to learn complex patterns.

---

## 5.4 The Full Picture

Putting it all together, one GCN layer does:

```
Step 1: Linear transform
    Z = H^(l) W^(l)                     Project features

Step 2: Neighborhood aggregation with normalization
    Z' = D̂^(-1/2) Â D̂^(-1/2) Z        Aggregate from neighbors

Step 3: Non-linearity
    H^(l+1) = σ(Z')                     Apply activation
```

Or equivalently, for a single node $i$:

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\hat{d}_i}\sqrt{\hat{d}_j}} \mathbf{h}_j^{(l)} W^{(l)}\right)$$

This says: "Update node $i$'s embedding by taking a normalized weighted sum of its neighbors' (and its own) transformed features, then apply ReLU."

---

## 5.5 Layer-by-Layer Feature Propagation

A GCN with $L$ layers stacks this operation:

```
Layer 0 → Layer 1 → Layer 2 → ... → Layer L
  X         H^(1)    H^(2)           H^(L)

Input     1-hop     2-hop          L-hop
features  context   context        context
```

### What Each Layer Does

**Layer 1:** Each node's embedding aggregates information from its direct neighbors.

```
Before Layer 1:            After Layer 1:
Node knows only            Node knows about
its own features           its 1-hop neighborhood

     A [features_A]            A [features_A + avg(features_B, features_C)]
    / \                       / \
   B   C                     B   C
```

**Layer 2:** Each node's embedding now aggregates from neighbors whose embeddings already contain their own neighbor information.

```
After Layer 2:
Node A knows about
its 2-hop neighborhood

   D ── B ── A ── C ── E

A's embedding contains signal from D (via B) and E (via C)
```

**Layer 3:** Three hops of context. And so on.

---

## 5.6 The Oversmoothing Problem

With too many layers, all node embeddings converge to similar values. This is called **oversmoothing**.

```
Layer 1:  Nodes have distinct local features
Layer 2:  Nodes have somewhat broader features
Layer 3:  Nodes start to look similar
Layer 5:  Many nodes have nearly identical embeddings
Layer 10: Almost all nodes are the same → useless!
```

Why? Each layer averages over a larger neighborhood. After enough layers, every node has "seen" the entire graph, and the averaging washes out local structure.

For recommendation:
- 2 layers is standard
- 3 layers sometimes helps
- 4+ layers usually hurts

You'll experiment with this in your BTP (Phase 14, Experiment 2).

---

## 5.7 Simple GCN Implementation in PyTorch

Here's a minimal GCN layer:

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, A_hat_norm, H):
        """
        A_hat_norm: normalized adjacency (D^-1/2 A_hat D^-1/2), shape (n, n)
        H: node features, shape (n, in_dim)
        """
        # Step 1: Linear transform
        Z = self.linear(H)              # (n, out_dim)
        
        # Step 2: Neighborhood aggregation
        H_new = A_hat_norm @ Z          # (n, out_dim)
        
        # Step 3: Activation
        H_new = torch.relu(H_new)
        
        return H_new
```

A 2-layer GCN model:

```python
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, out_dim)
    
    def forward(self, A_hat_norm, X):
        H = self.layer1(A_hat_norm, X)     # Layer 1: (n, hidden_dim)
        H = self.layer2(A_hat_norm, H)     # Layer 2: (n, out_dim)
        return H
```

Computing the normalized adjacency:

```python
def compute_normalized_adjacency(A):
    """Compute D^(-1/2) A_hat D^(-1/2) from adjacency matrix A."""
    A_hat = A + torch.eye(A.size(0))           # Add self-loops
    D_hat = torch.diag(A_hat.sum(dim=1))       # Degree matrix
    D_hat_inv_sqrt = torch.diag(1.0 / torch.sqrt(A_hat.sum(dim=1)))
    A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
    return A_norm
```

---

## 5.8 GCN for Recommendation

How does GCN apply to your BTP?

### The Setup

1. Build the user-item bipartite graph
2. Initialize user and item embeddings (randomly or with features)
3. Run GCN layers to propagate information through the graph
4. Use the final embeddings to predict user-item preferences

```
User-Item Bipartite Graph
         │
         ↓
    GCN Layer 1
    (users learn from items,
     items learn from users)
         │
         ↓
    GCN Layer 2
    (2-hop: users learn from
     similar users via items)
         │
         ↓
    Final Embeddings
    u_i, v_j for all users/items
         │
         ↓
    Score: u_i^T v_j
         │
         ↓
    Top-K Recommendation
```

### The Connection to Matrix Factorization

```
Matrix Factorization:
  Random embeddings → dot product → prediction
  (no graph structure used)

GCN Recommendation:
  Random embeddings → GCN propagation → refined embeddings → dot product → prediction
  (graph structure refines embeddings before scoring)
```

The GCN acts as a "refinement layer" that improves embeddings by incorporating graph structure. If the graph is informative (similar users connect through shared items), the refined embeddings should produce better recommendations.

---

## 5.9 What GCN Doesn't Do Well for Recommendation

Standard GCN was designed for node classification (e.g., classifying papers by topic in a citation graph). When applied to recommendation, it has some limitations:

1. **Unnecessary complexity:** The weight matrix $W$ and activation $\sigma$ add parameters but don't necessarily help for the collaborative filtering task.

2. **Feature transformation changes the embedding space:** The weight matrix can distort the embedding space in ways that don't benefit ranking.

3. **Self-loops mix node types:** In a bipartite graph, users and items are different types. Adding self-loops treats them uniformly.

These observations led to simplified GNN architectures specifically designed for recommendation — most notably **LightGCN** (Phase 8), which removes $W$ and $\sigma$ entirely and keeps only the normalized aggregation. 

But understanding standard GCN is essential because:
- It's the baseline GNN architecture
- It helps you appreciate WHY LightGCN's simplifications work
- You'll compare GCN vs LightGCN in your BTP experiments

---

## Exercises

1. **Compute by hand.** Given this graph and features:

```
A ── B ── C

X = [ 1.0  0.0 ]   ← Node A
    [ 0.0  1.0 ]   ← Node B
    [ 0.5  0.5 ]   ← Node C
```

Compute $\hat{A}$, $\hat{D}$, and the normalized adjacency $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$. Then compute $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}X$ (one round of normalized aggregation without weight matrix).

2. **Think about normalization.** In a recommendation graph, a very popular item (e.g., a bestselling product) is connected to millions of users. How does symmetric normalization affect the messages this item sends? Why is this desirable?

3. **Layer depth.** If your user-item graph has the property that most users are reachable from any item within 4 hops, how many GCN layers would you need at minimum to propagate a signal between them? Why would you NOT use 4 layers in practice?

4. **Compare.** What are the key differences between the GCN update rule and simple mean aggregation from Phase 4? (Hint: weight matrix, normalization style, self-loops.)

---

## Key Takeaways

- GCN implements message passing with **symmetric normalization**, a **learnable weight matrix**, and **non-linear activation**.
- The equation $H^{(l+1)} = \sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$ has four key components: self-loops ($\hat{A}$), normalization ($\hat{D}^{-1/2}$), feature transformation ($W$), and non-linearity ($\sigma$).
- Symmetric normalization weights messages by both sender and receiver degree — high-degree nodes contribute less per connection.
- Each GCN layer extends the receptive field by 1 hop. For recommendation, 2–3 layers is standard.
- **Oversmoothing** limits the number of layers — too many layers make all embeddings converge.
- For recommendation, GCN applies to the user-item bipartite graph: users learn from items and vice versa.
- Standard GCN has limitations for recommendation (unnecessary $W$ and $\sigma$), which motivates LightGCN in Phase 8.
