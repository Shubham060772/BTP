# Phase 2 — Matrix Factorization

**Week 1–2 | BTP Roadmap**

---

## 2.1 The Core Idea

Matrix Factorization (MF) is the most important traditional recommendation technique. It decomposes the sparse User × Item interaction matrix into two smaller dense matrices:

$$R \approx U V^T$$

Where:

- $R$ is the interaction matrix of shape $(m \times n)$ — $m$ users, $n$ items
- $U$ is the user embedding matrix of shape $(m \times d)$
- $V$ is the item embedding matrix of shape $(n \times d)$
- $d$ is the **embedding dimension** (a hyperparameter, e.g., 32 or 64)

Visually:

```
R (m × n)          U (m × d)        V^T (d × n)

[ 5  4  ?  ? ]     [ u1 ]           [ v1  v2  v3  v4 ]
[ 4  ?  5  3 ]  ≈  [ u2 ]     ×
[ ?  5  4  ? ]     [ u3 ]
[ ?  ?  ?  4 ]     [ u4 ]
```

Each row of $U$ is a **user embedding** — a $d$-dimensional vector that captures that user's latent preferences.

Each row of $V$ is an **item embedding** — a $d$-dimensional vector that captures that item's latent characteristics.

---

## 2.2 What Are Embeddings?

An embedding is a learned dense vector that represents an entity in a low-dimensional space.

Consider $d = 2$ for illustration:

```
                  Dimension 2 (e.g., "skincare vs electronics")
                       ↑
                       │    ● User 1
                       │
                       │         ● Product A
                       │
          ─────────────┼──────────────────→ Dimension 1 (e.g., "budget vs premium")
                       │
                       │  ● Product B
                       │
                       │              ● User 2
                       │
```

Key insight: **Users and items live in the same embedding space.** If a user embedding is close to an item embedding, the model predicts the user will like that item.

The dimensions don't have predefined meanings. The model *learns* what each dimension represents during training. With $d = 64$, you get 64 latent factors that might capture things like category preference, brand affinity, price sensitivity, etc. — but you never label them explicitly.

---

## 2.3 Making Predictions

To predict user $u$'s preference for item $i$:

$$\hat{R}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{k=1}^{d} u_{u,k} \cdot v_{i,k}$$

This is simply the **dot product** of the user embedding and the item embedding.

### Worked Example

Suppose $d = 3$:

```
User 1 embedding:    u1 = [0.8,  0.3, -0.5]
Product A embedding: vA = [0.7,  0.4, -0.6]
Product B embedding: vB = [-0.2, 0.9,  0.1]
```

Prediction for User 1 → Product A:
$$\hat{R}_{1,A} = (0.8)(0.7) + (0.3)(0.4) + (-0.5)(-0.6) = 0.56 + 0.12 + 0.30 = 0.98$$

Prediction for User 1 → Product B:
$$\hat{R}_{1,B} = (0.8)(-0.2) + (0.3)(0.9) + (-0.5)(0.1) = -0.16 + 0.27 - 0.05 = 0.06$$

User 1 is predicted to prefer Product A (score 0.98) over Product B (score 0.06). We'd recommend Product A.

---

## 2.4 How Do We Learn the Embeddings?

### The Training Process

We don't compute $U$ and $V$ analytically. We **learn** them by minimizing a loss function.

For **explicit feedback** (ratings), we use Mean Squared Error:

$$\mathcal{L}_{MSE} = \sum_{(u,i) \in \mathcal{O}} \left( R_{ui} - \mathbf{u}_u^T \mathbf{v}_i \right)^2$$

Where $\mathcal{O}$ is the set of observed (user, item) pairs.

The training loop:

```
1. Initialize U and V with small random values
2. For each observed interaction (u, i, R_ui):
   a. Compute prediction: R̂_ui = u_u^T v_i
   b. Compute error: e = R_ui - R̂_ui
   c. Compute gradients of the loss w.r.t. u_u and v_i
   d. Update u_u and v_i using gradient descent
3. Repeat for many epochs
```

### Regularization

Without regularization, the model can overfit to observed ratings and produce wild predictions for unseen pairs. We add L2 regularization:

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} \left( R_{ui} - \mathbf{u}_u^T \mathbf{v}_i \right)^2 + \lambda \left( \|\mathbf{u}_u\|^2 + \|\mathbf{v}_i\|^2 \right)$$

Where $\lambda$ is the regularization strength (typical values: 0.001 to 0.1). This penalizes large embedding values, keeping the model from memorizing the training data.

### Adding Biases

In practice, MF often includes bias terms:

$$\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

Where:
- $\mu$ = global average rating
- $b_u$ = user bias (some users rate everything high)
- $b_i$ = item bias (some items are generally popular)

This separates "User 3 rates everything 4+" from "User 3 specifically likes skincare products."

---

## 2.5 SVD vs Learned Factorization

You may hear about **Singular Value Decomposition (SVD)** in the context of recommendation. Here's the distinction:

**Classical SVD:**
- Decomposes the *entire* matrix: $R = U \Sigma V^T$
- Requires a complete matrix (no missing values)
- Not directly applicable to sparse interaction matrices

**Learned MF (what we use):**
- Only fits on *observed* entries
- Learns $U$ and $V$ via gradient descent
- Handles sparsity naturally
- Can incorporate biases, regularization, and more

For your BTP, you'll use learned MF, not classical SVD.

---

## 2.6 PyTorch Implementation

Here's a minimal implementation to understand the mechanics:

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize with small random values
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Look up embeddings
        u = self.user_embeddings(user_ids)   # (batch_size, d)
        v = self.item_embeddings(item_ids)   # (batch_size, d)

        # Dot product prediction
        scores = (u * v).sum(dim=1)          # (batch_size,)
        return scores
```

Training loop:

```python
model = MatrixFactorization(num_users=1000, num_items=5000, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.MSELoss()

for epoch in range(100):
    for user_ids, item_ids, ratings in dataloader:
        predictions = model(user_ids, item_ids)
        loss = loss_fn(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Notice the structure:

```
User IDs  ──→  Embedding Lookup  ──→  User vectors (batch, d)
                                                │
                                           Dot product  ──→  Predicted scores
                                                │
Item IDs  ──→  Embedding Lookup  ──→  Item vectors (batch, d)
```

This structure is remarkably similar to what your GNN will do later — the difference is *how* the embeddings are computed.

---

## 2.7 Connection to Your BTP

Matrix Factorization is critical for your thesis in two ways:

### As a Baseline

Your BTP experiments will compare multiple models. MF is the standard CF baseline:

```
Model              Recall@10    NDCG@10
────────────────────────────────────────
Popularity         0.05         0.03
Matrix Fact.       0.12         0.08       ← your CF baseline
GCN                0.14         0.09
LightGCN           0.17         0.11       ← your main model
```

Without MF as a baseline, you can't claim your GNN approach is better than traditional CF.

### As Conceptual Foundation

MF and GNN recommendation share the same core structure:

```
Matrix Factorization:
    Random init → Embeddings → Dot product → Score

GNN Recommendation:
    Random init → Graph propagation → Embeddings → Dot product → Score
                  ^^^^^^^^^^^^^^^^
                  This is the new part
```

The GNN adds a step where embeddings are *refined* by propagating information through the user-item graph. You can think of GNN recommendation as MF + graph structure.

---

## 2.8 Implicit Feedback Preview

For explicit feedback (ratings), MSE loss works well. But most real-world recommendation uses **implicit feedback** (clicks, views, purchases), where you only observe positive interactions.

In Phase 9, you'll learn **BPR Loss**, which is designed for implicit feedback:

$$\mathcal{L}_{BPR} = -\sum \log \sigma(\hat{R}_{u,i^+} - \hat{R}_{u,i^-})$$

This trains the model to rank items the user interacted with *higher* than items they didn't. We'll cover this in detail later, but keep in mind that MF can work with both explicit and implicit feedback — the loss function changes.

---

## Exercises

1. **By hand.** Given these 2D embeddings:
   ```
   u1 = [1.0, 0.5]    v_A = [0.8, 0.6]
   u2 = [0.3, 0.9]    v_B = [0.1, 1.0]
   ```
   Compute $\hat{R}_{1,A}$, $\hat{R}_{1,B}$, $\hat{R}_{2,A}$, $\hat{R}_{2,B}$. Which item does each user prefer?

2. **Think about dimensions.** If $d = 1$ (1-dimensional embeddings), what can the model learn? What can it not capture? What about $d = 1000$ with only 500 interactions?

3. **Implement.** Using the PyTorch code above, create a small synthetic dataset (10 users, 20 items, random ratings) and train MF. Plot the loss over epochs.

4. **Conceptual.** Why is regularization important? What happens to predictions for unseen (user, item) pairs without it?

---

## Key Takeaways

- Matrix Factorization decomposes $R \approx UV^T$, learning user and item **embeddings** in a shared latent space.
- Prediction is a **dot product**: $\hat{R}_{ui} = \mathbf{u}_u^T \mathbf{v}_i$.
- Embeddings are learned by minimizing a loss function (MSE for ratings) via gradient descent.
- MF is **the** standard CF baseline for your BTP. Your GNN must outperform it.
- GNN recommendation builds on MF by adding graph-based embedding propagation before the dot product.
- Understanding MF deeply will make GNN-based recommendation feel like a natural extension, not a separate concept.
