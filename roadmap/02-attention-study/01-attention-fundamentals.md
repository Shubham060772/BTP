# Attention Fundamentals

> **Goal:** Understand the attention mechanism from first principles — what it computes, why it works, and how it applies to recommendation.

---

## 1. The Core Intuition

Not all inputs are equally important. When predicting what product a user will buy next, their purchase of a moisturizer last week matters more than a phone charger they bought six months ago.

Attention is a mechanism that lets a model **dynamically decide how much to focus on each part of the input**.

```
Traditional approach:              Attention approach:

[Product A]  ──┐                   [Product A]  ── 0.05 ──┐
[Product B]  ──┤── Average ── h    [Product B]  ── 0.70 ──┤── Weighted Sum ── h
[Product C]  ──┤                   [Product C]  ── 0.20 ──┤
[Product D]  ──┘                   [Product D]  ── 0.05 ──┘

All items contribute equally       Some items matter more for
                                   the current prediction
```

---

## 2. Bahdanau Attention (Additive)

The first widely-used attention mechanism (Bahdanau et al., 2015). Originally designed for machine translation, but the idea generalizes everywhere.

### How It Works

Given a **query** $q$ (what we're looking for) and a set of **keys** $h_1, h_2, \ldots, h_n$ (what we're looking through):

$$
e_i = v^T \tanh(W_q q + W_k h_i)
$$

$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^{n} \exp(e_j)}
$$

$$
\text{context} = \sum_{i=1}^{n} \alpha_i h_i
$$

Where:
- $e_i$ = alignment score (how relevant is $h_i$ to query $q$?)
- $\alpha_i$ = attention weight (normalized via softmax)
- context = weighted combination of all values

```
Step by step:

Query q: "What skincare product should user buy next?"

         Score        Softmax       Weighted
h1 ──→  e1 = 1.2  →  α1 = 0.08  →  0.08 * h1
h2 ──→  e2 = 4.1  →  α2 = 0.72  →  0.72 * h2   ← Most relevant
h3 ──→  e3 = 2.0  →  α3 = 0.15  →  0.15 * h3
h4 ──→  e4 = 0.5  →  α4 = 0.05  →  0.05 * h4
                                     ↓
                                  context = sum
```

### In PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys):
        # query: [batch, query_dim]
        # keys:  [batch, seq_len, key_dim]
        query = self.W_q(query).unsqueeze(1)     # [batch, 1, hidden]
        keys_proj = self.W_k(keys)               # [batch, seq_len, hidden]
        scores = self.v(torch.tanh(query + keys_proj))  # [batch, seq_len, 1]
        scores = scores.squeeze(-1)              # [batch, seq_len]
        weights = F.softmax(scores, dim=-1)      # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), keys)  # [batch, 1, key_dim]
        return context.squeeze(1), weights
```

---

## 3. Luong Attention (Multiplicative)

Simpler than Bahdanau — computes alignment using a dot product or bilinear form.

### Three Variants

```
Dot:       score(q, k) = q^T k
General:   score(q, k) = q^T W k
Concat:    score(q, k) = v^T tanh(W[q; k])   ← similar to Bahdanau
```

**Dot product** is the simplest and most common:

$$
e_i = q^T h_i
$$

Intuition: if query and key point in the same direction in embedding space, the score is high.

```
                q = [0.8, 0.2]  (user interested in skincare)

h1 = [0.9, 0.1]  →  dot = 0.74   ← High (skincare product)
h2 = [0.1, 0.8]  →  dot = 0.24   ← Low  (electronics)
h3 = [0.7, 0.3]  →  dot = 0.62   ← Medium (beauty product)
```

---

## 4. Scaled Dot-Product Attention

The foundation of modern transformers. Adds a scaling factor to prevent softmax from saturating in high dimensions.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### Why Scale by $\sqrt{d_k}$?

When $d_k$ is large, dot products grow in magnitude, pushing softmax into regions with tiny gradients:

```
d_k = 4:    dot products ≈ [-2, 3, 1, -1]     → softmax works well
d_k = 512:  dot products ≈ [-40, 55, 20, -30]  → softmax → [0, 1, 0, 0]
                                                   (almost one-hot, no gradient)

Scaling by √512 ≈ 22.6 brings values back to reasonable range.
```

### Query, Key, Value — What Are They?

```
Think of it like searching a product catalog:

Query (Q):  "What am I looking for?"
            → The current user's need/interest

Key (K):    "What does each product offer?"
            → A label/tag for each candidate

Value (V):  "What information does each product carry?"
            → The actual content/embedding of each candidate

Score = how well Query matches each Key
Output = weighted sum of Values, weighted by scores
```

In practice, Q, K, V are all linear projections of the input:

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_Q(x)  # [batch, seq_len, d_k]
        K = self.W_K(x)  # [batch, seq_len, d_k]
        V = self.W_V(x)  # [batch, seq_len, d_k]

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, seq, seq]
        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights, V)  # [batch, seq_len, d_k]
        return output, weights
```

### Visualizing the Computation

```
User's purchase history: [Moisturizer, Shampoo, Sunscreen, Phone Case]

         Q (queries)              K^T (keys transposed)
    ┌──────────────────┐    ┌─────────────────────────┐
    │ Moist  q1 → ...  │    │ k1  k2  k3  k4         │
    │ Shamp  q2 → ...  │  × │  ↓   ↓   ↓   ↓         │
    │ Sunsc  q3 → ...  │    │ ...                     │
    │ Phone  q4 → ...  │    │                         │
    └──────────────────┘    └─────────────────────────┘
              ↓
        Score Matrix (before softmax):
        ┌──────────────────────────────┐
        │      Moist  Shamp  Sunsc  Phone
        │ Moist  0.9   0.6    0.8   0.1  │  Skincare items attend
        │ Shamp  0.6   0.8    0.5   0.1  │  to each other strongly
        │ Sunsc  0.8   0.5    0.9   0.1  │
        │ Phone  0.1   0.1    0.1   0.7  │  Phone is isolated
        └──────────────────────────────┘
              ↓ softmax per row
        Attention Weights → weighted sum of V
```

---

## 5. Multi-Head Attention

One attention head can only focus on one type of relationship. Multi-head attention runs **multiple attention heads in parallel**, each learning a different pattern.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

### Intuition

```
Head 1: "Which products are in the same category?"
  Moisturizer ←→ Sunscreen  (both skincare)

Head 2: "Which products were bought in the same session?"
  Moisturizer ←→ Shampoo  (bought together last Tuesday)

Head 3: "Which products are in the same price range?"
  Sunscreen ←→ Phone Case  (both ~$15)

Each head captures a different notion of "relevance."
```

### In PyTorch

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Project and reshape: [batch, seq, d_model] → [batch, heads, seq, d_k]
        Q = self.W_Q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)  # [batch, heads, seq, d_k]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_O(context)
```

---

## 6. Attention vs. Other Aggregation Methods

```
Method          Formula                    Captures order?   Adaptive?
─────────────────────────────────────────────────────────────────────
Mean pooling    h = (1/n) Σ h_i            No               No
Max pooling     h = max(h_1, ..., h_n)     No               No
RNN (LSTM)      h = RNN(h_1, ..., h_n)     Yes              No (fixed weights)
Attention       h = Σ α_i h_i              Depends          Yes (α adapts)
─────────────────────────────────────────────────────────────────────

"Adaptive" = weights change based on the input, not just learned once.
```

For recommendation: a user who bought 100 products doesn't care equally about all 100 when deciding to buy the next one. Attention adapts the weights **per prediction**.

---

## 7. Connection to Recommendation

```
User purchase history:
  [CeraVe Moisturizer, USB Cable, Neutrogena Sunscreen, Book: Deep Learning]

Predicting next purchase: candidate = "CeraVe Cleanser"

Without attention:
  User embedding = average of all 4 items → diluted by USB Cable and Book

With attention (target-aware):
  Query = CeraVe Cleanser embedding
  Keys/Values = history item embeddings

  CeraVe Moisturizer  → α = 0.55  (same brand, same category)
  Neutrogena Sunscreen → α = 0.30  (same category)
  USB Cable            → α = 0.08  (irrelevant)
  Book: Deep Learning  → α = 0.07  (irrelevant)

  User embedding = 0.55 * Moist + 0.30 * Sun + 0.08 * USB + 0.07 * Book
                 → heavily weighted toward skincare
```

This is the key insight that drives models like DIN (Deep Interest Network).

---

## 8. Exercises

1. **Implement scaled dot-product attention** from scratch in PyTorch. Given random Q, K, V tensors of shape [2, 5, 64], compute the attention output and visualize the weight matrix as a heatmap.

2. **Compare aggregation methods.** Given 10 random item embeddings (dim=32), compute user representations using: (a) mean pooling, (b) max pooling, (c) attention with a learnable query vector. How different are the results?

3. **Scaling experiment.** Compute dot products between two random vectors for d_k = 4, 64, 512. Observe how the magnitude grows. Verify that scaling by $\sqrt{d_k}$ normalizes the variance back to ~1.

4. **Multi-head attention.** Implement a 4-head attention layer with d_model=64. Verify that each head operates on d_k=16 dimensional subspaces. Visualize the attention weights from different heads — are they different?

---

## Key Takeaways

- **Attention computes a weighted sum** where weights are determined by input similarity
- **Scaled dot-product** is the standard: $\text{softmax}(QK^T / \sqrt{d_k}) V$
- **Multi-head attention** captures multiple types of relationships simultaneously
- **For recommendation:** attention lets the model focus on relevant purchase history, ignoring noise
- Attention is the foundation of transformers (next doc) and modern recommendation models
