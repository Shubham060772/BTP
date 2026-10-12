# Self-Attention and Transformers

> **Goal:** Understand self-attention, the Transformer architecture, and why it dominates sequence modeling — the foundation for attention-based recommendation.

---

## 1. From Attention to Self-Attention

Regular attention has a **separate** query attending to a set of keys/values. Self-attention is when **every element in a sequence attends to every other element** in the same sequence.

```
Regular attention:                  Self-attention:
  External query → attends to       Each item in sequence → attends to
  a sequence                         every other item in same sequence

  "CeraVe Cleanser" → attends       Product 1 ←→ Product 2
  to user history                    Product 1 ←→ Product 3
                                     Product 2 ←→ Product 3
                                     (all pairs)
```

Why? Self-attention discovers **relationships within the sequence itself** — which products in a user's history are related to each other.

### Formal Definition

Given a sequence $X = [x_1, x_2, \ldots, x_n]$:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

$$
\text{SelfAttn}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

The input generates its own queries, keys, and values.

```
User history: [Moisturizer, Shampoo, Sunscreen, Phone Case]

Self-attention output for "Moisturizer":
  = 0.4 * Moisturizer_value    (self-connection)
  + 0.15 * Shampoo_value       (beauty, somewhat related)
  + 0.4 * Sunscreen_value      (skincare, very related)
  + 0.05 * Phone Case_value    (unrelated)

Each item's representation is enriched by context from related items.
```

---

## 2. Why Self-Attention Beats RNNs

### The Sequential Bottleneck

```
RNN processing:
  Product 1 → h1 → Product 2 → h2 → Product 3 → h3 → Product 4 → h4
                                                                    ↑
  Information from Product 1 must pass through h2, h3 to reach h4.
  Long-range dependencies get diluted or lost.

Self-attention processing:
  Product 1 ──────────────────────────────────── direct connection
  Product 2 ──────────────────── direct connection
  Product 3 ────── direct connection
  Product 4 ← attends to ALL directly

  Every item can directly access every other item. No bottleneck.
```

### Comparison

```
Property              RNN/LSTM             Self-Attention
──────────────────────────────────────────────────────────
Sequential?           Yes (step by step)   No (fully parallel)
Long-range deps       Degrades over time   Direct connections
Training speed        Slow (sequential)    Fast (parallel)
Max path length       O(n)                 O(1)
Computation per       O(1)                 O(n) per position
  position
Total computation     O(n * d²)            O(n² * d)
──────────────────────────────────────────────────────────

For recommendation: sequences are typically short (10-50 items),
so O(n²) is affordable and the parallelism advantage dominates.
```

---

## 3. The Transformer Architecture

The Transformer (Vaswani et al., 2017) stacks self-attention layers into a deep architecture. For recommendation, we primarily use the **encoder** side.

### Transformer Encoder Block

```
Input Embeddings
       ↓
┌──────────────────────────────┐
│   Multi-Head Self-Attention  │ ← Each item attends to all others
│          + Residual          │
│      + Layer Norm            │
├──────────────────────────────┤
│   Feed-Forward Network       │ ← Position-wise: same MLP per item
│     (Linear → ReLU → Linear)│
│          + Residual          │
│      + Layer Norm            │
└──────────────────────────────┘
       ↓
 Contextualized Embeddings
       ↓
   (Stack N blocks)
```

### In PyTorch

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
```

---

## 4. Positional Encoding

Self-attention is **permutation-invariant** — it treats the sequence as a set. But order matters in recommendation (recent purchases are more relevant).

### The Problem

```
Without positional encoding:
  [Moisturizer, Shampoo, Sunscreen] → same output as
  [Sunscreen, Shampoo, Moisturizer]

But the user bought them in a specific order!
  Moisturizer (Jan) → Shampoo (Mar) → Sunscreen (Jun)
  The recency and sequence pattern carry information.
```

### Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]  = [0, 1, 0, 1, ...]
Position 1: [sin(1), cos(1), sin(0.01), cos(0.01), ...]
Position 2: [sin(2), cos(2), sin(0.02), cos(0.02), ...]

Each position gets a unique "fingerprint" that the model can learn to use.
Low-frequency dimensions encode coarse position, high-frequency encode fine.
```

### Learned Positional Encoding

Simpler — just an embedding table:

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embedding(positions)
```

For recommendation, **learned positional encodings** are more common since sequences are short and fixed-length (e.g., last 50 items).

---

## 5. Layer Normalization

Normalizes across the feature dimension (not batch dimension like BatchNorm):

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

```
BatchNorm: normalizes across the batch for each feature
  → problematic with variable-length sequences

LayerNorm: normalizes across features for each position
  → consistent regardless of batch composition

For a single item embedding [0.5, -1.2, 0.3, 2.1]:
  mean = 0.425, std = 1.17
  normalized = [0.06, -1.39, -0.11, 1.43]
```

### Why It Matters

- Stabilizes training with deep stacks of attention layers
- Makes residual connections work better (prevents values from exploding)

---

## 6. Residual Connections

```
Without residual:                With residual:
  x → Attention → y               x → Attention → y
                                        ↓
                                   output = x + y

Problem: deep stacks lose          Gradient flows directly through
the original signal                skip connections — easier to train
```

The pattern is always: **sublayer → dropout → add residual → layer norm**

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

---

## 7. Causal (Autoregressive) Masking

For sequential recommendation, we need to prevent items from "seeing the future" during training:

```
User history: [A, B, C, D]  (temporal order)

Without mask (bidirectional):     With causal mask:
  A sees [A, B, C, D]             A sees [A]
  B sees [A, B, C, D]             B sees [A, B]
  C sees [A, B, C, D]             C sees [A, B, C]
  D sees [A, B, C, D]             D sees [A, B, C, D]
```

```python
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask  # True = masked (cannot attend)

# For seq_len=4:
# [[False,  True,  True,  True],
#  [False, False,  True,  True],
#  [False, False, False,  True],
#  [False, False, False, False]]
```

This is critical for models like SASRec (next doc).

---

## 8. Putting It Together: Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_items, d_model, num_heads, num_layers,
                 max_seq_len, d_ff, dropout=0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, item_ids, mask=None):
        # item_ids: [batch, seq_len]
        seq_len = item_ids.size(1)
        positions = torch.arange(seq_len, device=item_ids.device)

        x = self.item_embedding(item_ids) + self.pos_encoding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.norm(x)  # [batch, seq_len, d_model]
```

```
Input:  [Product IDs: 42, 17, 88, 5]
           ↓ item embedding + positional encoding
        [e42+p0, e17+p1, e88+p2, e5+p3]
           ↓ Transformer Block 1
        [h1_42, h1_17, h1_88, h1_5]   ← context-aware
           ↓ Transformer Block 2
        [h2_42, h2_17, h2_88, h2_5]   ← deeper context
           ↓ Layer Norm
        Output: each item's embedding now encodes
                its relationship to all other items in the sequence
```

---

## 9. Computational Complexity

```
Self-attention: O(n² * d)
  n = sequence length, d = embedding dimension

For recommendation:
  n = 50 (last 50 purchases), d = 64
  → 50² * 64 = 160,000 operations per layer

For NLP (where this is a problem):
  n = 4096 (tokens), d = 512
  → 4096² * 512 = 8.6 billion operations per layer

Recommendation sequences are short → n² is not a bottleneck.
The quadratic cost only matters for very long sequences (1000+).
```

When n² does become a problem (very active users), solutions exist:
- **Truncation**: keep only the last k items (most common in recsys)
- **Sparse attention**: attend only to nearby or important items
- **Linear attention**: approximate softmax attention in O(n * d)

---

## 10. Exercises

1. **Build a Transformer block.** Implement a single block with 4-head attention, d_model=64, d_ff=256. Pass a random sequence of 10 item embeddings through it. Verify the output shape matches the input.

2. **Causal masking.** Implement causal masking and verify that position $i$ cannot attend to positions $j > i$. Visualize the attention weights as a heatmap — it should be lower-triangular.

3. **Positional encoding comparison.** Implement both sinusoidal and learned positional encodings. For a sequence of 50 items with d_model=64, visualize both as heatmaps. Which has more structure?

4. **Ablation study.** Take your Transformer block and measure the output difference when you remove: (a) positional encoding, (b) layer norm, (c) residual connections. Which removal causes the largest change?

---

## Key Takeaways

- **Self-attention** lets every item in a sequence attend to every other — capturing arbitrary-range dependencies
- **Transformers** stack self-attention + feed-forward blocks with residual connections and layer norm
- **Positional encoding** injects order information (learned embeddings work well for short recsys sequences)
- **Causal masking** prevents future information leakage for sequential prediction
- For recommendation, sequence lengths are short enough that the O(n²) cost is not a concern
- This architecture is the backbone of SASRec, BERT4Rec, and other attention-based recommenders
