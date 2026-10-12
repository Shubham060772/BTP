# Advanced Attention Techniques for Recommendation

> **Goal:** Understand advanced attention patterns — multi-head interest modeling, cross-attention, attention pooling, and efficiency techniques — that go beyond basic SASRec/BERT4Rec.

---

## 1. Multi-Head Attention for Diverse User Interests

Users don't have a single interest — they have **multiple parallel interests** that coexist.

```
User's history: [Moisturizer, Python Book, Sunscreen, Mechanical Keyboard,
                 Eye Cream, USB Hub, Face Mask, Monitor Stand]

Interest 1 (Head 1): Skincare routine
  → Moisturizer, Sunscreen, Eye Cream, Face Mask

Interest 2 (Head 2): Desk setup
  → Mechanical Keyboard, USB Hub, Monitor Stand

Interest 3 (Head 3): Learning
  → Python Book

Interest 4 (Head 4): Recent purchases
  → Face Mask, Monitor Stand (recency-weighted)
```

Each attention head can specialize in capturing a different interest dimension. This is why multi-head attention is particularly well-suited for recommendation.

### Multi-Interest Extraction

Some models go further and explicitly extract **k interest vectors** per user:

```python
class MultiInterestExtractor(nn.Module):
    def __init__(self, d_model, num_interests):
        super().__init__()
        self.num_interests = num_interests
        self.interest_queries = nn.Parameter(
            torch.randn(num_interests, d_model)
        )
        self.attention = nn.MultiheadAttention(d_model, num_heads=1,
                                                batch_first=True)

    def forward(self, item_sequence):
        # item_sequence: [batch, seq_len, d_model]
        batch_size = item_sequence.size(0)

        # Repeat interest queries for each batch element
        queries = self.interest_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_interests, d_model]

        # Each interest query attends to the full sequence
        interests, weights = self.attention(
            queries, item_sequence, item_sequence
        )  # interests: [batch, num_interests, d_model]

        return interests, weights
```

```
Output per user: k interest vectors

Interest 1: [0.8, 0.1, ...]  → "skincare" direction
Interest 2: [0.1, 0.7, ...]  → "tech accessories" direction
Interest 3: [0.3, 0.3, ...]  → "general" direction

At prediction time: score candidate against ALL interest vectors,
take the max → "does this item match ANY of the user's interests?"
```

### Models Using This

- **MIND** (Multi-Interest Network with Dynamic Routing) — Alibaba, uses capsule networks
- **ComiRec** (Controllable Multi-Interest Rec) — uses multi-head attention as interest extractor
- **Re4** — uses separate attention heads per interest

---

## 2. Cross-Attention Between User History and Candidates

Self-attention operates **within** a single sequence. Cross-attention operates **between two different sequences**.

```
Self-attention:
  History items attend to each other
  [A, B, C, D] → each attends to all → [A', B', C', D']

Cross-attention:
  Candidate items attend to history items
  Candidates = [X, Y, Z]     (items to rank)
  History    = [A, B, C, D]   (what user bought)

  X attends to [A, B, C, D] → X' (X in context of this user)
  Y attends to [A, B, C, D] → Y' (Y in context of this user)
  Z attends to [A, B, C, D] → Z' (Z in context of this user)
```

### Implementation

```python
class CrossAttentionRanker(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, candidate_embs, history_embs):
        # candidate_embs: [batch, num_candidates, d_model]
        # history_embs:   [batch, seq_len, d_model]

        # Candidates attend to history
        # Q = candidates, K = V = history
        enriched, attn_weights = self.cross_attn(
            candidate_embs, history_embs, history_embs
        )
        enriched = self.norm(candidate_embs + enriched)

        scores = self.scorer(enriched).squeeze(-1)  # [batch, num_candidates]
        return scores, attn_weights
```

### Why Cross-Attention Is Powerful

```
Without cross-attention:
  user_emb = encode(history)           → fixed vector
  score(item) = dot(user_emb, item_emb) → same user_emb for all items

With cross-attention:
  For each candidate item:
    enriched_item = cross_attn(item, history)  → item is now user-aware
    score = f(enriched_item)

  The candidate "looks at" the user's history and understands
  its own relevance in that specific context.
```

This is similar to DIN's target-aware attention but more flexible — the candidate can attend to the **full history with multi-head attention** rather than a single-pass MLP.

---

## 3. Attention Pooling vs. Other Pooling Methods

After encoding a sequence, you need a **single vector** for the user. How you aggregate matters.

### Mean Pooling

```python
user_emb = sequence_output.mean(dim=1)  # [batch, d_model]
```

```
Pros: Simple, no extra parameters
Cons: All items weighted equally — noise dilutes signal
```

### Last-Item Pooling

```python
user_emb = sequence_output[:, -1, :]  # [batch, d_model]
```

```
Pros: Captures recency, used by SASRec
Cons: Ignores long-term interests
```

### Attention Pooling

A learnable query vector attends to the full sequence:

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = d_model ** 0.5

    def forward(self, sequence_output, mask=None):
        # sequence_output: [batch, seq_len, d_model]
        scores = torch.matmul(sequence_output, self.query) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)           # [batch, seq_len]
        pooled = torch.bmm(weights.unsqueeze(1),
                          sequence_output).squeeze(1)  # [batch, d_model]
        return pooled, weights
```

```
Pros: Learns which items matter for the "overall" user representation
Cons: Static query — doesn't adapt to different prediction contexts

Best of both worlds: use attention pooling for overall user embedding,
then cross-attention for per-candidate adaptation.
```

### Comparison on a Concrete Example

```
User sequence after Transformer: [h1, h2, h3, h4, h5]
h1 = Moisturizer (relevant)   h4 = USB Cable (noise)
h2 = Sunscreen (relevant)     h5 = Eye Cream (relevant)
h3 = Python Book (noise)

Mean pooling:     (h1 + h2 + h3 + h4 + h5) / 5
                  → diluted by h3, h4

Last pooling:     h5
                  → captures Eye Cream but misses Moisturizer

Attention pooling: 0.30*h1 + 0.25*h2 + 0.05*h3 + 0.05*h4 + 0.35*h5
                  → focuses on skincare items
```

---

## 4. Handling Long Sequences Efficiently

When user histories grow long (hundreds or thousands of items), standard O(n²) attention becomes expensive.

### Truncation (Simplest)

```python
max_len = 50
history = history[-max_len:]  # keep most recent
```

Most common in practice. Works well because recent items are most predictive.

### Sparse Attention

Only attend to a subset of positions:

```
Full attention (n=8):          Sparse attention patterns:

x x x x x x x x              Local:        Strided:
x x x x x x x x              x x x . . . . .    x . . x . . x .
x x x x x x x x              x x x x . . . .    . x . . x . . x
x x x x x x x x              . x x x x . . .    x . x . . x . .
x x x x x x x x              . . x x x x . .    . x . x . . x .
x x x x x x x x              . . . x x x x .    . . x . x . . x
x x x x x x x x              . . . . x x x x    x . . x . x . .
x x x x x x x x              . . . . . x x x    . x . . x . x .

O(n²)                         O(n * w)            O(n * √n)
```

### Linear Attention

Replace softmax attention with a kernel approximation:

$$
\text{Standard: } \text{softmax}(QK^T)V \quad \text{— O}(n^2 d)
$$

$$
\text{Linear: } \phi(Q)(\phi(K)^T V) \quad \text{— O}(n d^2)
$$

When $d \ll n$, this is much cheaper.

```python
class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = F.elu(self.W_Q(x)) + 1  # kernel feature map (must be non-negative)
        K = F.elu(self.W_K(x)) + 1
        V = self.W_V(x)

        # Compute K^T V first: [batch, d, d] — doesn't depend on n²
        KV = torch.bmm(K.transpose(1, 2), V)      # [batch, d, d]
        Z = 1.0 / (torch.bmm(Q, K.sum(dim=1, keepdim=True).transpose(1, 2)) + 1e-6)
        output = torch.bmm(Q, KV) * Z              # [batch, seq, d]
        return output
```

### Practical Guidance

```
Sequence length    Best approach
──────────────────────────────────
< 50               Full attention (standard)
50-200              Full attention (still fine on GPU)
200-1000            Truncation or sparse attention
> 1000              Linear attention or hierarchical
──────────────────────────────────

For Amazon product recommendation: most users have < 100 interactions
after filtering → full attention is fine. Don't over-engineer.
```

---

## 5. Limitations of Attention-Based Recommenders

### Cold Start

```
New user with 0-2 interactions:
  → Attention has nothing to attend to
  → Falls back to popularity or content-based features

Solutions:
  - Use item metadata as initial features
  - Pre-train embeddings from product descriptions
  - Hybrid: combine attention with content-based features
```

### No Collaborative Signal

```
Attention models process each user's sequence independently.
They don't leverage the insight that "users who bought A also bought B."

   User 1: [A, B, C] → Attention → User 1 embedding
   User 2: [A, B, D] → Attention → User 2 embedding

These two users are processed independently even though they
share items A and B. A GNN would connect them through shared items.
```

This is a fundamental limitation and the key motivation for combining attention with GNNs.

### Position Bias

```
Attention models with positional encoding learn that recent items
are more important. This is usually correct but can miss:
  - Periodic purchases (re-buy moisturizer every 3 months)
  - Seasonal patterns (sunscreen in summer)

The positional encoding treats position as a proxy for relevance,
which is imperfect.
```

### Computational Cost at Scale

```
Industrial recommendation (Alibaba, Amazon):
  - Millions of candidate items
  - Scoring every candidate with cross-attention is too expensive

Solution: two-stage pipeline
  1. Retrieval: fast approximate search (ANN) → top 1000 candidates
  2. Ranking: attention-based model scores 1000 candidates

Your BTP operates at research scale where this isn't a concern.
```

---

## 6. Recent Trends

### Disentangled Attention

Separate attention into content and position components:

$$
\text{score}(i, j) = \underbrace{q_i^c \cdot k_j^c}_{\text{content-content}} + \underbrace{q_i^c \cdot k_j^p}_{\text{content-position}} + \underbrace{q_i^p \cdot k_j^c}_{\text{position-content}}
$$

### Mixture of Experts + Attention

Use different expert networks for different user interest segments, with an attention-based gating mechanism to combine them.

### Contrastive Learning + Attention

Augment attention-based models with contrastive objectives:
- Mask/crop/reorder the sequence → create augmented views
- Train the model to produce similar representations for augmented views of the same user

Example: **CL4SRec** — contrastive learning for sequential recommendation.

---

## 7. Exercises

1. **Multi-interest extraction.** Implement a model that extracts 4 interest vectors per user using learnable query vectors. On Amazon-Beauty, visualize which items each interest vector attends to. Do the interests correspond to interpretable categories?

2. **Cross-attention ranking.** Implement a ranker that uses cross-attention between candidates and user history. Compare against dot-product scoring (SASRec-style). Does cross-attention improve ranking quality?

3. **Pooling comparison.** For the same Transformer encoder, compare Recall@10 using: (a) last-item pooling, (b) mean pooling, (c) attention pooling. Which works best on Amazon-Book?

4. **Efficiency.** Measure the forward-pass time of standard attention vs. linear attention for sequence lengths [50, 100, 200, 500, 1000]. At what length does linear attention become faster?

---

## Key Takeaways

- **Multi-head attention** naturally captures diverse user interests — each head can specialize
- **Cross-attention** between candidates and history enables per-candidate user representations
- **Attention pooling** is a principled alternative to mean/last pooling for aggregating sequences
- **Efficiency techniques** (sparse, linear attention) exist but are rarely needed for recsys-scale sequences
- **Fundamental limitation:** attention models process users independently — they miss collaborative signals that GNNs capture
- This limitation motivates **combining GNNs with attention** (next: 03-combined-approaches/)
