# Attention-Based Recommendation Models

> **Goal:** Understand the key models that apply attention to recommendation — SASRec, BERT4Rec, NARM, and DIN — their architectures, strengths, and when to use each.

---

## 1. Why Attention for Recommendation?

Users have **diverse, context-dependent interests**. A simple average of their history loses this nuance.

```
User's full purchase history:
  [CeraVe Cleanser, USB-C Cable, Sunscreen SPF50, Python Book,
   Lip Balm, HDMI Adapter, Eye Cream, Mechanical Keyboard]

Predicting: "Will user buy Vitamin C Serum?"

Mean pooling: averages ALL items → diluted by tech purchases
Attention:    focuses on [CeraVe Cleanser, Sunscreen, Lip Balm, Eye Cream]
              → strong skincare signal
```

Different models use attention differently:

```
Model     Attention Type     Input            Predicts
──────────────────────────────────────────────────────────
SASRec    Causal self-attn   User sequence    Next item
BERT4Rec  Bidirectional      Masked sequence  Masked item
NARM      Session attention  Session items    Next click
DIN       Target-aware       History + target Click probability
```

---

## 2. SASRec — Self-Attentive Sequential Recommendation

**Paper:** Kang & McAuley, 2018

The most influential attention-based recommender. Applies a **causal Transformer encoder** to the user's interaction sequence.

### Architecture

```
User sequence: [Product 1, Product 2, Product 3, Product 4]
                    ↓           ↓           ↓           ↓
              Item Embedding + Positional Encoding
                    ↓           ↓           ↓           ↓
            ┌────────────────────────────────────────────┐
            │     Causal Self-Attention (masked)         │
            │     Product 1 sees: [1]                    │
            │     Product 2 sees: [1, 2]                 │
            │     Product 3 sees: [1, 2, 3]              │
            │     Product 4 sees: [1, 2, 3, 4]           │
            ├────────────────────────────────────────────┤
            │     Feed-Forward Network                   │
            └────────────────────────────────────────────┘
                    ↓           ↓           ↓           ↓
               h1          h2          h3          h4
                                                    ↓
                                              dot product with
                                              all item embeddings
                                                    ↓
                                              Next item prediction
```

### Key Ideas

1. **Causal masking**: position $t$ only attends to positions $\leq t$ (no future leakage)
2. **Shared item embedding table**: same embeddings used for input and prediction
3. **Training**: at each position, predict the next item — get $n-1$ training signals from a sequence of length $n$

### Simplified Implementation

```python
class SASRec(nn.Module):
    def __init__(self, num_items, d_model, num_heads, num_layers,
                 max_seq_len, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, item_seq):
        # item_seq: [batch, seq_len] — padded with 0s on the left
        seq_len = item_seq.size(1)
        positions = torch.arange(seq_len, device=item_seq.device)
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.dropout(x)

        # Causal mask: prevent attending to future items
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device),
                         diagonal=1).bool()

        for block in self.blocks:
            x = block(x, mask=mask)

        return self.norm(x)  # [batch, seq_len, d_model]

    def predict(self, seq_output, candidate_items):
        # seq_output: [batch, d_model] (last position)
        # candidate_items: [batch, num_candidates]
        item_embs = self.item_emb(candidate_items)  # [batch, num_cand, d_model]
        scores = torch.bmm(item_embs, seq_output.unsqueeze(-1)).squeeze(-1)
        return scores  # [batch, num_candidates]
```

### Training

```
Sequence: [A, B, C, D, E]

Training pairs (input → target):
  [A]          → B
  [A, B]       → C
  [A, B, C]    → D
  [A, B, C, D] → E

Loss: Binary cross-entropy or softmax cross-entropy
      per position, summed over the sequence.
```

### Typical Hyperparameters

```
d_model:       64
num_heads:     2
num_layers:    2
max_seq_len:   50
dropout:       0.2 (SASRec uses heavy dropout)
learning_rate: 0.001
batch_size:    128
```

---

## 3. BERT4Rec — Bidirectional Self-Attention

**Paper:** Sun et al., 2019

Instead of causal (left-to-right) attention, BERT4Rec uses **bidirectional** self-attention with a **masked item prediction** objective (like BERT in NLP).

### Key Difference from SASRec

```
SASRec (causal):
  [A, B, C, D] → predict E (next item)
  Each position only sees past items.

BERT4Rec (bidirectional):
  [A, [MASK], C, D] → predict B (masked item)
  Each position sees ALL other items (past and future).
```

### Architecture

```
Input: [Moisturizer, [MASK], Sunscreen, Eye Cream]
            ↓           ↓         ↓          ↓
       Embedding + Position
            ↓           ↓         ↓          ↓
   ┌─────────────────────────────────────────────┐
   │  Bidirectional Self-Attention               │
   │  [MASK] sees: Moisturizer, Sunscreen,       │
   │               Eye Cream (all context)       │
   └─────────────────────────────────────────────┘
            ↓           ↓         ↓          ↓
                    h_mask
                        ↓
                  Predict: Shampoo
```

### Training: Cloze Task

```
Original:  [A, B, C, D, E]
Masked:    [A, [M], C, [M], E]   (random 20% of items masked)
Target:    predict B at position 2, D at position 4

At test time:
  Append [MASK] at the end → predict next item
  [A, B, C, D, E, [MASK]] → predict F
```

### SASRec vs BERT4Rec

```
Aspect              SASRec              BERT4Rec
────────────────────────────────────────────────────
Attention           Causal (left→right) Bidirectional
Training objective  Next-item pred      Masked item pred
Inference           Natural (predict    Append [MASK] token
                    from last pos)
Context used        Only past items     All surrounding items
Performance         Strong baseline     Often similar or slightly
                                        better, but more complex
Recommendation fit  Sequential tasks    When context from
                                        both sides helps
```

---

## 4. NARM — Neural Attentive Session Recommendation

**Paper:** Li et al., 2017

Designed for **session-based** recommendation where you only have the current session (no long-term history).

### Architecture

```
Session: [Product clicked 1, clicked 2, clicked 3]
              ↓            ↓            ↓
          GRU encoder (captures sequential patterns)
              ↓            ↓            ↓
           h1            h2           h3
              ↓            ↓            ↓
         ┌────────────────────────────────┐
         │  Attention (query = h3, the    │
         │  last hidden state)            │
         │  α1 * h1 + α2 * h2 + α3 * h3  │
         └────────────────────────────────┘
              ↓
         Session representation = concat(attention_output, h3)
              ↓
         Score all items → predict next click
```

### Key Ideas

1. **GRU** captures sequential patterns
2. **Attention over GRU states** captures the user's main purpose in the session
3. Combines **global purpose** (attention) with **current interest** (last state)

```
Session: [Shampoo, Conditioner, Hair Mask, Blow Dryer]

GRU last state h4: current focus = hair styling tools
Attention output:   main purpose = haircare routine

Combined → predict: "Heat Protectant Spray"
(related to both the routine and the current tool interest)
```

---

## 5. DIN — Deep Interest Network

**Paper:** Zhou et al., 2018 (Alibaba)

Designed for **click-through rate (CTR) prediction** in e-commerce. The key innovation: **target-aware attention** — the candidate item itself determines which history items to focus on.

### Architecture

```
Candidate item: "Vitamin C Serum"
                      ↓
User history:   [Moisturizer, USB Cable, Sunscreen, Book, Eye Cream]
                      ↓
                ┌─────────────────────────────────┐
                │  Target-Aware Attention          │
                │                                  │
                │  For each history item:          │
                │    score = f(history_i, target)  │
                │                                  │
                │  Moisturizer → 0.35 (skincare)   │
                │  USB Cable   → 0.02 (irrelevant) │
                │  Sunscreen   → 0.30 (skincare)   │
                │  Book        → 0.03 (irrelevant) │
                │  Eye Cream   → 0.30 (skincare)   │
                └─────────────────────────────────┘
                      ↓
                User interest = weighted sum
                      ↓
                concat(user_interest, target_emb, user_features)
                      ↓
                    MLP → click probability
```

### Key Difference from Others

```
SASRec/BERT4Rec: user representation is FIXED for all candidates
  → same embedding regardless of which item you're scoring

DIN: user representation CHANGES per candidate item
  → when scoring a skincare product, focus on skincare history
  → when scoring a tech gadget, focus on tech history
```

### Attention Function in DIN

```python
class DINAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, target, history, mask=None):
        # target:  [batch, emb_dim]
        # history: [batch, seq_len, emb_dim]
        seq_len = history.size(1)
        target = target.unsqueeze(1).expand_as(history)

        # Interaction features
        concat = torch.cat([
            target, history,
            target - history,          # difference
            target * history,          # element-wise product
        ], dim=-1)  # [batch, seq_len, emb_dim * 4]

        scores = self.mlp(concat).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), history).squeeze(1)
        return output  # [batch, emb_dim]
```

---

## 6. Model Comparison

| Aspect | SASRec | BERT4Rec | NARM | DIN |
|--------|--------|----------|------|-----|
| **Input** | Full sequence | Full sequence (masked) | Session only | History + target |
| **Attention** | Causal self-attn | Bidirectional self-attn | Query-based over GRU | Target-aware |
| **Architecture** | Transformer | Transformer | GRU + Attention | MLP + Attention |
| **Task** | Next-item | Masked-item | Next-click (session) | CTR prediction |
| **User repr** | Fixed per sequence | Fixed per sequence | Fixed per session | Varies per candidate |
| **Strengths** | Simple, strong | Rich context | No long history needed | Adaptive interests |
| **Best for** | Sequential rec | Sequential rec | Session-based rec | Ranking/CTR |
| **Complexity** | Medium | Medium | Low | Low-Medium |

### When to Use What

```
"I have full user purchase history, ordered by time"
  → SASRec or BERT4Rec

"I only have the current browsing session"
  → NARM

"I need to rank candidates for a specific user"
  → DIN (target-aware attention adapts per candidate)

"I want the simplest strong baseline"
  → SASRec (2 layers, 2 heads is often enough)
```

---

## 7. The Key Insight Across All Models

All four models share one insight: **not all items in a user's history are equally relevant, and the relevance depends on context**.

```
Static approaches:
  User embedding = f(all interactions)
  → Same representation regardless of what you're predicting

Attention-based approaches:
  User embedding = f(interactions, context)
  → "Context" = candidate item (DIN), position in sequence (SASRec),
    or session purpose (NARM)
```

This is exactly analogous to how GNNs propagate information through neighborhoods — but attention models work on **sequences** rather than **graphs**.

```
GNN view:                       Attention view:
  User → neighbors              User → history sequence
  (graph structure)              (temporal structure)
  Message passing                Self-attention
  → neighbor-aware embedding     → context-aware embedding
```

The question for your BTP: **can we combine both views?** (See 03-combined-approaches/)

---

## 8. Practical Notes

### Sequence Truncation

Most users have hundreds of interactions. Attention models typically truncate to the **last k items** (k=50 is common):

```python
max_len = 50
if len(user_history) > max_len:
    user_history = user_history[-max_len:]  # keep most recent
else:
    user_history = [0] * (max_len - len(user_history)) + user_history  # left-pad
```

### Item Embedding Sharing

SASRec and BERT4Rec share the item embedding table between input and output:

```
Input:  item_id → embedding table → input embedding
Output: last hidden state → dot product with SAME embedding table → scores

This reduces parameters and provides a natural similarity metric.
```

### Loss Functions

```
SASRec:   Binary cross-entropy (1 positive + 1 negative per position)
BERT4Rec: Softmax cross-entropy over all items (or sampled softmax)
NARM:     Cross-entropy over all items
DIN:      Binary cross-entropy (click vs no-click)
```

---

## 9. Exercises

1. **Implement SASRec.** Build a 2-layer, 2-head SASRec with d_model=64 on Amazon-Beauty. Train with BPR-style loss (1 positive + 1 negative sample per position). Report Recall@10 and NDCG@10.

2. **Visualize attention.** For a trained SASRec model, extract the attention weights for a user's sequence. Which items attend to which? Do skincare products attend to other skincare products?

3. **SASRec vs BERT4Rec.** Implement both on the same dataset. Compare: (a) training time per epoch, (b) final Recall@10, (c) attention weight patterns.

4. **DIN attention.** Implement target-aware attention. For the same user, visualize how attention weights change when the target item is (a) a skincare product vs (b) an electronics product.

---

## Key Takeaways

- **SASRec** is the go-to sequential recommender — causal Transformer, simple and strong
- **BERT4Rec** uses bidirectional attention with masked item prediction — richer context but more complex
- **NARM** combines GRU with attention for session-based recommendation
- **DIN** pioneered target-aware attention — user representation adapts per candidate item
- All models share the insight: **context-dependent attention over user history beats static aggregation**
- The natural next question: can graph structure (GNN) and sequential attention (Transformer) be combined?
