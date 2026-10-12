# Phase 9 — BPR Loss and Negative Sampling (Week 6)

## Why This Phase Matters

You've built the model architecture (LightGCN). Now you need to **train** it. The loss function determines what the model learns, and for recommendation, the standard classification or regression losses don't work well. BPR (Bayesian Personalized Ranking) is the loss function you'll use throughout your BTP.

---

## 1. The Problem with Pointwise Prediction

### 1.1 Why Not MSE on Ratings?

The naive approach to recommendation training:

```
Given: User u rated Item i with rating 4.5
Goal:  Predict ŷ_ui as close to 4.5 as possible
Loss:  MSE = (ŷ_ui - 4.5)²
```

This has several problems:

**Problem 1: Most real data is implicit, not explicit.**

```
Explicit feedback:           Implicit feedback:
  User rated product 4/5       User purchased product
  User rated product 2/5       User clicked on product
  (rare, biased)               User added to cart
                                (abundant, natural)

Most real-world systems use implicit feedback.
There's no "rating" to predict — just interactions.
```

**Problem 2: Optimizing for rating accuracy ≠ good recommendations.**

```
Model A: Predicts ratings with MSE = 0.7
  But ranks items poorly → bad recommendations

Model B: Predicts ratings with MSE = 1.2
  But ranks items correctly → good recommendations

You'd rather have Model B.
The user doesn't see the predicted rating — they see the ranking.
```

**Problem 3: Missing data is ambiguous.**

```
            Item A    Item B    Item C
User 1      5.0       ?         3.0

What does "?" mean?
  a) User 1 hasn't seen Item B (no opinion)
  b) User 1 doesn't like Item B (negative)
  c) User 1 would love Item B (just hasn't found it yet)

MSE would need a target value for "?" — but there isn't one.
```

### 1.2 Explicit vs Implicit Feedback

| | Explicit | Implicit |
|---|---|---|
| **Signal** | Ratings (1-5 stars) | Clicks, views, purchases |
| **Abundance** | Sparse (few users rate) | Dense (most users interact) |
| **Reliability** | Moderate (rating bias) | Low (clicks ≠ preference) |
| **Missing data** | Missing = unknown | Missing = likely negative |
| **For your BTP** | Can convert to implicit | Natural fit for BPR |

For most modern recommendation systems (and your BTP), you'll work with **implicit feedback**: the fact that a user interacted with an item, regardless of rating.

```
Convert explicit → implicit:
  Rating ≥ threshold → positive interaction (1)
  No rating          → no interaction (0)

  Original:              Implicit:
  User 1: Product A=5   User 1: Product A=1
  User 1: Product B=2   User 1: Product B=1 (or drop if threshold=3)
  User 1: Product C=?   User 1: Product C=0
```

---

## 2. Why Ranking Matters More Than Rating Prediction

Recommendation is fundamentally a **ranking** problem:

```
What the user sees:

  "Recommended for you:"
  1. Product A      ← position matters
  2. Product B
  3. Product C
  4. Product D
  5. Product E
  ...

The user only sees the top-K items.
They care about ORDER, not about predicted scores.
```

What we actually need:

$$
\hat{y}_{u,\text{relevant}} > \hat{y}_{u,\text{irrelevant}}
$$

We need relevant items to be ranked ABOVE irrelevant ones. We don't need to predict exact scores.

---

## 3. BPR — Bayesian Personalized Ranking

**Paper:** Rendle et al. (2009) — "BPR: Bayesian Personalized Ranking from Implicit Feedback"

### 3.1 The Core Idea

Instead of predicting individual scores, BPR learns **pairwise preferences**:

```
For user u:
  Positive item i: user interacted with this item
  Negative item j: user did NOT interact with this item

BPR's goal:
  The model should score item i HIGHER than item j for user u.

  ŷ_ui > ŷ_uj
```

### 3.2 The BPR Triplet

Each training sample is a triplet $(u, i, j)$:

```
(user, positive_item, negative_item)

Example:
  User Alice bought "CeraVe Moisturizer" but not "Random Shampoo"

  Triplet: (Alice, CeraVe Moisturizer, Random Shampoo)
  Goal:    score(Alice, CeraVe Moisturizer) > score(Alice, Random Shampoo)
```

### 3.3 The BPR Loss Formula

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D} \ln\,\sigma\!\left(\hat{y}_{ui} - \hat{y}_{uj}\right)
$$

where:
- $(u, i, j)$: user, positive item, negative item
- $\hat{y}_{ui}$: predicted score for user $u$ and positive item $i$
- $\hat{y}_{uj}$: predicted score for user $u$ and negative item $j$
- $\sigma$: sigmoid function $\sigma(x) = \frac{1}{1+e^{-x}}$
- $D$: set of all training triplets

Let's break this down piece by piece:

```
Step 1: Compute the score difference
  x_uij = ŷ_ui - ŷ_uj

  If x_uij > 0: model correctly ranks i above j ✓
  If x_uij < 0: model incorrectly ranks j above i ✗

Step 2: Apply sigmoid
  σ(x_uij) = 1 / (1 + exp(-x_uij))

  Maps the difference to (0, 1):
    x_uij >> 0  →  σ ≈ 1  (very confident correct ranking)
    x_uij = 0   →  σ = 0.5 (uncertain)
    x_uij << 0  →  σ ≈ 0  (wrong ranking)

Step 3: Apply -log
  -log(σ(x_uij))

  Converts to a loss:
    σ ≈ 1  →  -log(1) ≈ 0     (low loss, good)
    σ = 0.5 →  -log(0.5) ≈ 0.69
    σ ≈ 0  →  -log(0) → ∞    (high loss, bad)
```

### 3.4 Derivation Intuition

The formal derivation comes from a Bayesian framework, but the intuition is:

```
Assumption:
  For user u, if they interacted with item i but not item j,
  then u PREFERS i over j.

  P(i >_u j) = σ(ŷ_ui - ŷ_uj)

We want to MAXIMIZE this probability for all observed preferences.

Maximizing probability = Minimizing negative log-likelihood:
  L = -Σ log P(i >_u j)
    = -Σ log σ(ŷ_ui - ŷ_uj)

That's the BPR loss.
```

### 3.5 Why Log-Sigmoid Works

The log-sigmoid loss has nice properties:

```
Loss landscape for different score differences:

  -log(σ(x))
  │
  │\
  │ \
  │  \
  │   \.
  │     '·.
  │         '·····───────
  ├─────────────────────── x = ŷ_ui - ŷ_uj
  -3  -2  -1   0   1   2   3

Key properties:
  • Smooth everywhere (good for gradient descent)
  • High penalty when ŷ_uj > ŷ_ui (wrong order)
  • Low penalty when ŷ_ui >> ŷ_uj (correct with large margin)
  • Gradient is strongest when the model is uncertain (x ≈ 0)
```

### 3.6 Gradient of BPR Loss

The gradient of BPR loss with respect to model parameters $\theta$:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\sum_{(u,i,j)} \frac{e^{-(\hat{y}_{ui}-\hat{y}_{uj})}}{1 + e^{-(\hat{y}_{ui}-\hat{y}_{uj})}} \cdot \frac{\partial(\hat{y}_{ui}-\hat{y}_{uj})}{\partial \theta}
$$

Simplified:

$$
= -\sum_{(u,i,j)} \left(1 - \sigma(\hat{y}_{ui}-\hat{y}_{uj})\right) \cdot \frac{\partial(\hat{y}_{ui}-\hat{y}_{uj})}{\partial \theta}
$$

```
Interpretation:
  When σ(ŷ_ui - ŷ_uj) ≈ 1: gradient ≈ 0 (already correct, no update)
  When σ(ŷ_ui - ŷ_uj) ≈ 0: gradient ≈ -1 (wrong ranking, big update)
  When σ(ŷ_ui - ŷ_uj) = 0.5: gradient ≈ -0.5 (uncertain, medium update)
```

---

## 4. Negative Sampling Strategies

BPR requires negative items — items the user **didn't** interact with. How you sample them significantly affects training.

### 4.1 Uniform Random Sampling

```python
def sample_negative_uniform(user, num_items, positive_items):
    """Sample a random item that the user hasn't interacted with"""
    while True:
        neg_item = random.randint(0, num_items - 1)
        if neg_item not in positive_items[user]:
            return neg_item
```

```
Pros:
  ✓ Simple to implement
  ✓ Fast
  ✓ Unbiased

Cons:
  ✗ Most negatives are "easy" — obviously irrelevant items
  ✗ Slow convergence (model learns nothing from easy negatives)

Example for User Alice (loves skincare):
  Random negative: "Dog food"             → too easy, no learning signal
  Random negative: "Camping tent"         → too easy, no learning signal
  Random negative: "Neutrogena Serum"     → useful! Hard negative
```

### 4.2 Popularity-Based Sampling

Sample negative items proportional to their popularity.

```python
def sample_negative_popularity(user, item_popularity, positive_items):
    """Sample negatives proportional to item popularity"""
    while True:
        neg_item = np.random.choice(
            len(item_popularity),
            p=item_popularity / item_popularity.sum()
        )
        if neg_item not in positive_items[user]:
            return neg_item
```

```
Intuition:
  Popular items are harder negatives because they're
  "good" items that the user specifically CHOSE not to interact with.

  A popular bestseller is a hard negative for a user who
  bought many similar products but skipped this one.

  A random obscure product is an easy negative —
  the user probably just never saw it.
```

### 4.3 Hard Negative Mining

Sample items that the model currently ranks HIGH but are actually negative.

```python
def sample_hard_negative(user, model, positive_items, k=100):
    """Sample from items the model currently thinks the user would like"""
    with torch.no_grad():
        scores = model.predict(user, all_items)
        # Get top-k scoring items that are NOT positive
        candidates = []
        for item in scores.argsort(descending=True):
            if item not in positive_items[user]:
                candidates.append(item)
            if len(candidates) >= k:
                break
        return random.choice(candidates)
```

```
Intuition:
  Hard negatives are items that "fool" the model — the model
  thinks the user would like them, but they're actually negative.

  These provide the strongest learning signal.

  But be careful: too-hard negatives can be false negatives
  (items the user would actually like but hasn't seen yet).
```

### 4.4 Strategy Comparison

```
Strategy          | Speed    | Quality  | Risk
──────────────────|──────────|──────────|────────────
Uniform random    | Fastest  | Low      | None
Popularity-based  | Fast     | Medium   | Low
Hard negative     | Slow     | High     | False negatives

For your BTP:
  Start with uniform random (it works well enough).
  Then try popularity-based for comparison.
  Hard negative mining is a possible experiment.
```

---

## 5. Implementation in PyTorch

### 5.1 Complete BPR Training Step

```python
import torch
import torch.nn.functional as F
import random

class BPRLoss(torch.nn.Module):
    def __init__(self, reg_weight=1e-4):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, pos_scores, neg_scores, *reg_embeddings):
        """
        pos_scores: [batch_size] — scores for positive items
        neg_scores: [batch_size] — scores for negative items
        reg_embeddings: embedding tensors to regularize
        """
        # BPR loss
        bpr_loss = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-10
        ).mean()

        # L2 regularization
        reg_loss = 0
        for emb in reg_embeddings:
            reg_loss += emb.norm(2).pow(2)
        reg_loss = self.reg_weight * reg_loss / pos_scores.size(0)

        return bpr_loss + reg_loss
```

### 5.2 Triplet Sampling

```python
def sample_bpr_triplets(edge_index, num_users, num_items, batch_size):
    """Sample (user, positive_item, negative_item) triplets"""
    # Build user → positive items mapping
    user_pos = {}
    src, dst = edge_index
    for u, i in zip(src.tolist(), dst.tolist()):
        if u < num_users and i >= num_users:
            user_pos.setdefault(u, set()).add(i - num_users)

    users = []
    pos_items = []
    neg_items = []

    user_list = list(user_pos.keys())

    for _ in range(batch_size):
        # Random user
        u = random.choice(user_list)

        # Random positive item for this user
        pos = random.choice(list(user_pos[u]))

        # Random negative item (not in user's positives)
        neg = random.randint(0, num_items - 1)
        while neg in user_pos[u]:
            neg = random.randint(0, num_items - 1)

        users.append(u)
        pos_items.append(pos)
        neg_items.append(neg)

    return (torch.tensor(users),
            torch.tensor(pos_items),
            torch.tensor(neg_items))
```

### 5.3 Full Training Loop

```python
model = LightGCN(num_users, num_items, embed_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bpr_loss_fn = BPRLoss(reg_weight=1e-4)

for epoch in range(100):
    model.train()

    # Forward pass: get all embeddings
    user_emb, item_emb = model(data.edge_index)

    # Sample triplets
    users, pos_items, neg_items = sample_bpr_triplets(
        data.edge_index, num_users, num_items, batch_size=2048
    )

    # Compute scores
    pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
    neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)

    # Compute loss
    loss = bpr_loss_fn(
        pos_scores, neg_scores,
        model.user_embedding(users),      # for regularization
        model.item_embedding(pos_items),
        model.item_embedding(neg_items),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

---

## 6. Common Pitfalls

### Pitfall 1: Sampling a Positive as Negative

```
User 1 interacted with: [Item 3, Item 7, Item 12]
Negative sample: Item 7  ← WRONG! This is actually positive!

Solution: Always check that the sampled negative is NOT
in the user's positive set.
```

### Pitfall 2: Numerical Instability

```python
# BAD: can produce log(0) = -inf
loss = -torch.log(torch.sigmoid(pos - neg)).mean()

# GOOD: add small epsilon
loss = -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()

# BETTER: use logsigmoid directly (numerically stable)
loss = -F.logsigmoid(pos - neg).mean()
```

### Pitfall 3: Not Enough Negatives Per Epoch

```
Dataset: 100K interactions
Batch size: 1024
Batches per epoch: ~100

Each batch samples 1024 negatives.
Total negatives per epoch: ~100K

But there could be millions of possible negatives.
The model only sees a tiny fraction each epoch.

Solution: Train for many epochs so the model eventually
sees diverse negatives.
```

### Pitfall 4: Forgetting Regularization

```
Without regularization:
  The model can make embeddings arbitrarily large
  to maximize the score difference.
  ŷ_ui = 1000, ŷ_uj = -1000 → loss ≈ 0 but overfitting

With L2 regularization:
  Embeddings are kept small → better generalization
```

---

## 7. Why BPR Is Appropriate for Implicit Feedback

```
Implicit feedback:
  ✓ User purchased product   → positive signal
  ✗ User didn't buy product  → unknown (not necessarily negative)

BPR's assumption:
  "If user u interacted with item i but not item j,
   then u PROBABLY prefers i over j."

This is a WEAKER assumption than:
  "User u dislikes item j" (pointwise negative)

BPR only claims relative preference, not absolute dislike.
This matches the nature of implicit feedback better.

  Pointwise:  ŷ_ui → 1, ŷ_uj → 0  (claims j is negative)
  Pairwise:   ŷ_ui > ŷ_uj           (only claims relative order)
```

---

## 8. BPR vs Other Loss Functions

| Loss | Type | Formula | Best For |
|---|---|---|---|
| **MSE** | Pointwise | $(y - \hat{y})^2$ | Explicit ratings |
| **BCE** | Pointwise | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Click prediction |
| **BPR** | Pairwise | $-\log\sigma(\hat{y}_i - \hat{y}_j)$ | Ranking with implicit feedback |
| **Hinge** | Pairwise | $\max(0, m - \hat{y}_i + \hat{y}_j)$ | Ranking (margin-based) |

For your BTP with implicit feedback and ranking evaluation (Recall@K, NDCG@K), **BPR is the standard choice**.

---

## Exercises

1. **By hand:** Given the scores $\hat{y}_{u,A}=2.5$, $\hat{y}_{u,B}=1.8$, $\hat{y}_{u,C}=3.1$, where A is positive and B, C are negative, compute the BPR loss for triplets $(u,A,B)$ and $(u,A,C)$. Which triplet contributes more to the loss? Why?

2. **Implementation:** Implement `BPRLoss` and `sample_bpr_triplets` from scratch. Test on a synthetic dataset with 100 users, 50 items, and random interactions.

3. **Experiment:** Compare uniform random sampling vs popularity-based sampling on Amazon-Book. Train LightGCN with each strategy for 50 epochs and compare Recall@10.

4. **Analysis:** Why does hard negative mining risk creating false negatives? Give a concrete example with products.

5. **Numerical stability:** Implement the BPR loss three ways — naive `log(sigmoid(...))`, with epsilon, and using `F.logsigmoid`. Create a test case where the naive version produces `nan` or `inf` but the stable versions don't.

---

## Key Takeaways

1. **Recommendation is a ranking problem**, not a rating-prediction problem. BPR optimizes for ranking directly.

2. **BPR uses pairwise comparisons:** for each user, it learns that positive items should score higher than negative items. The loss is $-\log\sigma(\hat{y}_{ui} - \hat{y}_{uj})$.

3. **Negative sampling matters.** Uniform random is the simplest and a good starting point. Popularity-based and hard negative mining are experiments you can run in your BTP.

4. **Always add L2 regularization** on embeddings to prevent overfitting.

5. **Use `F.logsigmoid`** in PyTorch for numerical stability instead of `torch.log(torch.sigmoid(...))`.

6. **BPR makes a weaker assumption** than pointwise losses — it only claims relative preference, not absolute like/dislike — which better matches the nature of implicit feedback.
