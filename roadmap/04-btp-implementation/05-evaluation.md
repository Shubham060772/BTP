# Phase 11 — Evaluation Metrics (Week 7–8)

## Why This Phase Matters

> "If you can't measure it, you can't improve it."

Evaluation is one of the **most important** parts of your thesis. A bad model with proper evaluation is still a useful contribution. A good model with poor evaluation is unconvincing. This phase teaches you exactly how to measure recommendation quality.

---

## 1. Why Accuracy and MSE Aren't Enough

### Rating Prediction ≠ Good Recommendations

```
Model A: predicts ratings with MSE = 0.72
  Predicted scores: [4.1, 3.9, 3.8, 2.1, 1.5]
  True relevant:            item at position 3
  → Relevant item is ranked 3rd ✓

Model B: predicts ratings with MSE = 1.30
  Predicted scores: [3.0, 2.5, 4.8, 1.2, 0.5]
  True relevant:            item at position 3
  → Relevant item is ranked 1st ✓✓✓

Model B has WORSE MSE but BETTER ranking!
The user sees a ranked list, not predicted scores.
```

### The Fundamental Point

Recommendation is a **ranking** problem. We care about:

1. Are the relevant items in the top-K? → **Recall@K**
2. Are relevant items ranked as HIGH as possible within top-K? → **NDCG@K**

---

## 2. Setup: The Evaluation Scenario

For each user $u$ in the test set:

```
1. Model produces scores for ALL items
2. Mask items the user already saw in training
3. Rank remaining items by predicted score
4. Take the top K items
5. Compare with ground truth (items in test set)

Example:
  User U's training items: {A, B, C}
  User U's test items:     {D, H}       ← ground truth

  Model scores (after masking A, B, C):
    D: 4.2,  E: 3.8,  F: 3.5,  G: 3.1,  H: 2.9,  I: 2.5, ...

  Top-5 recommendation: [D, E, F, G, H]

  Ground truth in top-5: {D, H} → both found!
```

---

## 3. Precision@K

**Question:** Of the K items recommended, how many are relevant?

$$
\text{Precision@K} = \frac{|\text{recommended@K} \cap \text{relevant}|}{K}
$$

### Worked Example

```
K = 5
Recommended:  [D, E, F, G, H]
Relevant:     {D, H}

Hits:         [1, 0, 0, 0, 1]

Precision@5 = 2/5 = 0.40
```

```
Interpretation:
  40% of our recommendations were relevant.
  This seems low, but in recommendation K is usually
  much larger than the number of relevant items.

  If user has 2 relevant items and K=10:
    Best possible Precision@10 = 2/10 = 0.20
    → Precision@K can be misleading when |relevant| << K
```

### Limitation

Precision@K penalizes recommendations when there are **few** relevant items. If a user has only 1 relevant item, Precision@10 can never exceed 0.1 even with a perfect model. This is why **Recall@K** is preferred in recommendation research.

---

## 4. Recall@K

**Question:** Of all relevant items, how many did we recommend in the top-K?

$$
\text{Recall@K} = \frac{|\text{recommended@K} \cap \text{relevant}|}{|\text{relevant}|}
$$

### Worked Example

```
K = 5
Recommended:  [D, E, F, G, H]
Relevant:     {D, H}

Hits:         [1, 0, 0, 0, 1]

Recall@5 = 2/2 = 1.00

→ We found ALL relevant items in the top 5. Perfect recall!
```

### Another Example

```
K = 5
Recommended:  [E, F, D, G, I]
Relevant:     {D, H, J}

Hits:         [0, 0, 1, 0, 0]

Recall@5 = 1/3 = 0.333

→ We only found 1 out of 3 relevant items.
```

### Why Recall@K Is Preferred

```
Recall answers: "Did we find the relevant items?"
  → This is what matters to users.

If there are 3 relevant items out of 1000:
  Recall@10 can reach 1.0 (find all 3 in top 10)
  Precision@10 maxes out at 0.3 (3/10)

Recall is not penalized by the K/relevant ratio.
```

---

## 5. Hit Rate@K (HR@K)

**Question:** Did we recommend **at least one** relevant item in the top-K?

$$
\text{HR@K} = \begin{cases} 1 & \text{if } |\text{recommended@K} \cap \text{relevant}| > 0 \\ 0 & \text{otherwise} \end{cases}
$$

This is equivalent to Recall@K when each user has exactly **one** test item (leave-one-out evaluation).

```
User 1: recommended [A,B,C,D,E], test item = C → HR@5 = 1
User 2: recommended [A,B,D,E,F], test item = C → HR@5 = 0
User 3: recommended [C,A,B,D,E], test item = C → HR@5 = 1

Average HR@5 = (1 + 0 + 1) / 3 = 0.667
```

---

## 6. NDCG@K — Normalized Discounted Cumulative Gain

This is the **most important metric** for your BTP. While Recall@K only checks IF a relevant item appears in top-K, NDCG@K also checks WHERE it appears.

### 6.1 Intuition

```
Two recommendations, both with Recall@5 = 1.0:

Recommendation A:  [relevant, -, -, -, -]
  → Relevant item at position 1. Excellent!

Recommendation B:  [-, -, -, -, relevant]
  → Relevant item at position 5. Worse, user had to scroll.

Recall@5 says both are equal (1/1 = 1.0).
NDCG@5 says A is better than B.
```

### 6.2 Relevance Scores

For binary relevance (implicit feedback):

$$
\text{rel}_i = \begin{cases} 1 & \text{if item at position } i \text{ is relevant} \\ 0 & \text{otherwise} \end{cases}
$$

### 6.3 DCG — Discounted Cumulative Gain

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}
$$

The $\frac{1}{\log_2(i+1)}$ is the **discount** — items at lower positions contribute less.

```
Position i:    1      2      3      4      5      6      7      8      9     10
Discount:     1.00   0.63   0.50   0.43   0.39   0.36   0.33   0.31   0.29  0.28

Position 1 gets full credit (discount = 1.0)
Position 2 gets 63% credit
Position 5 gets 39% credit
Position 10 gets 28% credit
```

### 6.4 IDCG — Ideal DCG

The **best possible** DCG, achieved when all relevant items are ranked at the top.

$$
\text{IDCG@K} = \sum_{i=1}^{\min(K, |\text{relevant}|)} \frac{1}{\log_2(i+1)}
$$

### 6.5 NDCG = DCG / IDCG

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

This normalizes DCG to [0, 1], where 1.0 is perfect ranking.

### 6.6 Complete Worked Example

```
K = 5
Recommended:  [E, D, F, H, G]
Relevant:     {D, H}

Step 1: Compute relevance at each position
  Position 1 (E): rel = 0
  Position 2 (D): rel = 1  ← relevant!
  Position 3 (F): rel = 0
  Position 4 (H): rel = 1  ← relevant!
  Position 5 (G): rel = 0

Step 2: Compute DCG@5
  DCG = 0/log₂(2) + 1/log₂(3) + 0/log₂(4) + 1/log₂(5) + 0/log₂(6)
      = 0 + 0.6309 + 0 + 0.4307 + 0
      = 1.0616

Step 3: Compute IDCG@5
  Ideal ranking: [relevant, relevant, -, -, -]
  IDCG = 1/log₂(2) + 1/log₂(3)
       = 1.0 + 0.6309
       = 1.6309

Step 4: NDCG@5
  NDCG = 1.0616 / 1.6309 = 0.6509
```

### 6.7 Another Example: Compare Two Rankings

```
Relevant items: {A, C}

Ranking 1: [A, B, C, D, E]
  DCG  = 1/log₂(2) + 0 + 1/log₂(4) + 0 + 0
       = 1.0 + 0.5 = 1.5
  IDCG = 1.0 + 0.6309 = 1.6309
  NDCG = 1.5 / 1.6309 = 0.9197

Ranking 2: [B, D, A, E, C]
  DCG  = 0 + 0 + 1/log₂(4) + 0 + 1/log₂(6)
       = 0.5 + 0.3869 = 0.8869
  IDCG = 1.6309
  NDCG = 0.8869 / 1.6309 = 0.5438

Ranking 1 (NDCG = 0.92) >> Ranking 2 (NDCG = 0.54)

Both have Recall@5 = 1.0, but Ranking 1 puts relevant
items at positions 1 and 3, while Ranking 2 puts them
at positions 3 and 5. NDCG captures this difference.
```

---

## 7. MAP — Mean Average Precision

### Average Precision for One User

$$
\text{AP@K} = \frac{1}{|\text{relevant}|} \sum_{i=1}^{K} \text{Precision@}i \times \text{rel}_i
$$

### Worked Example

```
K = 5
Recommended:  [D, E, H, F, G]
Relevant:     {D, H}

Position 1 (D): rel=1, Precision@1 = 1/1 = 1.000
Position 2 (E): rel=0, skip
Position 3 (H): rel=1, Precision@3 = 2/3 = 0.667
Position 4 (F): rel=0, skip
Position 5 (G): rel=0, skip

AP@5 = (1.000 + 0.667) / 2 = 0.833
```

MAP@K is the mean of AP@K across all users:

$$
\text{MAP@K} = \frac{1}{|U|} \sum_{u \in U} \text{AP@K}(u)
$$

### MAP vs NDCG

```
Both reward higher-ranked relevant items.

MAP: uses Precision@i as the position-dependent weight
NDCG: uses 1/log₂(i+1) as the discount factor

In practice, NDCG@K is more common in recommendation research.
Use NDCG@K as your primary metric and MAP as supplementary.
```

---

## 8. Implementation

### 8.1 All Metrics in Python

```python
import numpy as np

def recall_at_k(recommended, relevant, k):
    """
    recommended: list of item indices (ordered by score)
    relevant: set of relevant item indices
    k: cutoff
    """
    topk = set(recommended[:k])
    if len(relevant) == 0:
        return 0.0
    return len(topk & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Compute NDCG@K"""
    topk = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(topk):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

    # IDCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(recommended, relevant, k):
    """Compute Precision@K"""
    topk = set(recommended[:k])
    if k == 0:
        return 0.0
    return len(topk & relevant) / k


def hit_rate_at_k(recommended, relevant, k):
    """Compute HR@K"""
    topk = set(recommended[:k])
    return 1.0 if len(topk & relevant) > 0 else 0.0


def ap_at_k(recommended, relevant, k):
    """Compute Average Precision@K"""
    topk = recommended[:k]
    if len(relevant) == 0:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(topk):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(relevant)
```

### 8.2 Full Evaluation Pipeline

```python
def evaluate_model(model, data, train_user_items, test_user_items,
                   num_items, k_values=[5, 10, 20]):
    """Evaluate a model on all metrics at multiple K values"""
    model.eval()

    with torch.no_grad():
        # Get embeddings
        user_emb, item_emb = model(data.edge_index)

        results = {k: {'recall': [], 'ndcg': [], 'precision': [],
                        'hr': [], 'ap': []}
                   for k in k_values}

        for user in test_user_items:
            if len(test_user_items[user]) == 0:
                continue

            # Score all items
            scores = user_emb[user] @ item_emb.T

            # Mask training items
            train_items = list(train_user_items.get(user, set()))
            scores[train_items] = -float('inf')

            # Rank items
            _, ranked_items = torch.sort(scores, descending=True)
            ranked_items = ranked_items.tolist()

            relevant = test_user_items[user]

            for k in k_values:
                results[k]['recall'].append(recall_at_k(ranked_items, relevant, k))
                results[k]['ndcg'].append(ndcg_at_k(ranked_items, relevant, k))
                results[k]['precision'].append(precision_at_k(ranked_items, relevant, k))
                results[k]['hr'].append(hit_rate_at_k(ranked_items, relevant, k))
                results[k]['ap'].append(ap_at_k(ranked_items, relevant, k))

    # Average across users
    final = {}
    for k in k_values:
        final[k] = {
            metric: np.mean(values) for metric, values in results[k].items()
        }
    return final
```

### 8.3 Pretty-Printing Results

```python
def print_results(model_results):
    """
    model_results: dict of {model_name: {k: {metric: value}}}
    """
    k_values = sorted(next(iter(model_results.values())).keys())

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"K = {k}")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Recall':<10} {'NDCG':<10} {'Prec':<10} {'HR':<10}")
        print(f"{'-'*55}")
        for model_name, results in model_results.items():
            r = results[k]
            print(f"{model_name:<15} {r['recall']:<10.4f} {r['ndcg']:<10.4f} "
                  f"{r['precision']:<10.4f} {r['hr']:<10.4f}")
```

---

## 9. Creating the Comparison Table

Your BTP should include a table like this:

```
Table 1: Recommendation performance on Amazon-Book

K = 10
┌──────────────┬────────────┬────────────┬────────────┬────────────┐
│ Model        │ Recall@10  │ NDCG@10    │ Prec@10    │ HR@10      │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Popularity   │ 0.0523     │ 0.0312     │ 0.0284     │ 0.2156     │
│ MF           │ 0.0834     │ 0.0621     │ 0.0451     │ 0.3542     │
│ GCN          │ 0.0912     │ 0.0698     │ 0.0487     │ 0.3821     │
│ LightGCN     │ 0.1103     │ 0.0841     │ 0.0562     │ 0.4312     │
└──────────────┴────────────┴────────────┴────────────┴────────────┘

(Values are illustrative — your actual numbers will differ)

Key observations:
1. LightGCN outperforms all baselines on every metric
2. GCN improves over MF → graph structure helps
3. LightGCN improves over GCN → simpler is better for CF
4. All models beat Popularity → personalization matters
```

Report **at least** Recall@K and NDCG@K. These are the standard metrics in recommendation papers.

---

## 10. Statistical Significance

For a strong thesis, mention whether differences are **statistically significant**.

```python
from scipy import stats

# Compare two models' per-user NDCG scores
model_a_ndcg = [...]  # NDCG@10 for each user, model A
model_b_ndcg = [...]  # NDCG@10 for each user, model B

t_stat, p_value = stats.ttest_rel(model_a_ndcg, model_b_ndcg)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Difference is statistically significant")
```

You don't need deep statistics knowledge. Just:
1. Save per-user metric scores (not just averages)
2. Run a paired t-test between models
3. Report whether $p < 0.05$

---

## 11. Common Evaluation Mistakes

### Mistake 1: Not Masking Training Items

```
BAD:  Score all items, take top-K
      → Model recommends items the user ALREADY interacted with
      → Inflated metrics (easy to "predict" known interactions)

GOOD: Mask training items BEFORE ranking
      → Only score unseen items
```

### Mistake 2: Using the Full Item Set for All Users

```
BAD:  Score 100K items for every user
      → Very slow evaluation

ACCEPTABLE: Score all items (if feasible)
      → Most accurate but slow

ALTERNATIVE: Sample 100 negatives per user + their positives
      → Faster but introduces sampling bias
      → If you do this, be explicit about it in your thesis
```

### Mistake 3: Averaging Metrics Over Items Instead of Users

```
BAD:  avg_ndcg = sum(all item ndcgs) / num_items

GOOD: avg_ndcg = sum(per_user_ndcgs) / num_users
      → Each USER contributes equally
```

### Mistake 4: Not Using Validation for Hyperparameter Tuning

```
BAD:  Tune hyperparameters on test set → overfitting to test

GOOD: Tune on validation set → report final numbers on test set

Split: Train (80%) → Val (10%) → Test (10%)
  1. Train models with different hyperparameters
  2. Pick best hyperparameters based on Val metrics
  3. Report final performance on Test set (ONE time)
```

### Mistake 5: Comparing Models with Different Embedding Dimensions

```
BAD:
  MF with embed_dim=128  vs  LightGCN with embed_dim=64
  → MF has more parameters, unfair comparison

GOOD:
  MF with embed_dim=64   vs  LightGCN with embed_dim=64
  → Same embedding dimension, fair comparison
```

---

## Exercises

1. **By hand:** Given `recommended = [C, A, D, B, E]` and `relevant = {A, B}`, compute Recall@5, NDCG@5, Precision@5, HR@5, and AP@5 by hand. Show all steps.

2. **Edge cases:** What happens to NDCG@K when (a) there are no relevant items? (b) all recommended items are relevant? (c) there's exactly 1 relevant item at position K?

3. **Implementation:** Implement all five metrics from scratch. Verify them on the example from Exercise 1.

4. **Sensitivity analysis:** How does K affect each metric? Plot Recall@K and NDCG@K for K = 1, 5, 10, 20, 50, 100 for your trained LightGCN model.

5. **Statistical testing:** Train LightGCN with 3 different random seeds. Compute per-user NDCG@10 for each run. Are the differences between runs statistically significant? What does this tell you about reporting results with a single run?

---

## Key Takeaways

1. **Recommendation is a ranking problem.** Don't use MSE or accuracy. Use ranking metrics: Recall@K and NDCG@K.

2. **Recall@K** measures whether relevant items appear in the top-K. It doesn't care about the position within the top-K.

3. **NDCG@K** measures whether relevant items are ranked as HIGH as possible within the top-K. It gives more credit to relevant items at position 1 than position K. **This should be your primary metric.**

4. **Always mask training items** before computing metrics, or your results will be inflated and meaningless.

5. **Use validation for tuning, test for final reporting.** Never tune hyperparameters on the test set.

6. **Report a table** with Recall@K and NDCG@K for all models at the same K values. This is the core result of your BTP.

7. **Mention statistical significance** if possible — it strengthens your thesis considerably.
