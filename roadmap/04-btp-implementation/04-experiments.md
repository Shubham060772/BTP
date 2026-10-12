# Phase 14 — Experimental Design (Week 9–10)

> **Goal:** Design rigorous, reproducible experiments that produce the results tables for your thesis.

---

## 1. Experimental Rigor in a Thesis

Your thesis is judged by the quality of your experiments, not the complexity of your code. A simple model with thorough experiments beats a complex model evaluated carelessly.

```
Careless experiment:                 Rigorous experiment:

- One run, one metric               - Multiple runs, mean ± std
- "I tuned it until it worked"      - Documented hyperparameters
- Different settings per model      - Same settings for fair comparison
- No ablation                       - Ablation isolates each component
- Only reports best numbers         - Reports all conditions
```

---

## 2. Experiment 1 — Model Comparison

This is your **most important table**. It directly answers: "Does the GNN approach work better?"

### Setup

| Setting | Value |
|---------|-------|
| Dataset | Amazon-Book |
| Split | Leave-one-out (temporal) |
| Embedding dim | 64 for all models |
| Training | BPR loss, Adam optimizer |
| Epochs | 500 (early stopping on val NDCG@10, patience=50) |
| Metrics | Recall@10, Recall@20, NDCG@10, NDCG@20 |
| Runs | 5 random seeds, report mean ± std |

### Models

```
Level 0 — Trivial Baseline:
┌──────────────┐
│  Popularity  │   Most-interacted items for all users
└──────────────┘

Level 1 — Classical Baseline:
┌──────────────────────────┐
│  Matrix Factorization    │   R ≈ UV^T with BPR loss
└──────────────────────────┘

Level 2 — Basic GNN:
┌──────────────┐
│     GCN      │   Standard GCN on user-item graph
└──────────────┘

Level 3 — Recommendation GNN:
┌──────────────┐
│  GraphSAGE   │   Sampling-based GNN
├──────────────┤
│    NGCF      │   Neural graph collaborative filtering
├──────────────┤
│  LightGCN    │   Simplified recommendation GNN
└──────────────┘

Level 4 — Your Proposal (if applicable):
┌──────────────────────────┐
│  LightGCN + modification │
└──────────────────────────┘
```

### Expected Results Table

```
Table 1: Model comparison on Amazon-Book

Model           Recall@10    Recall@20    NDCG@10      NDCG@20
─────────────────────────────────────────────────────────────────
Popularity      0.0XXX       0.0XXX       0.0XXX       0.0XXX
MF-BPR          0.0XXX       0.0XXX       0.0XXX       0.0XXX
GCN             0.0XXX       0.0XXX       0.0XXX       0.0XXX
GraphSAGE       0.0XXX       0.0XXX       0.0XXX       0.0XXX
NGCF            0.0XXX       0.0XXX       0.0XXX       0.0XXX
LightGCN        0.0XXX       0.0XXX       0.0XXX       0.0XXX
Yours           0.0XXX       0.0XXX       0.0XXX       0.0XXX

Bold = best result per column.
± standard deviations from 5 runs included.
```

### Fair Comparison Checklist

- [ ] Same dataset and split for all models
- [ ] Same embedding dimension (64)
- [ ] Same optimizer and learning rate (or tuned per model with documented search)
- [ ] Same negative sampling strategy
- [ ] Same evaluation protocol (full ranking or sampled)
- [ ] Same early stopping criterion
- [ ] Same random seeds across runs

---

## 3. Experiment 2 — GNN Layer Depth

This experiment investigates **oversmoothing**: as you add more GNN layers, node representations become increasingly similar, hurting performance.

### The Oversmoothing Problem

```
Layer 0:    Each node has its own unique embedding
               ↓
Layer 1:    Each node aggregates from 1-hop neighbors
               ↓
Layer 2:    Each node aggregates from 2-hop neighborhood
               ↓
Layer 3:    Each node aggregates from 3-hop neighborhood
               ↓
 ...
               ↓
Layer K:    All nodes have nearly identical representations
            (oversmoothed — useless for recommendation)
```

### Setup

Fix everything except the number of layers:

```python
configs = []
for num_layers in [1, 2, 3, 4, 5]:
    configs.append({
        'model': 'LightGCN',
        'num_layers': num_layers,
        'embedding_dim': 64,
        'lr': 1e-3,
        'dataset': 'amazon-book',
    })
```

### Expected Results

```
Table 2: Effect of GNN depth on LightGCN performance

Layers    Recall@10    NDCG@10     Training Time
────────────────────────────────────────────────
1         0.0XXX       0.0XXX      XX min
2         0.0XXX       0.0XXX      XX min
3         0.0XXX       0.0XXX      XX min
4         0.0XXX       0.0XXX      XX min
5         0.0XXX       0.0XXX      XX min
```

### What to Analyze

```
Performance
    ^
    |      *  *
    |    *      *
    |  *          *
    | *              *
    +--+--+--+--+--+--→ Layers
       1  2  3  4  5

Typical finding:
- Performance peaks at 2-3 layers
- Degrades after 3-4 layers (oversmoothing)
- 1 layer is too shallow to capture collaborative signals
```

In your thesis, explain:
- **Why 1 layer is weak:** only direct neighbors contribute — no higher-order collaborative filtering
- **Why 2-3 layers is optimal:** captures the pattern "users who liked similar items also liked X"
- **Why 4+ layers degrade:** embeddings converge, losing discriminative power

---

## 4. Experiment 3 — Embedding Dimension

### Setup

```python
for embed_dim in [16, 32, 64, 128, 256]:
    configs.append({
        'model': 'LightGCN',
        'num_layers': 3,  # fixed at best from Experiment 2
        'embedding_dim': embed_dim,
    })
```

### Expected Results

```
Table 3: Effect of embedding dimension on LightGCN

Embed Dim    Recall@10    NDCG@10    Parameters    Time
──────────────────────────────────────────────────────────
16           0.0XXX       0.0XXX     XXX K         XX min
32           0.0XXX       0.0XXX     XXX K         XX min
64           0.0XXX       0.0XXX     XXX K         XX min
128          0.0XXX       0.0XXX     XXX K         XX min
256          0.0XXX       0.0XXX     XXX K         XX min
```

### What to Analyze

```
Trade-off:

     Underfitting        Sweet spot        Overfitting
     ◄──────────────────────|──────────────────────►
     dim=16               dim=64            dim=256
     |                     |                     |
     v                     v                     v
   Not enough           Good balance          Too many
   capacity             of capacity           parameters
   to learn             and regularization    → overfits
```

Report the **parameter count** for each setting:

$$
\text{Parameters} = (\text{num\_users} + \text{num\_items}) \times \text{embed\_dim}
$$

For Amazon-Book with 64-dim embeddings:

$$
(52{,}643 + 91{,}599) \times 64 = 9{,}231{,}488 \text{ parameters}
$$

---

## 5. Experiment 4 — Data Sparsity

This is one of the most interesting experiments for a BTP. It answers: "How do models behave when data is scarce?"

### Setup

Subsample the training data to simulate different sparsity levels:

```python
import numpy as np

sparsity_levels = [1.0, 0.75, 0.50, 0.25, 0.10]

for ratio in sparsity_levels:
    # Subsample training interactions (keep val/test fixed)
    n_keep = int(len(train) * ratio)
    indices = np.random.choice(len(train), n_keep, replace=False)
    train_subset = train.iloc[indices]

    # Rebuild graph from subsampled training data
    # Train all models on this subset
    # Evaluate on same test set
```

### Expected Results

```
Table 4: Model performance under varying data sparsity

              Recall@10                          NDCG@10
Data %   Pop    MF    GCN    LightGCN      Pop    MF    GCN    LightGCN
─────────────────────────────────────────────────────────────────────────
100%     .XXX   .XXX  .XXX   .XXX          .XXX   .XXX  .XXX   .XXX
75%      .XXX   .XXX  .XXX   .XXX          .XXX   .XXX  .XXX   .XXX
50%      .XXX   .XXX  .XXX   .XXX          .XXX   .XXX  .XXX   .XXX
25%      .XXX   .XXX  .XXX   .XXX          .XXX   .XXX  .XXX   .XXX
10%      .XXX   .XXX  .XXX   .XXX          .XXX   .XXX  .XXX   .XXX
```

### What to Analyze

```
Recall@10
   ^
   |  ──── LightGCN
   | ╱──── MF
   |╱
   |╲     At extreme sparsity, graph
   | ╲    becomes disconnected.
   |  ╲── GNN advantage may shrink or vanish.
   +--+--+--+--+--→ Data %
   100 75 50 25 10

Key questions:
1. Does LightGCN beat MF at all sparsity levels?
2. At what sparsity does the GNN advantage disappear?
3. Which model degrades most gracefully?
```

---

## 6. Ablation Studies

An ablation study removes or changes one component at a time to measure its contribution.

### What Is an Ablation?

```
Full model:

Component A + Component B + Component C  →  Score: 0.85
                                                    |
Ablation 1: Remove A                                |
                                                    v
Component B + Component C                →  Score: 0.78
                                                    |
Ablation 2: Remove B                                |
                                                    v
Component A + Component C                →  Score: 0.83
                                                    |
Ablation 3: Remove C                                |
                                                    v
Component A + Component B                →  Score: 0.80

Conclusion:
- A contributes 0.85 - 0.78 = 0.07
- B contributes 0.85 - 0.83 = 0.02
- C contributes 0.85 - 0.80 = 0.05
```

### Example Ablation for LightGCN

```
Table 5: Ablation study

Variant                          Recall@10    NDCG@10
──────────────────────────────────────────────────────
LightGCN (full)                  0.0XXX       0.0XXX
— without layer combination      0.0XXX       0.0XXX
— without symmetric norm         0.0XXX       0.0XXX
— with nonlinear activation      0.0XXX       0.0XXX
— with feature transformation    0.0XXX       0.0XXX
```

This is powerful because it shows you understand **why** each design choice matters.

---

## 7. Hyperparameter Tuning Strategy

### Don't Overfit Your Hyperparameters

Tune on the **validation set**, report final results on the **test set**.

```
Training data ──→ Train model
                      |
Validation data ──→ Select best hyperparameters
                      |
Test data ─────────→ Report final performance (ONCE)
```

### Grid Search (Simple, Recommended for BTP)

```python
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'embedding_dim': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'reg_weight': [1e-5, 1e-4, 1e-3],
}

# Total: 3 × 3 × 3 × 3 = 81 configurations
# That's manageable for LightGCN on Amazon-Book
```

### Report Your Hyperparameters

In your thesis, include a table:

```
Table: Hyperparameters for each model

Parameter         Pop    MF      GCN     LightGCN
───────────────────────────────────────────────────
Embedding dim     —      64      64      64
Layers            —      —       3       3
Learning rate     —      1e-3    1e-3    1e-3
L2 reg            —      1e-4    1e-4    1e-5
Batch size        —      1024    1024    1024
Neg. samples      —      1       1       1
Optimizer         —      Adam    Adam    Adam
```

---

## 8. Results Presentation

### Tables

Use the format shown throughout this document. Key rules:
- Bold the best result per column
- Include ± standard deviation
- Align decimal points
- Include all models in every table

### Charts

For experiments with a continuous variable (layers, embedding dim, sparsity):

```python
import matplotlib.pyplot as plt

layers = [1, 2, 3, 4, 5]
recall_mean = [0.062, 0.071, 0.074, 0.072, 0.068]
recall_std  = [0.002, 0.001, 0.002, 0.003, 0.002]

plt.figure(figsize=(8, 5))
plt.errorbar(layers, recall_mean, yerr=recall_std,
             marker='o', capsize=5, linewidth=2)
plt.xlabel('Number of GNN Layers')
plt.ylabel('Recall@10')
plt.title('Effect of GNN Depth on Recommendation Quality')
plt.xticks(layers)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/layer_depth.pdf', dpi=300)
```

### Writing About Results

**Bad:** "Table 1 shows that LightGCN gets 0.074 Recall@10."

**Good:** "LightGCN outperforms all baselines, achieving a Recall@10 of 0.074 — a 12.1% relative improvement over matrix factorization (0.066). This suggests that capturing higher-order collaborative signals through graph propagation provides meaningful benefit beyond the latent factor model."

Always:
1. State the result
2. Compare to the baseline
3. Interpret **why**

---

## 9. Reproducibility

### Set Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

SEEDS = [42, 123, 456, 789, 1024]

results = []
for seed in SEEDS:
    set_seed(seed)
    result = train_and_evaluate(model, config)
    results.append(result)

mean_recall = np.mean([r['recall@10'] for r in results])
std_recall  = np.std([r['recall@10'] for r in results])
print(f"Recall@10: {mean_recall:.4f} ± {std_recall:.4f}")
```

### Save Configs

```python
import json

config = {
    'model': 'LightGCN',
    'dataset': 'amazon-book',
    'embedding_dim': 64,
    'num_layers': 3,
    'lr': 1e-3,
    'reg_weight': 1e-5,
    'batch_size': 1024,
    'epochs': 500,
    'early_stop_patience': 50,
    'seed': 42,
}

with open(f'configs/{config["model"]}_{config["dataset"]}.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### Log Training

```python
# Simple logging — no need for Weights & Biases unless you want it
import csv

with open('logs/training_log.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([epoch, train_loss, val_recall, val_ndcg])
```

---

## 10. Exercises

1. **Create a configuration file** for your main experiment (model comparison). Specify every hyperparameter for every model.

2. **Write the code for Experiment 2** (layer depth sweep). Include:
   - Config generation loop
   - Training loop
   - Results collection
   - Plotting

3. **Design an ablation study** for your chosen model. What are the components you can remove or modify?

4. **Practice writing about results.** Take any result from a paper you've read and write a one-paragraph analysis: state the number, compare to baseline, interpret why.

---

## Key Takeaways

- **Fair comparison** is non-negotiable: same data, same splits, same embedding dimension
- Design **four types of experiments:** model comparison, hyperparameter sensitivity, sparsity analysis, and ablation
- Run each experiment with **5 random seeds** and report mean ± std
- The **sparsity experiment** is especially interesting for a BTP — it reveals when GNNs actually help
- **Ablation studies** show you understand your model, not just used it
- Always tune on **validation**, report on **test** — never mix them
- **Write about results**, don't just show tables — state, compare, interpret
- Save configs and logs for **reproducibility** — your future self (and committee) will thank you
