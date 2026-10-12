# Phase 16 — Final BTP Model and Thesis Structure (Week 11–12)

> **Goal:** Assemble your final system, write your thesis, and present a defensible undergraduate research contribution.

---

## 1. Your Final System Architecture

```
                         Raw Data
                           │
                           v
                ┌─────────────────────┐
                │  Data Preprocessing │
                │  - Filter users     │
                │  - Implicit feedback│
                │  - Re-index IDs     │
                │  - Train/Val/Test   │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │  Graph Construction │
                │  - User nodes       │
                │  - Item nodes       │
                │  - Interaction edges│
                │  - (Metadata edges) │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │      LightGCN      │
                │  Layer 1: propagate │
                │  Layer 2: propagate │
                │  Layer 3: propagate │
                │  Layer combination  │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │  User Embeddings    │
                │  Item Embeddings    │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │   BPR Optimization  │
                │  (u, pos_i, neg_i)  │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │  Candidate Scoring  │
                │  score = u^T · v    │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │    Top-K Items      │
                │  Sort by score      │
                │  Return top K       │
                └──────────┬──────────┘
                           │
                           v
                ┌─────────────────────┐
                │   Evaluation        │
                │  Recall@K, NDCG@K   │
                └─────────────────────┘
```

### Complete Pipeline Code

```python
# train.py — simplified but complete

import torch
import torch.nn as nn
from torch_geometric.nn import LGConv  # LightGCN conv layer

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_embed.weight, self.item_embed.weight])
        all_x = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_x.append(x)

        # Layer combination: average all layers
        x = torch.stack(all_x, dim=0).mean(dim=0)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        return user_emb, item_emb

    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss


# Training loop
model = LightGCN(num_users, num_items, embed_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    user_emb, item_emb = model(edge_index)

    # Sample batch
    users, pos_items, neg_items = sample_batch(train_data, batch_size=1024)

    loss = model.bpr_loss(
        user_emb[users],
        item_emb[pos_items],
        item_emb[neg_items]
    )

    # L2 regularization
    reg_loss = (
        model.user_embed(users).norm(2).pow(2) +
        model.item_embed(pos_items).norm(2).pow(2) +
        model.item_embed(neg_items).norm(2).pow(2)
    ) / len(users) * 1e-5

    total_loss = loss + reg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Validate
    if epoch % 10 == 0:
        recall, ndcg = evaluate(model, edge_index, val_data, K=10)
        print(f"Epoch {epoch}: Loss={total_loss:.4f} Recall@10={recall:.4f} NDCG@10={ndcg:.4f}")
```

---

## 2. The "Novelty" — Your Thesis Contribution

You don't need to invent a new architecture. A realistic BTP contribution:

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Established model:  LightGCN                          │
│                                                        │
│  Identified limitation:                                │
│    Only uses user-item interaction structure.           │
│    When interactions are sparse, the graph has          │
│    too few edges for effective message passing.         │
│                                                        │
│  Proposed modification:                                │
│    Augment the graph with item-metadata edges           │
│    (item → genre, item → category) to densify          │
│    the graph without adding noisy interaction edges.   │
│                                                        │
│  Experimental evaluation:                              │
│    Compare original LightGCN vs modified LightGCN     │
│    across different sparsity levels.                   │
│                                                        │
│  Finding:                                              │
│    Metadata-augmented graphs yield X% improvement      │
│    in NDCG@10 when only 25% of interactions are        │
│    available, while showing no significant change      │
│    under full-data conditions.                         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### How to Describe Your Contribution

In your thesis abstract:

> "In this work, we investigate whether augmenting the LightGCN interaction graph with item metadata improves recommendation quality under sparse data conditions. Through systematic experiments on Amazon-Book, we show that metadata-enriched graphs provide a statistically significant improvement in NDCG@10 under 25% data availability (0.XXX vs 0.XXX, p < 0.05), while offering comparable performance under full-data conditions. Our analysis suggests that metadata edges serve as an effective regularization mechanism when interaction signals are insufficient."

---

## 3. Thesis Structure

Your thesis should follow this structure. Each section is described with approximate page counts.

### Chapter 1: Introduction (3–4 pages)

```
1.1 Background and Motivation
    - Why recommendation systems matter
    - The shift from traditional to graph-based approaches

1.2 Problem Statement
    - Context → Gap → Contribution
    (See Phase 13 for how to write this)

1.3 Research Questions
    - Main RQ
    - Sub-RQs (if any)

1.4 Thesis Outline
    - Brief description of each chapter
```

### Chapter 2: Literature Review (8–10 pages)

```
2.1 Recommendation Systems
    2.1.1 Content-based filtering
    2.1.2 Collaborative filtering
    2.1.3 Matrix factorization

2.2 Graph Neural Networks
    2.2.1 Graph representation
    2.2.2 Message passing framework
    2.2.3 GCN
    2.2.4 GraphSAGE
    2.2.5 GAT

2.3 GNN-Based Recommendation
    2.3.1 NGCF
    2.3.2 LightGCN
    2.3.3 Other approaches (PinSage, UltraGCN)

2.4 Evaluation in Recommendation Systems
    2.4.1 Ranking metrics (Recall@K, NDCG@K)
    2.4.2 BPR loss and implicit feedback

2.5 Summary and Research Gap
```

### Chapter 3: Methodology (6–8 pages)

```
3.1 Problem Formulation
    - User-item bipartite graph definition
    - Link prediction as recommendation

3.2 Baseline Models
    3.2.1 Popularity baseline
    3.2.2 Matrix Factorization with BPR

3.3 LightGCN
    3.3.1 Architecture
    3.3.2 Propagation rule
    3.3.3 Layer combination
    3.3.4 Training with BPR loss

3.4 Proposed Modification (if applicable)
    3.4.1 Motivation
    3.4.2 Modified architecture
    3.4.3 Training procedure

3.5 Evaluation Protocol
    3.5.1 Dataset
    3.5.2 Train/Val/Test split
    3.5.3 Metrics
    3.5.4 Hyperparameter tuning
```

### Chapter 4: Experimental Setup (4–5 pages)

```
4.1 Dataset Description
    - Statistics table
    - Sparsity analysis

4.2 Implementation Details
    - Framework (PyTorch, PyG)
    - Hyperparameters (full table)
    - Hardware

4.3 Experiment Design
    - Experiment 1: Model comparison
    - Experiment 2: Layer depth
    - Experiment 3: Embedding dimension
    - Experiment 4: Data sparsity
    - Ablation study
```

### Chapter 5: Results and Discussion (8–10 pages)

```
5.1 Model Comparison (Table + analysis)
5.2 Effect of Graph Depth (Table + chart + analysis)
5.3 Effect of Embedding Dimension (Table + analysis)
5.4 Sparsity Analysis (Table + chart + analysis)
5.5 Ablation Study (Table + analysis)
5.6 Discussion
    - Key findings
    - Why results make sense
    - Unexpected results
```

### Chapter 6: Conclusion (2–3 pages)

```
6.1 Summary of Findings
6.2 Limitations
6.3 Future Work
```

### References (2–3 pages)

### Appendix

```
A. Full hyperparameter tables
B. Additional results
C. Code repository link
```

**Total: approximately 35–45 pages**

---

## 4. Papers to Read (Prioritized)

### Must Read (before you start writing)

| Priority | Paper | Year | Why |
|----------|-------|------|-----|
| 1 | Koren et al. — "Matrix Factorization Techniques for Recommender Systems" | 2009 | Foundation of your MF baseline |
| 2 | Kipf & Welling — "Semi-Supervised Classification with Graph Convolutional Networks" | 2017 | The original GCN paper |
| 3 | Wang et al. — "Neural Graph Collaborative Filtering" (NGCF) | 2019 | First major GNN recommender |
| 4 | He et al. — "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" | 2020 | Your central paper |
| 5 | Rendle et al. — "BPR: Bayesian Personalized Ranking from Implicit Feedback" | 2009 | Your loss function |

### Should Read (for depth)

| Priority | Paper | Year | Why |
|----------|-------|------|-----|
| 6 | Hamilton et al. — "Inductive Representation Learning on Large Graphs" (GraphSAGE) | 2017 | Comparison model |
| 7 | Velickovic et al. — "Graph Attention Networks" (GAT) | 2018 | Comparison / explainability |
| 8 | Ying et al. — "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage) | 2018 | Industry perspective |

### Optional (for advanced direction)

| Paper | Why |
|-------|-----|
| Mao et al. — "UltraGCN" (2021) | Simplified further from LightGCN |
| Wang et al. — "KGAT: Knowledge Graph Attention Network" (2019) | Heterogeneous graphs |
| Fan et al. — "Graph Neural Networks for Social Recommendation" (2019) | Social graph |

### Reading Strategy

```
Week 8:   Read Papers 1 and 5 (MF + BPR — your baselines)
Week 9:   Read Papers 2 and 3 (GCN + NGCF — build understanding)
Week 10:  Read Paper 4 deeply (LightGCN — your main model)
Week 11:  Read Papers 6-7 as needed for comparison/advanced direction
```

Don't try to read all papers front-to-back. For each paper:
1. Read **abstract + introduction** (5 min)
2. Read **method section** (20 min)
3. Read **experiments** (10 min)
4. Skip proofs and supplementary material unless relevant

---

## 5. Tech Stack Summary

```
Core:
├── Python 3.9+
├── PyTorch 2.x            ← Deep learning framework
├── PyTorch Geometric 2.x  ← Graph neural networks
├── NumPy                  ← Numerical operations
├── Pandas                 ← Data handling
├── Scikit-learn           ← Preprocessing, baseline metrics
├── NetworkX               ← Graph visualization, analysis
└── Matplotlib / Seaborn   ← Charts and figures

Optional:
├── Weights & Biases       ← Experiment tracking (nice to have)
└── tqdm                   ← Progress bars
```

### Project Directory Structure

```
BTP/
├── data/
│   ├── raw/                 # Original dataset files
│   └── processed/           # Preprocessed tensors
├── src/
│   ├── data/
│   │   ├── dataset.py       # Data loading and preprocessing
│   │   └── split.py         # Train/val/test splitting
│   ├── models/
│   │   ├── popularity.py    # Popularity baseline
│   │   ├── mf.py            # Matrix factorization
│   │   ├── gcn.py           # Basic GCN recommender
│   │   └── lightgcn.py      # LightGCN
│   ├── evaluation/
│   │   └── metrics.py       # Recall@K, NDCG@K
│   ├── train.py             # Training loop
│   └── evaluate.py          # Evaluation script
├── configs/
│   └── lightgcn_amazon_book.json  # Hyperparameter configs
├── results/
│   ├── tables/              # Result tables (CSV)
│   └── figures/             # Generated plots
├── notebooks/
│   └── analysis.ipynb       # Result analysis
├── thesis/
│   └── main.tex             # Thesis document
└── README.md
```

---

## 6. Timeline Management

### Weeks 1–7: Build (done by now)

```
✓ Understand recommendation systems
✓ Implement baselines (popularity, MF)
✓ Learn graph theory and GNNs
✓ Implement GCN, study GraphSAGE/GAT
✓ Learn PyTorch Geometric
✓ Study LightGCN
✓ Understand BPR loss
```

### Weeks 8–9: Experiment

```
□ Preprocess dataset
□ Implement evaluation metrics
□ Run model comparison experiment
□ Run hyperparameter experiments
□ Run sparsity experiment
□ Run ablation study
```

### Weeks 10–11: Extend and Analyze

```
□ Implement advanced direction
□ Run advanced experiments
□ Generate all tables and figures
□ Analyze results — write the "why"
```

### Weeks 11–12: Write

```
□ Write methodology chapter
□ Write experiment setup chapter
□ Write results chapter
□ Write literature review
□ Write introduction and conclusion
□ Proofread and format
```

### Time-Saving Tips

1. **Write as you go.** Don't wait until week 12 to start writing. Draft methodology while implementing, draft experiment setup while running experiments.

2. **Automate experiment running.**
```python
# run_all_experiments.py
configs = load_all_configs()
for config in configs:
    for seed in SEEDS:
        result = train_and_evaluate(config, seed)
        save_result(result)
```

3. **Generate tables from code.**
```python
# Don't manually create tables. Generate them.
import pandas as pd
results_df = pd.DataFrame(all_results)
print(results_df.to_latex(float_format="%.4f"))
```

4. **Version your experiments.** Save every config and result. You will re-run things.

---

## 7. Common BTP Mistakes to Avoid

### Mistake 1: No Baselines

```
Bad:  "LightGCN achieves 0.074 Recall@10"
      (Compared to what? Is that good?)

Good: "LightGCN achieves 0.074 Recall@10, a 12.1% improvement
       over Matrix Factorization (0.066) and 48.0% over
       Popularity (0.050)"
```

### Mistake 2: Unfair Comparisons

```
Bad:  MF with embed_dim=32 vs LightGCN with embed_dim=256
      (Of course the model with 8x parameters wins)

Good: All models use embed_dim=64, same training budget
```

### Mistake 3: No Error Analysis

```
Bad:  "Our model outperforms baselines on all metrics."

Good: "Our model outperforms baselines overall, but
       underperforms MF for users with >100 interactions.
       We hypothesize that message passing over-smooths
       heavy users' embeddings."
```

### Mistake 4: Overclaiming

```
Bad:  "We achieve state-of-the-art recommendation performance."

Good: "On Amazon-Book, our modification improves LightGCN's
       NDCG@10 by 3.2% under 25% data availability."
```

### Mistake 5: Spending Too Long on Infrastructure

```
Bad:  Weeks 1-6 building a beautiful web dashboard
      Weeks 7-12 panic-implementing the actual GNN

Good: Weeks 1-7 implementing and understanding models
      Weeks 8-10 running experiments
      Weeks 11-12 writing thesis
```

### Mistake 6: Not Understanding Your Own Model

Your committee will ask:
- "Why does LightGCN not use nonlinear activation?"
- "What happens if you increase layers to 10?"
- "Why BPR instead of MSE loss?"

If you can't answer these without reading slides, you haven't understood deeply enough. Go back to Phases 5, 8, and 9.

### Mistake 7: Ignoring Negative Results

If GNN doesn't outperform MF on your dataset, that's a **finding**, not a failure:

> "Contrary to expectations, LightGCN did not significantly outperform matrix factorization on our dataset. Analysis reveals that the dataset's high density (only 93.7% sparse) means most collaborative signals are already capturable by direct factorization, and graph propagation provides diminishing returns."

This shows more research maturity than cherry-picked positive results.

---

## 8. Exercises

1. **Write the abstract** of your thesis (150–200 words). Include: context, gap, method, key finding.

2. **Create the project directory structure** shown in Section 5. Stub out each Python file with a class/function signature and a docstring.

3. **Write a 2-page thesis outline** with bullet points for what each section will contain.

4. **Prepare your defense.** Write answers to these questions:
   - Why did you choose LightGCN?
   - Why BPR instead of MSE?
   - What are the limitations of your approach?
   - What would you do with 6 more months?

---

## Key Takeaways

- Your final system is: **Data → Graph → LightGCN → BPR → Top-K → Evaluate**
- A realistic BTP novelty: **established model + identified limitation + proposed fix + experiments**
- Your thesis should be **~35-45 pages** with clear chapters
- Read **5 core papers** deeply rather than 50 shallowly — LightGCN is your central paper
- Keep the tech stack **simple** — Python, PyTorch, PyG, Matplotlib
- **Write as you go** — don't save all writing for the last week
- **Avoid common mistakes:** no baselines, unfair comparisons, overclaiming, ignoring negative results
- The strongest signal of thesis quality is **your analysis of WHY results happen**, not the results themselves
