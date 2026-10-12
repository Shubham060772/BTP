# Phase 13 — Formulating Your BTP Research Question (Week 8–9)

> **Goal:** Move from "I built a recommendation system" to "I investigated a specific research question about graph-based recommendation."

---

## 1. Why You Need a Research Question

A BTP is not a project report. It's an undergraduate research thesis.

```
Project report:                     Research thesis:
"I made a thing"                    "I investigated a question"
   |                                   |
   v                                   v
Description of what you built       Problem → Hypothesis → Experiment → Analysis
   |                                   |
   v                                   v
Demo / screenshots                  Evidence-based conclusions
```

Your thesis committee evaluates **your reasoning**, not your code. A beautifully engineered system with no research question is a portfolio project, not a thesis.

---

## 2. What Makes a Good Undergraduate Research Question

A good BTP research question is:

| Property | Good | Bad |
|----------|------|-----|
| **Specific** | "Does adding genre metadata to LightGCN improve Recall@10 on sparse datasets?" | "Can GNNs do recommendation?" |
| **Testable** | You can design an experiment to answer it | It requires subjective judgment |
| **Scoped** | Answerable in 3–4 weeks of experiments | Requires a PhD-level investigation |
| **Comparative** | Compares approaches or conditions | Only describes one thing |
| **Non-trivial** | The answer isn't obvious | Everyone already knows the answer |

### The Formula

A reliable formula for an undergraduate research question:

$$
\text{RQ} = \text{Does } [\text{modification}] \text{ improve } [\text{metric}] \text{ compared to } [\text{baseline}] \text{ under } [\text{condition}] \text{?}
$$

---

## 3. Example Research Questions

### RQ1: Graph-Based CF vs Traditional MF

> **Does graph-based collaborative filtering (LightGCN) outperform matrix factorization for implicit feedback recommendation?**

```
Why it works:
- Clear comparison: LightGCN vs MF
- Clear metric: Recall@K, NDCG@K
- Clear setting: implicit feedback

How to answer:
- Implement both models
- Same dataset, same splits, same embedding dim
- Compare metrics across K values
```

**Difficulty:** Low — good as a foundational experiment even if it's not your main RQ.

---

### RQ2: Effect of Graph Depth

> **How does the number of GNN propagation layers affect recommendation quality, and at what depth does oversmoothing degrade performance?**

```
Experiment design:

Layers:     1      2      3      4      5
            |      |      |      |      |
            v      v      v      v      v
Recall@10:  ?      ?      ?      ?      ?
NDCG@10:    ?      ?      ?      ?      ?

Expected pattern:
Performance
    ^
    |     *
    |   *   *
    | *       *
    |*          * *
    +--+--+--+--+---> Layers
    1  2  3  4  5
```

**Difficulty:** Medium — requires understanding oversmoothing and careful hyperparameter control.

---

### RQ3: Item Metadata in GNN Recommendation

> **Does incorporating item side information (genre, category) into graph propagation improve recommendation accuracy, particularly under sparse interaction settings?**

```
Comparison:

Model A: LightGCN on User-Item bipartite graph
                                                    ──→ Compare
Model B: LightGCN on User-Item-Genre graph

Tested under:
- Full data (100% interactions)
- Sparse data (25% interactions)
- Very sparse data (10% interactions)
```

**Difficulty:** Medium-High — involves heterogeneous graph construction.

---

### RQ4: Graph Sparsity and GNN Performance

> **How does interaction sparsity affect the relative performance of GNN-based recommendation compared to non-graph baselines?**

```
Hypothesis: GNNs benefit more from graph structure
when data is moderately sparse, but degrade when
data becomes extremely sparse (graph is disconnected).

Experiment:

Sparsity:  100%  75%  50%  25%  10%
             |    |    |    |    |
MF:          .85  .80  .70  .55  .35
LightGCN:    .90  .87  .78  .60  .30
                                   ^
                                   |
                       At extreme sparsity,
                       graph becomes too disconnected
                       for message passing to help
```

**Difficulty:** Medium — requires systematic data subsampling experiments.

---

### RQ5: Heterogeneous vs Homogeneous Graphs

> **Can heterogeneous graphs (with item metadata as separate node types) improve recommendation quality compared to homogeneous user-item bipartite graphs?**

```
Graph A (homogeneous):       Graph B (heterogeneous):

User ─── Item                User ─── Item ─── Category
                                        |
                                      Brand
```

**Difficulty:** High — requires implementing heterogeneous GNNs.

---

## 4. How to Frame Your Contribution

The most realistic BTP contribution follows this pattern:

```
┌─────────────────────────────┐
│  Take an established model  │
│       (e.g., LightGCN)     │
└──────────────┬──────────────┘
               │
               v
┌─────────────────────────────┐
│   Identify a limitation     │
│  (e.g., ignores metadata)   │
└──────────────┬──────────────┘
               │
               v
┌─────────────────────────────┐
│   Propose a modification    │
│  (e.g., add genre edges)    │
└──────────────┬──────────────┘
               │
               v
┌─────────────────────────────┐
│   Experimentally evaluate   │
│  (compare before & after)   │
└─────────────────────────────┘
```

### Concrete Example

**Established model:** LightGCN uses only user-item interactions.

**Limitation:** When interactions are sparse, LightGCN struggles because the graph has too few edges for effective message passing.

**Modification:** Add item-metadata edges (item → genre, item → category) to densify the graph without adding noisy interaction edges.

**Evaluation:** Compare original LightGCN vs modified LightGCN across different sparsity levels on Amazon-Book.

**Your contribution sentence:**

> "We investigate whether incorporating item metadata into the LightGCN propagation graph improves recommendation quality under sparse interaction settings, and find that metadata-augmented graphs yield X% improvement in NDCG@10 when only 25% of interactions are available."

---

## 5. Writing Your Problem Statement

A problem statement has three parts:

### Part 1: Context (what exists)

> Recommendation systems help users discover relevant items. Collaborative filtering is the dominant paradigm, and recent work has shown that formulating recommendation as link prediction on a user-item graph allows graph neural networks to capture higher-order interaction patterns.

### Part 2: Gap (what's missing or unclear)

> However, most GNN-based recommenders rely solely on user-item interaction structure and do not leverage available item metadata. It remains unclear whether incorporating side information into the graph propagation mechanism improves recommendation quality, particularly in sparse-data regimes common in real-world applications.

### Part 3: Your contribution (what you do about it)

> In this work, we investigate the effect of metadata-augmented graph structures on GNN-based recommendation. We extend LightGCN to propagate through item-metadata edges and evaluate the approach against standard baselines on Amazon-Book under varying sparsity conditions.

---

## 6. Connecting Research Question to Experiments

Each RQ should map directly to one or more experiments:

```
RQ1: GNN vs MF
  └──→ Experiment: Model comparison table
       (Popularity, MF, GCN, LightGCN)

RQ2: Graph depth
  └──→ Experiment: Layer sweep (1–5 layers)
       with Recall@10 and NDCG@10

RQ3: Metadata
  └──→ Experiment: LightGCN vs LightGCN+metadata
       across sparsity levels

RQ4: Sparsity
  └──→ Experiment: Subsample training data
       (100%, 75%, 50%, 25%, 10%)

RQ5: Heterogeneous
  └──→ Experiment: Bipartite vs heterogeneous graph
       with same model architecture
```

If you can't design an experiment to answer your RQ, **rewrite the RQ**.

---

## 7. What Your Thesis Committee Looks For

```
Strong thesis:                          Weak thesis:
                                        
✓ Clear research question               ✗ Vague "we built X"
✓ Justified methodology                 ✗ "We used GNN because it's popular"
✓ Fair baselines                        ✗ Only reports one model
✓ Multiple experiments                  ✗ One table of results
✓ Analysis of WHY results happen        ✗ Only reports numbers
✓ Acknowledges limitations              ✗ Claims everything works
✓ Reproducible setup                    ✗ "We tuned until it worked"
```

Your committee will likely ask:

1. **"Why did you choose this approach?"** — You need a justified answer, not "because it's cool."
2. **"What if your hypothesis is wrong?"** — A negative result (GNN doesn't outperform MF in your setting) is still a valid finding if you analyze why.
3. **"How do you know it's not just hyperparameters?"** — You need fair comparison: same embedding dim, same training epochs, same data splits.
4. **"What are the limitations?"** — Demonstrating you understand what you didn't cover is a sign of maturity.

---

## 8. Tips for Making Your Thesis Defensible

### Tip 1: Don't Overclaim

Bad: "Our model achieves state-of-the-art recommendation."

Good: "Our modification shows a 3.2% improvement in NDCG@10 on Amazon-Book under 25% sparsity."

### Tip 2: Embrace Negative Results

If LightGCN doesn't beat MF on your dataset, that's interesting. Analyze why:
- Is the graph too sparse?
- Is the dataset too small for GNN benefits?
- Does the domain have different characteristics?

### Tip 3: Have Baselines

Always compare against:
- A trivial baseline (popularity)
- A standard non-graph baseline (MF)
- A standard graph baseline (GCN)
- Your chosen model (LightGCN)
- Your proposed modification

### Tip 4: Report Fairly

- Same embedding dimension across all models
- Same number of training epochs (or early stopping criterion)
- Same train/val/test split
- Report mean and standard deviation over multiple runs

### Tip 5: Pick One Main RQ, Not Five

You can have sub-questions, but one main question drives the thesis:

```
Main RQ: Does metadata improve GNN recommendation under sparsity?
  |
  ├── Sub-RQ 1: Which metadata types help most? (category vs brand vs price)
  ├── Sub-RQ 2: At what sparsity level does metadata stop helping?
  └── Sub-RQ 3: How does this interact with graph depth?
```

---

## 9. Exercises

1. **Write three candidate research questions** for your BTP. For each one, write:
   - The question
   - What experiments you would run to answer it
   - What a positive result would look like
   - What a negative result would tell you

2. **Write a one-paragraph problem statement** using the Context → Gap → Contribution structure.

3. **Map your main RQ to a specific comparison table.** What would the rows (models) and columns (metrics) be?

4. **Identify two potential "why" questions** your committee might ask and draft answers.

---

## Key Takeaways

- A BTP requires a **research question**, not just a working system
- Use the formula: Does [modification] improve [metric] compared to [baseline] under [condition]?
- The strongest BTP contribution: **established model → identified limitation → proposed modification → experimental evaluation**
- Every research question must map to a concrete experiment
- Negative results are valid findings if you analyze **why**
- Pick **one main research question** — depth beats breadth in a thesis
- Your committee evaluates your **reasoning and methodology**, not your code
