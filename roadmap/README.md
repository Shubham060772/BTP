# BTP Roadmap: GNN-Based Product Recommendation System

## BTP Goal

```
                    DATA (Amazon-Book / Amazon-Beauty)
                      |
          +-----------+-----------+
          v                       v
        Users                  Products
          |                       |
          +------ Interactions ---+
                      |
                      v
                   GRAPH
                      |
                      v
               GNN + Attention
              Representation
                      |
                      v
              User Embeddings
              Item Embeddings
                      |
                      v
              Link Prediction
                      |
                      v
             Recommendation
                      |
                      v
          Top-K Recommended Products
```

**Key Insight:** Recommendation can be treated as a **link-prediction problem** on a user-item graph. Combining GNN structure-awareness with attention-based importance weighting is the path to surpassing current baselines.

---

## Study Tracks

This roadmap is organized into **4 tracks** — study GNNs and Attention independently, then learn how they combine, then implement your BTP.

### Track 1: GNN Study (`01-gnn-study/`)

| # | Topic | File |
|---|-------|------|
| 1 | Recommendation Systems Fundamentals | [01-recsys-fundamentals.md](01-gnn-study/01-recsys-fundamentals.md) |
| 2 | Matrix Factorization | [02-matrix-factorization.md](01-gnn-study/02-matrix-factorization.md) |
| 3 | Graph Theory for Recommendation | [03-graph-theory.md](01-gnn-study/03-graph-theory.md) |
| 4 | Message Passing | [04-message-passing.md](01-gnn-study/04-message-passing.md) |
| 5 | Graph Convolutional Networks (GCN) | [05-gcn.md](01-gnn-study/05-gcn.md) |
| 6 | GraphSAGE + GAT | [06-graphsage-gat.md](01-gnn-study/06-graphsage-gat.md) |
| 7 | PyTorch Geometric | [07-pytorch-geometric.md](01-gnn-study/07-pytorch-geometric.md) |
| 8 | Recommendation-Specific GNNs (NGCF, LightGCN) | [08-recsys-gnns.md](01-gnn-study/08-recsys-gnns.md) |
| 9 | BPR Loss | [09-bpr-loss.md](01-gnn-study/09-bpr-loss.md) |

### Track 2: Attention Study (`02-attention-study/`)

| # | Topic | File |
|---|-------|------|
| 1 | Attention Fundamentals | [01-attention-fundamentals.md](02-attention-study/01-attention-fundamentals.md) |
| 2 | Self-Attention & Transformers | [02-self-attention-transformers.md](02-attention-study/02-self-attention-transformers.md) |
| 3 | Attention in Recommendation (SASRec, BERT4Rec, DIN) | [03-attention-in-recsys.md](02-attention-study/03-attention-in-recsys.md) |
| 4 | Advanced Attention for Recsys | [04-advanced-attention-recsys.md](02-attention-study/04-advanced-attention-recsys.md) |

### Track 3: Combined GNN + Attention (`03-combined-approaches/`)

| # | Topic | File |
|---|-------|------|
| 1 | GAT for Recommendation | [01-gat-for-recommendation.md](03-combined-approaches/01-gat-for-recommendation.md) |
| 2 | Graph Transformers | [02-graph-transformers.md](03-combined-approaches/02-graph-transformers.md) |
| 3 | Hybrid GNN-Attention Models | [03-hybrid-gnn-attention-models.md](03-combined-approaches/03-hybrid-gnn-attention-models.md) |
| 4 | State of the Art (2024-2025) | [04-state-of-the-art-2024-2025.md](03-combined-approaches/04-state-of-the-art-2024-2025.md) |

### Track 4: BTP Implementation (`04-btp-implementation/`)

| # | Topic | File |
|---|-------|------|
| 0 | **Master Plan** (start here) | [00-btp-master-plan.md](04-btp-implementation/00-btp-master-plan.md) |
| 1 | Dataset Setup | [01-dataset-setup.md](04-btp-implementation/01-dataset-setup.md) |
| 2 | Research Question | [02-research-question.md](04-btp-implementation/02-research-question.md) |
| 3 | First Recommender | [03-first-recommender.md](04-btp-implementation/03-first-recommender.md) |
| 4 | Experiments | [04-experiments.md](04-btp-implementation/04-experiments.md) |
| 5 | Evaluation | [05-evaluation.md](04-btp-implementation/05-evaluation.md) |
| 6 | Advanced Direction | [06-advanced-direction.md](04-btp-implementation/06-advanced-direction.md) |
| 7 | Final Model | [07-final-model.md](04-btp-implementation/07-final-model.md) |

---

## Recommended Study Order

```
Track 1 (GNN)                Track 2 (Attention)
     |                              |
     v                              v
  01-09                          01-04
  (Weeks 1-6)                   (Weeks 3-6, overlap)
     |                              |
     +----------+-------------------+
                |
                v
        Track 3 (Combined)
             01-04
           (Weeks 6-8)
                |
                v
        Track 4 (BTP Implementation)
          00-07 (Weeks 8-12)
```

Start Track 1 and Track 2 in parallel (Track 2 can begin once you have GCN basics from Track 1). Track 3 requires both. Track 4 is your actual BTP work.

---

## Datasets

Both datasets are in `data/`:

| Dataset | Location | Size | Feedback |
|---------|----------|------|----------|
| **Amazon-Book** | `data/amazon-book/` | 52,643 users, 91,599 items, 3M interactions | Implicit only (pre-processed) |
| **Amazon-Beauty** | `data/amazon-beauty/` | 631,986 users, 112,565 items, 701K interactions | Rich: ratings, text, timestamps, images |

---

## Tech Stack

```
Python
|
+-- NumPy
+-- Pandas
+-- Scikit-learn
+-- PyTorch
+-- PyTorch Geometric
+-- NetworkX
+-- Matplotlib
```

Optional: Weights & Biases, MLflow.

---

## Papers to Read (Prioritized)

**Foundations (read first):**
1. Collaborative Filtering / Matrix Factorization
2. GCN (Kipf & Welling, 2017)
3. Attention Is All You Need (Vaswani et al., 2017)
4. NGCF (Wang et al., 2019)
5. LightGCN (He et al., 2020)

**Attention in Recsys:**
6. SASRec (Kang & McAuley, 2018)
7. BERT4Rec (Sun et al., 2019)
8. DIN (Zhou et al., 2018)

**Combined / Recent:**
9. KGAT (Wang et al., 2019)
10. SGL (Wu et al., 2021)
11. LightGCL (Cai et al., 2023)
12. GFormer (Li et al., 2023)

---

## Slides

An interactive HTML slide deck is available at:

[slides/btp-gnn-recsys-slides.html](slides/btp-gnn-recsys-slides.html)
