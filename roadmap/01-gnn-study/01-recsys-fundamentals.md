# Phase 1 — Recommendation Systems Fundamentals

**Week 1 | BTP Roadmap**

---

## 1.1 What Are Recommendation Systems?

A recommendation system is a model that predicts which items a user is likely to interact with, purchase, watch, or enjoy. Every time you see "You might also like..." on an e-commerce site, or "Recommended for you" on a streaming platform, a recommendation system is at work.

Why do they matter?

- **For users:** They reduce the effort of finding relevant items in a sea of choices.
- **For businesses:** They increase engagement, sales, and retention.
- **At scale:** Netflix estimates its recommendation system saves $1B per year in reduced churn.

For your BTP, the recommendation system you build will be the *application layer* that sits on top of your GNN model. Before we learn GNNs, we need to understand what we're trying to solve.

---

## 1.2 The User-Item Interaction Matrix

The starting point for almost every recommendation system is a **User × Item interaction matrix**.

### Explicit Feedback (Ratings)

```
              Product A  Product B  Product C  Product D  Product E
User 1           5          4          ?          ?          1
User 2           4          ?          5          3          ?
User 3           ?          5          4          ?          ?
User 4           ?          ?          ?          4          5
User 5           3          ?          ?          5          ?
```

Each cell holds a rating (1–5). A `?` means we have no observation — the user never rated that item.

### Implicit Feedback (Interactions)

In many real systems, you don't have explicit ratings. Instead you have *implicit* signals:

```
              Item A    Item B    Item C    Item D
User 1           1         1         0         0       (1 = clicked/purchased, 0 = no interaction)
User 2           1         0         1         1
User 3           0         1         1         0
```

Implicit feedback is more common in practice (clicks, purchases, views, time-spent). Your BTP will likely work with implicit feedback.

### The Formal Goal

Let $R$ be the interaction matrix.

- $R_{ui}$ = the observed interaction between user $u$ and item $i$
- $\hat{R}_{ui}$ = the **predicted** preference of user $u$ for item $i$

The goal of a recommendation system:

$$R_{ui} \rightarrow \hat{R}_{ui}$$

We want to fill in the `?` cells with predicted scores, then recommend the items with the highest predicted scores.

---

## 1.3 Three Approaches to Recommendation

### Approach 1: Popularity-Based

The simplest approach: recommend the most popular items to everyone.

```
Most purchased products this week:
1. Product A  (purchased by 10,000 users)
2. Product C  (purchased by 8,500 users)
3. Product B  (purchased by 7,200 users)
         ↓
Recommend [A, C, B] to ALL users
```

**Pros:**
- Dead simple to implement
- Works surprisingly well as a baseline
- No cold-start problem for users

**Cons:**
- Not personalized at all
- Popular items dominate; niche items are never surfaced

**Why it matters for your BTP:** This becomes your *first baseline*. If your GNN model can't beat popularity-based recommendation, something is wrong.

### Approach 2: Content-Based Filtering

Recommend items that are *similar to items the user already liked*, based on item features.

```
User liked: "CeraVe Moisturizing Cream"
            ↓
Features: [Skincare, Moisturizer, CeraVe, Fragrance-Free]
            ↓
Find similar items by feature overlap
            ↓
Recommend: "CeraVe Daily Lotion", "Cetaphil Cream", "Vanicream Moisturizer"
```

The process:

```
User preferences (from history)
         ↓
Item features (category, brand, tags, description)
         ↓
Compute similarity (cosine similarity, etc.)
         ↓
Rank items by similarity
         ↓
Recommend top-K
```

**Pros:**
- Personalized
- No need for other users' data
- Can explain recommendations ("because you purchased X")

**Cons:**
- Needs rich item features
- Creates a *filter bubble* — only recommends similar things
- Can't discover unexpected preferences

### Approach 3: Collaborative Filtering

Recommend items based on what *similar users* liked. This does not need item features at all — it only needs the interaction matrix.

```
User 1: liked [A, B, C]
User 2: liked [A, B, D]        ← User 2 is similar to User 1
                                   (both liked A and B)
         ↓
Recommend D to User 1          ← because similar User 2 liked D
```

The core assumption:

> Users who agreed in the past will agree in the future.

There are two flavors:

**User-based CF:**
```
Find users similar to target user
         ↓
Aggregate their preferences
         ↓
Recommend
```

**Item-based CF:**
```
Find items similar to items the user liked
         ↓
(Similarity = users who interacted with both)
         ↓
Recommend
```

**Pros:**
- No item features needed
- Can discover surprising preferences
- Naturally captures taste patterns

**Cons:**
- Needs enough interaction data
- Struggles with new users (cold-start)
- Struggles with new items (cold-start)

### Summary Table

```
Approach            Needs Features?    Personalized?    Cold-Start?
─────────────────────────────────────────────────────────────────
Popularity          No                 No               No problem
Content-Based       Yes (items)        Yes              User OK, Item needs features
Collaborative       No                 Yes              Struggles with both
```

---

## 1.4 Why Collaborative Filtering Is the Foundation for Your BTP

Your BTP will be built on collaborative filtering for these reasons:

1. **It maps directly to graphs.** User-item interactions form a natural bipartite graph. This is the bridge to GNNs.

2. **Matrix Factorization (next phase) is a form of CF.** The $R \approx UV^T$ decomposition learns latent user and item representations from interactions alone.

3. **GNN-based recommendation is an extension of CF.** Models like LightGCN propagate information through the user-item graph — this is collaborative filtering implemented through graph neural networks.

4. **It provides a clear experimental comparison.** Your thesis can compare traditional CF (Matrix Factorization) against graph-based CF (GCN, LightGCN).

```
Collaborative Filtering
         │
         ├── Traditional CF
         │     ├── User-based
         │     ├── Item-based
         │     └── Matrix Factorization  ← Phase 2
         │
         └── Graph-based CF
               ├── GCN                   ← Phase 5
               ├── NGCF                  ← Phase 8
               └── LightGCN             ← Phase 8 (your main model)
```

---

## 1.5 The Cold-Start Problem

A brief introduction to a problem you'll encounter later.

**New user cold-start:**
```
New User (just signed up)
         ↓
No interaction history
         ↓
CF has nothing to work with
         ↓
What do we recommend?
```

**New item cold-start:**
```
New Item (just added to catalog)
         ↓
No users have interacted with it
         ↓
CF can't learn its representation
         ↓
It never gets recommended → it never gets interactions → cycle
```

Solutions include using item features (content-based), asking users for initial preferences, or — relevant to your BTP — incorporating side information into the graph. We'll revisit this in Phase 15 as an advanced direction.

---

## 1.6 Putting It All Together

Here's where we are and where we're going:

```
Phase 1 (you are here)
    Understand the recommendation problem
         ↓
Phase 2
    Matrix Factorization — learn user/item embeddings from R
         ↓
Phase 3
    Represent R as a graph
         ↓
Phase 4–5
    Use GNNs to learn embeddings from the graph
         ↓
Phase 8
    LightGCN — a GNN designed specifically for recommendation
         ↓
Phase 10+
    Build, evaluate, and improve your BTP system
```

---

## Exercises

1. **Think about data.** Suppose you're building a product recommendation system for Amazon. What data would you collect? Write out what your interaction matrix might look like (explicit vs implicit).

2. **Compare approaches.** Given this interaction matrix:

```
              Product A  Product B  Product C  Product D
User 1           1         1         0         ?
User 2           1         1         1         0
User 3           0         0         1         1
```

   - What would a popularity-based recommender suggest for User 1?
   - Using user-based CF, which user is most similar to User 1? What would you recommend?

3. **Think about your BTP.** You'll be working with Amazon product datasets (Amazon-Book, Amazon-Beauty). What kind of feedback is available in these datasets (explicit ratings or implicit interactions)? How would you convert one to the other?

---

## Key Takeaways

- Recommendation is fundamentally a **matrix completion problem**: fill in the missing entries of the User × Item matrix.
- **Popularity** is the simplest baseline but not personalized.
- **Content-based** filtering uses item features but creates filter bubbles.
- **Collaborative filtering** uses interaction patterns and is the foundation for your BTP.
- Your BTP path: CF → Matrix Factorization → Graph representation → GNN-based CF.
- Always keep popularity-based and matrix factorization as baselines — your GNN model must beat them to be meaningful.
