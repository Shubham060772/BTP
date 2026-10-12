# Phase 3 — Graph Theory for Recommendation

**Week 2 | BTP Roadmap**

---

## 3.1 Why Graphs?

In Phase 2, you learned Matrix Factorization: decompose the interaction matrix $R$ into user and item embeddings. That works, but it treats each (user, item) pair independently.

What if User 1 purchased Product A, and User 2 also purchased Product A plus Product B? There's a transitive signal:

```
User 1 ── Product A ── User 2 ── Product B
```

User 1 might like Product B through their shared connection via Product A and User 2. MF captures some of this implicitly through shared latent factors, but a **graph** captures this structure explicitly.

This phase teaches you the graph concepts you need — nothing more, nothing less.

---

## 3.2 Nodes and Edges

A **graph** $G = (V, E)$ consists of:

- **Nodes** (or vertices) $V$: the entities
- **Edges** $E$: the connections between entities

```
    A ────── B
    │        │
    │        │
    C ────── D
```

Here: $V = \{A, B, C, D\}$, $E = \{(A,B), (A,C), (B,D), (C,D)\}$

For recommendation:
- **Nodes** = users and items
- **Edges** = interactions (watched, purchased, clicked)

---

## 3.3 The Adjacency Matrix

The **adjacency matrix** $A$ is a square matrix that represents which nodes are connected.

For the graph above:

```
     A  B  C  D
A [  0  1  1  0 ]
B [  1  0  0  1 ]
C [  1  0  0  1 ]
D [  0  1  1  0 ]
```

$A_{ij} = 1$ if there's an edge between node $i$ and node $j$, else 0.

Properties:
- For undirected graphs, $A$ is **symmetric**: $A_{ij} = A_{ji}$
- The diagonal is 0 (no self-loops, by default)
- For $n$ nodes, $A$ is $n \times n$

### Worked Example

```
    1 ── 2
    │  ╲
    3    4
```

$$A = \begin{bmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}$$

Node 1 is connected to nodes 2, 3, and 4. Nodes 2, 3, 4 are only connected to node 1.

---

## 3.4 Node Features and Edge Features

Nodes and edges can carry **features** (attributes).

**Node features** for a product recommendation graph:

```
User nodes:
  User 1: age=25, gender=M         → feature vector [25, 1]
  User 2: age=30, gender=F         → feature vector [30, 0]

Item nodes:
  Product A: price=15, category=Skincare   → feature vector [15, 1, 0, 0]
  Product B: price=25, category=Haircare   → feature vector [25, 0, 1, 0]
```

The feature matrix $X$ has shape $(n \times f)$ where $n$ = number of nodes and $f$ = feature dimension.

**Edge features** might include:

```
User 1 ──[rating=5]──> Product A
User 1 ──[rating=3]──> Product B
```

The rating, timestamp, or interaction type can all be edge features.

For your BTP: in the simplest setting (LightGCN), you may not use node features at all — the model learns embeddings from scratch. But understanding features is important if you explore richer models later.

---

## 3.5 Directed vs Undirected Graphs

**Undirected:** edges have no direction. "A is connected to B" is the same as "B is connected to A."

```
A ────── B       A_{AB} = A_{BA} = 1
```

**Directed:** edges have direction. "A points to B" is NOT the same as "B points to A."

```
A ──────→ B      A_{AB} = 1, A_{BA} = 0
```

For recommendation:
- "User purchased Item" is often modeled as **undirected** for collaborative filtering — we care that an interaction exists, not the direction.
- In more advanced models (e.g., knowledge graphs), direction matters: "Product → has_category → Skincare" is different from "Skincare → has_category → Product."

Your BTP's user-item graph will typically be **undirected**.

---

## 3.6 Weighted Graphs

Edges can carry **weights** indicating strength of connection.

```
User 1 ──[5]── Product A       (loved it)
User 1 ──[2]── Product B       (didn't enjoy)
```

The adjacency matrix becomes:

```
         Product A  Product B
User 1 [    5          2     ]
```

For implicit feedback (binary: interacted or not), you often use **unweighted** edges:

```
User 1 ──[1]── Product A       (interacted)
User 1 ──[0]── Product B       (no interaction → no edge)
```

---

## 3.7 Bipartite Graphs — THE Key Concept

This is the single most important graph concept for your BTP.

A **bipartite graph** has two types of nodes, and edges only connect nodes of *different* types.

```
Users              Items

User 1 ─────────── Product A
   │
   └────────────── Product C

User 2 ─────────── Product A
   │
   └────────────── Product B

User 3 ─────────── Product B
```

Formally:

$$G = (U, I, E)$$

Where:
- $U$ = set of user nodes
- $I$ = set of item nodes
- $E \subseteq U \times I$ = edges (interactions)

Crucially: **there are no user-user edges and no item-item edges.** Every edge connects a user to an item.

### Why Bipartite?

Your recommendation problem is *naturally* bipartite:

```
Interaction Matrix R              Bipartite Graph G
                                  
        Item1  Item2  Item3       User1 ── Item1
User1 [  1      0      1  ]      User1 ── Item3
User2 [  1      1      0  ]      User2 ── Item1
User3 [  0      1      1  ]      User2 ── Item2
                                  User3 ── Item2
                                  User3 ── Item3
```

The interaction matrix and the bipartite graph are **two representations of the same data**. Every 1 in $R$ corresponds to an edge in $G$.

### The Adjacency Matrix of a Bipartite Graph

For a bipartite graph with $m$ users and $n$ items, the full adjacency matrix is:

$$A = \begin{bmatrix} 0_{m \times m} & R \\ R^T & 0_{n \times n} \end{bmatrix}$$

Where $R$ is the interaction matrix. The block structure shows:
- No user-user edges (top-left zeros)
- No item-item edges (bottom-right zeros)
- User-item edges in $R$ (top-right) and $R^T$ (bottom-left, symmetric)

Example with 2 users, 3 items:

```
R = [ 1  0  1 ]       A = [ 0  0 | 1  0  1 ]    ← User 1
    [ 1  1  0 ]           [ 0  0 | 1  1  0 ]    ← User 2
                          [------+----------]
                          [ 1  1 | 0  0  0 ]    ← Item 1
                          [ 0  1 | 0  0  0 ]    ← Item 2
                          [ 1  0 | 0  0  0 ]    ← Item 3
```

---

## 3.8 Neighborhood and Degree

The **neighborhood** $\mathcal{N}(v)$ of a node $v$ is the set of nodes directly connected to $v$.

```
User 1 ── Product A
   │
   └───── Product C

N(User 1) = {Product A, Product C}
```

The **degree** $d(v)$ of a node is the number of edges connected to it: $d(v) = |\mathcal{N}(v)|$.

```
d(User 1) = 2     (connected to 2 products)
d(Product A) = 2  (connected to 2 users)
```

The **degree matrix** $D$ is a diagonal matrix:

$$D_{ii} = d(v_i)$$

The degree matrix will appear in the GCN formula in Phase 5.

### Multi-Hop Neighborhoods

This is where graphs become powerful for recommendation.

**1-hop neighbors** of User 1: items they directly interacted with.

```
User 1 ── Product A
   │
   └───── Product C
```

**2-hop neighbors** of User 1: users who also interacted with those items.

```
User 1 ── Product A ── User 2
   │
   └───── Product C ── User 3
```

**3-hop neighbors** of User 1: items those other users interacted with.

```
User 1 ── Product A ── User 2 ── Product B
   │                      │
   │                      └────── Product D
   │
   └───── Product C ── User 3 ── Product E
```

This is exactly how collaborative filtering works on a graph: User 1 might like Product B because User 2 (who shares taste via Product A) purchased it. GNNs will formalize this multi-hop information propagation.

---

## 3.9 Graph Sparsity

Real recommendation graphs are extremely **sparse**: most users interact with a tiny fraction of available items.

```
Netflix:
  ~200M users × ~15,000 titles
  = 3 trillion possible edges
  
  Average user watches ~100 titles
  = ~20 billion actual edges
  
  Sparsity = 1 - (20B / 3T) ≈ 99.3% sparse
```

Why sparsity matters:

1. **For MF:** Fewer observations mean less reliable embeddings.
2. **For graphs:** Sparse graphs have fewer paths between nodes, so multi-hop signals are weaker.
3. **For GNNs:** Graph propagation has less information to work with in sparse regions.
4. **For your BTP:** Graph sparsity is a potential research angle. How does your model perform as the graph gets sparser?

---

## 3.10 From Matrix to Graph — Full Example

Let's convert a complete interaction matrix to a graph representation.

**Starting point:** interaction matrix

```
           Item A   Item B   Item C   Item D
User 1  [    1        1        0        0   ]
User 2  [    1        0        1        1   ]
User 3  [    0        1        1        0   ]
```

**Step 1:** Identify edges (where value = 1)

```
(User 1, Item A), (User 1, Item B)
(User 2, Item A), (User 2, Item C), (User 2, Item D)
(User 3, Item B), (User 3, Item C)
```

**Step 2:** Draw bipartite graph

```
User 1 ──── Item A
  │            │
  │            │
  └──── Item B │
           │   │
User 2 ────┘   │
  │             │
  ├──── Item C  │
  │        │    │
  │        │    │
  └──── Item D  │
           │    │
User 3 ────┘    │
  │             │
  └─────────────┘
       (Item B)
```

Cleaner layout:

```
Users          Items
              
U1 ─────────── A
U1 ─────────── B
U2 ─────────── A
U2 ─────────── C
U2 ─────────── D
U3 ─────────── B
U3 ─────────── C
```

**Step 3:** Full adjacency matrix $(3 + 4 = 7$ nodes$)$

```
       U1  U2  U3  A   B   C   D
U1  [   0   0   0   1   1   0   0  ]
U2  [   0   0   0   1   0   1   1  ]
U3  [   0   0   0   0   1   1   0  ]
A   [   1   1   0   0   0   0   0  ]
B   [   1   0   1   0   0   0   0  ]
C   [   0   1   1   0   0   0   0  ]
D   [   0   1   0   0   0   0   0  ]
```

**Step 4:** In PyTorch Geometric format (edge_index)

```python
edge_index = torch.tensor([
    [0, 0, 1, 1, 1, 2, 2,   3, 4, 3, 5, 6, 4, 5],  # source
    [3, 4, 3, 5, 6, 4, 5,   0, 0, 1, 1, 1, 2, 2]   # target
])
# (undirected: each edge appears twice)
```

This is exactly how you'll represent your recommendation graph for the GNN.

---

## Exercises

1. **Convert this matrix to a graph.** Draw the bipartite graph and write the adjacency matrix.

```
           Item X   Item Y   Item Z
User A  [    1        0        1   ]
User B  [    1        1        0   ]
User C  [    0        1        1   ]
```

2. **Find neighborhoods.** Using the graph from Exercise 1:
   - What are the 1-hop neighbors of User A?
   - What are the 2-hop neighbors of User A?
   - What items might you recommend to User A based on 3-hop paths?

3. **Compute degree.** For each node in your graph, compute the degree. Write the degree matrix $D$.

4. **Think about sparsity.** If you have 10,000 users and 50,000 items, and each user interacts with an average of 20 items, what fraction of the interaction matrix is filled? How many edges does the graph have?

---

## Key Takeaways

- A **graph** $G = (V, E)$ explicitly represents the structure of user-item interactions.
- The **adjacency matrix** $A$ encodes connections numerically.
- A **bipartite graph** $G = (U, I, E)$ is the natural representation: users and items are two node types, edges are interactions.
- The interaction matrix $R$ and the bipartite graph are two views of the same data.
- **Multi-hop neighborhoods** capture the collaborative filtering intuition: users connected through shared items have similar taste.
- **Sparsity** is a real challenge: most recommendation graphs are >99% sparse.
- Understanding the bipartite graph structure is essential — every GNN model in your BTP will operate on this graph.
