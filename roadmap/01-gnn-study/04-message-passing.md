# Phase 4 — Message Passing

**Week 3 | BTP Roadmap**

---

## 4.1 The Key Question

Before diving into equations, ask yourself this question:

> How can a user's representation be influenced by the items they interacted with?

In Matrix Factorization, each user gets an embedding that's learned independently via gradient descent. The only connection between users is through the shared loss function.

In a graph neural network, we do something different: we let **information flow along edges**. A user's embedding is directly shaped by the items they interacted with, and those items are shaped by the other users who interacted with them.

This "information flow" is called **message passing**, and it's the core paradigm of all GNNs.

---

## 4.2 Message Passing: The Intuition

Consider this small graph:

```
User A
 ├──── Product 1  (Skincare, $15)
 └──── Product 2  (Haircare, $25)
```

User A purchased Product 1 and Product 2. Message passing says:

```
Product 1 sends a "message" to User A:
    "I am a Skincare product priced at $15"

Product 2 sends a "message" to User A:
    "I am a Haircare product priced at $25"

User A receives both messages and updates its representation:
    "I am a user who buys Skincare and Haircare products"
```

Now consider a bigger picture:

```
User A ──── Product 1 ──── User B ──── Product 3
  │
  └──────── Product 2 ──── User C ──── Product 4
```

After ONE round of message passing:
- User A learns about Product 1 and Product 2 (direct neighbors)

After TWO rounds:
- User A also learns about User B and User C (through Product 1 and Product 2)

After THREE rounds:
- User A learns about Product 3 and Product 4 (through Users B and C)

Each round of message passing extends the "receptive field" by one hop.

---

## 4.3 The Three Steps

Every message passing round has three steps:

### Step 1: MESSAGE

Each neighbor creates a message to send.

$$\mathbf{m}_{j \to i} = \text{MSG}(\mathbf{h}_j, \mathbf{h}_i, \mathbf{e}_{ji})$$

Where:
- $\mathbf{h}_j$ = representation of the sending node
- $\mathbf{h}_i$ = representation of the receiving node
- $\mathbf{e}_{ji}$ = edge features (optional)

In the simplest case, the message is just the sender's representation:
$$\mathbf{m}_{j \to i} = \mathbf{h}_j$$

### Step 2: AGGREGATE

The node collects all incoming messages and combines them.

$$\mathbf{m}_i = \text{AGG}\left(\{\mathbf{m}_{j \to i} : j \in \mathcal{N}(i)\}\right)$$

Common aggregation functions:
- **Sum:** $\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
- **Mean:** $\mathbf{m}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$
- **Max:** $\mathbf{m}_i = \max_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}$

### Step 3: UPDATE

The node updates its own representation using the aggregated message.

$$\mathbf{h}_i^{(l+1)} = \text{UPDATE}(\mathbf{h}_i^{(l)}, \mathbf{m}_i)$$

Common update rules:
- **Replace:** $\mathbf{h}_i^{(l+1)} = \mathbf{m}_i$
- **Add:** $\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \mathbf{m}_i$
- **Neural network:** $\mathbf{h}_i^{(l+1)} = \sigma(\mathbf{W} \cdot [\mathbf{h}_i^{(l)} \| \mathbf{m}_i])$

Where $\sigma$ is an activation function (like ReLU) and $\|$ means concatenation.

---

## 4.4 Worked Example

Let's trace message passing on a concrete graph with real numbers.

### Setup

```
User A ──── Product 1
  │
  └──────── Product 2 ──── User B
```

Initial embeddings (randomly initialized, dimension $d = 3$):

```
h_A^(0) = [0.1, 0.2, 0.3]     (User A)
h_1^(0) = [0.5, 0.1, 0.4]     (Product 1)
h_2^(0) = [0.3, 0.6, 0.2]     (Product 2)
h_B^(0) = [0.4, 0.3, 0.5]     (User B)
```

### Round 1: Using MEAN aggregation and REPLACE update

**Messages to User A:**

```
From Product 1: m_{1→A} = h_1^(0) = [0.5, 0.1, 0.4]
From Product 2: m_{2→A} = h_2^(0) = [0.3, 0.6, 0.2]
```

**Aggregate (mean):**

$$\mathbf{m}_A = \frac{1}{2}\left([0.5, 0.1, 0.4] + [0.3, 0.6, 0.2]\right) = [0.4, 0.35, 0.3]$$

**Update (replace):**

$$\mathbf{h}_A^{(1)} = [0.4, 0.35, 0.3]$$

User A's new representation is the average of the products they purchased.

**Messages to Product 2:**

```
From User A: m_{A→2} = h_A^(0) = [0.1, 0.2, 0.3]
From User B: m_{B→2} = h_B^(0) = [0.4, 0.3, 0.5]
```

**Aggregate (mean):**

$$\mathbf{m}_2 = \frac{1}{2}\left([0.1, 0.2, 0.3] + [0.4, 0.3, 0.5]\right) = [0.25, 0.25, 0.4]$$

**Update:**

$$\mathbf{h}_2^{(1)} = [0.25, 0.25, 0.4]$$

Product 2's new representation is the average of the users who purchased it.

**After Round 1:**

```
h_A^(1) = [0.40, 0.35, 0.30]   ← influenced by Product 1 and Product 2
h_1^(1) = [0.10, 0.20, 0.30]   ← influenced by User A (only neighbor)
h_2^(1) = [0.25, 0.25, 0.40]   ← influenced by User A and User B
h_B^(1) = [0.30, 0.60, 0.20]   ← influenced by Product 2 (only neighbor)
```

### Round 2: Now signals travel further

In Round 2, we use the updated embeddings $h^{(1)}$.

**Messages to User A:**

```
From Product 1: m_{1→A} = h_1^(1) = [0.10, 0.20, 0.30]
From Product 2: m_{2→A} = h_2^(1) = [0.25, 0.25, 0.40]
```

$$\mathbf{h}_A^{(2)} = \frac{1}{2}\left([0.10, 0.20, 0.30] + [0.25, 0.25, 0.40]\right) = [0.175, 0.225, 0.35]$$

Notice: User A's embedding at layer 2 now contains information from User B (because Product 2 absorbed User B's information in Round 1). **User A has never directly connected to User B, but through 2-hop message passing, User B's signal reaches User A.**

This is the power of message passing.

---

## 4.5 Multi-Hop Relationships and Recommendation

Let's connect this to your BTP goal.

### 1-Hop: Direct Interactions

```
User A ── Product 1
User A ── Product 2
```

After 1 round: User A's embedding captures "what products I purchased."

### 2-Hop: Similar Users

```
User A ── Product 1 ── User B
User A ── Product 2 ── User C
```

After 2 rounds: User A's embedding captures "what other users who buy similar products are like."

### 3-Hop: Recommendations

```
User A ── Product 1 ── User B ── Product 3
                                ↑
                     Potential recommendation!
```

After 3 rounds: User A's embedding has absorbed information about Product 3, even though User A never interacted with it. If User A's embedding is now similar to Product 3's embedding, the system will recommend Product 3.

This is collaborative filtering through graph propagation.

---

## 4.6 Visualizing the Flow

Here's a step-by-step diagram of information flow for 2 layers:

```
LAYER 0 (initial)          LAYER 1                    LAYER 2
─────────────────          ────────                    ────────

  [User A]                   [User A]                   [User A]
   ↑ gets:                    ↑ gets:                     now contains:
   nothing yet                Product 1's Layer 0           Product 1's info
                              Product 2's Layer 0           Product 2's info
                                                          User B's info (via Product 2)
                                                          User C's info (via Product 1)

  [Product 1]                  [Product 1]                  [Product 1]
   ↑ gets:                    ↑ gets:                     now contains:
   nothing yet                User A's Layer 0            User A's info
                              User C's Layer 0            User C's info
                                                          Product 2's info (via User A)

  ...and so on for every node
```

Each layer adds one hop of context. This is why the **number of GNN layers** is a key hyperparameter:

```
Layers    Context captured
──────    ───────────────────────────────────
  1       Direct neighbors (items you watched)
  2       2-hop (similar users)
  3       3-hop (items similar users watched)
  4       4-hop (getting very distant)
  5+      Risk of oversmoothing (all nodes look alike)
```

For recommendation, 2–3 layers is typically optimal. You'll experiment with this in your BTP.

---

## 4.7 Why This Is the Bridge to GNNs

Message passing is the **general framework**. Specific GNN architectures differ in HOW they implement the three steps:

```
Architecture    Message             Aggregate    Update
─────────────────────────────────────────────────────────────
GCN             Normalized h_j      Sum          Linear + σ
GraphSAGE       h_j                 Mean/Max     Concat + Linear
GAT             Attention-weighted  Sum          Linear + σ
LightGCN        Normalized h_j      Sum          Weighted average across layers
```

In the next phase, you'll learn GCN — the first and most foundational GNN. But every architecture follows this same message-aggregate-update pattern.

---

## 4.8 Pseudocode

Here's message passing in generic pseudocode:

```python
# h: dict of node embeddings {node_id: vector}
# G: graph with adjacency information
# L: number of message passing layers

for l in range(L):
    h_new = {}
    for node in G.nodes:
        # Collect messages from neighbors
        messages = []
        for neighbor in G.neighbors(node):
            msg = MESSAGE(h[neighbor], h[node])
            messages.append(msg)
        
        # Aggregate messages
        aggregated = AGGREGATE(messages)
        
        # Update node representation
        h_new[node] = UPDATE(h[node], aggregated)
    
    h = h_new  # All nodes update simultaneously

# After L rounds, h contains the final embeddings
```

In practice, GNN frameworks like PyTorch Geometric implement this efficiently using matrix operations on the full graph, not node-by-node loops.

---

## Exercises

1. **Trace by hand.** Given this graph and initial embeddings ($d = 2$), compute one round of mean-aggregation message passing for all nodes:

```
A ── B ── C

h_A^(0) = [1.0, 0.0]
h_B^(0) = [0.0, 1.0]
h_C^(0) = [0.5, 0.5]
```

Use the rule: $h_i^{(1)} = \text{mean}(\{h_j^{(0)} : j \in \mathcal{N}(i)\})$

2. **2-hop signal.** Using the same graph, compute a second round of message passing. Verify that node A's embedding at layer 2 contains information from node C (which is 2 hops away).

3. **Think about recommendation.** In the following graph, after 2 rounds of message passing, will User 1's embedding contain information about Product C? Trace the path.

```
User 1 ── Product A ── User 2 ── Product C
   │
   └───── Product B
```

4. **Aggregation choice.** Why might mean aggregation be better than sum aggregation for nodes with very different degrees? (Hint: think about a user who purchased 2 products vs one who purchased 200.)

---

## Key Takeaways

- **Message passing** is the core mechanism of all GNNs: nodes send information to their neighbors, who aggregate and update their representations.
- Three steps per round: **Message → Aggregate → Update**.
- Each round extends the receptive field by one hop. After $L$ rounds, a node has absorbed information from its $L$-hop neighborhood.
- For recommendation, this means: users learn from items, items learn from users, and after multiple rounds, the collaborative signal propagates through the graph.
- 2–3 message passing layers is typically enough for recommendation — more layers risk **oversmoothing**.
- All GNN architectures (GCN, GraphSAGE, GAT, LightGCN) are specific implementations of message passing — they differ in how they compute messages, aggregate, and update.
