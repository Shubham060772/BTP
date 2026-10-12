"""Legacy prototype retained only for audit history.

Do not use this script for experiments. Its Amazon-Book sequences have no
timestamps, and its former SASRec/hybrid training protocol contained target
leakage and detached graph representations. Use ``prepare_amazon_beauty.py``
and ``run_beauty.py`` instead.
"""
import os
import sys
import time
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, eye as sp_eye
from scipy.sparse import diags as sp_diags
from collections import defaultdict
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'roadmap', 'data', 'amazon-book')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'logs'), exist_ok=True)

# ─── Config ───
EMBED_DIM = 64
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 15
L2_REG = 1e-4
LIGHTGCN_LAYERS = 3
SASREC_LAYERS = 2
SASREC_HEADS = 1
MAX_SEQ_LEN = 50
DROPOUT = 0.2
SEEDS = [42]
K_VALUES = [10, 20]
EVAL_EVERY = 5
NUM_NEG_EVAL = 99
VAL_SAMPLE_USERS = 1500
TEST_SAMPLE_USERS = 3000
MAX_TRAIN_SAMPLES = 500000  # subsample training pairs per epoch for CPU speed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_amazon_book():
    log("Loading Amazon-Book dataset...")
    train_file = os.path.join(DATA_DIR, 'train.txt')
    test_file = os.path.join(DATA_DIR, 'test.txt')

    train_dict = {}
    num_users = 0
    num_items = 0

    with open(train_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            uid = parts[0]
            items = parts[1:]
            train_dict[uid] = items
            num_users = max(num_users, uid + 1)
            if items:
                num_items = max(num_items, max(items) + 1)

    test_dict = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            uid = parts[0]
            items = parts[1:]
            test_dict[uid] = items
            if items:
                num_items = max(num_items, max(items) + 1)

    # Create validation: hold out last item from training for users with >= 5 items
    val_dict = {}
    train_final = {}
    for uid, items in train_dict.items():
        if len(items) >= 5:
            val_dict[uid] = [items[-1]]
            train_final[uid] = items[:-1]
        else:
            train_final[uid] = items

    # Count interactions
    n_train = sum(len(v) for v in train_final.values())
    n_val = sum(len(v) for v in val_dict.values())
    n_test = sum(len(v) for v in test_dict.values())

    log(f"  Users: {num_users}, Items: {num_items}")
    log(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    log(f"  Density: {n_train / (num_users * num_items) * 100:.4f}%")

    return train_final, val_dict, test_dict, num_users, num_items


def build_adj_matrix(train_dict, num_users, num_items):
    """Build normalized adjacency for LightGCN: D^{-1/2} A D^{-1/2}"""
    rows, cols = [], []
    for uid, items in train_dict.items():
        for iid in items:
            rows.append(uid)
            cols.append(num_users + iid)  # offset items
            rows.append(num_users + iid)  # reverse edge
            cols.append(uid)

    n = num_users + num_items
    vals = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # D^{-1/2} A D^{-1/2}
    degrees = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp_diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt

    # To torch sparse
    coo = norm_adj.tocoo()
    indices = torch.LongTensor(np.stack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(DEVICE)


def build_train_set(train_dict):
    """Build flat list of (user, item) pairs for BPR training."""
    pairs = []
    for uid, items in train_dict.items():
        for iid in items:
            pairs.append((uid, iid))
    return pairs


def build_user_item_set(train_dict):
    """Build set of items per user for negative sampling."""
    user_items = defaultdict(set)
    for uid, items in train_dict.items():
        user_items[uid] = set(items)
    return user_items


def build_sequences(train_dict, max_len=MAX_SEQ_LEN):
    """Build interaction sequences per user (list order as implicit sequence)."""
    sequences = {}
    for uid, items in train_dict.items():
        seq = items[-max_len:]  # take last max_len items
        sequences[uid] = seq
    return sequences


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_sampled(model, test_dict, train_user_items, num_items, k_values,
                     sequences=None, adj=None, num_neg=NUM_NEG_EVAL,
                     max_users=None):
    """Evaluate with sampled negatives for faster CPU execution."""
    model.eval()
    metrics = {f'Recall@{k}': [] for k in k_values}
    metrics.update({f'NDCG@{k}': [] for k in k_values})
    metrics.update({f'HR@{k}': [] for k in k_values})

    all_items_list = list(range(num_items))

    # Sample users for speed
    eval_users = list(test_dict.keys())
    if max_users and len(eval_users) > max_users:
        eval_users = random.sample(eval_users, max_users)

    # Cache graph embeddings once for all eval users
    _cache_graph(model)

    with torch.no_grad():
        for uid in eval_users:
            test_items = test_dict[uid]
            if not test_items:
                continue

            train_items = train_user_items.get(uid, set())
            # Fast negative sampling
            neg_items = []
            test_set_local = set(test_items)
            while len(neg_items) < num_neg:
                cand = random.choice(all_items_list)
                if cand not in train_items and cand not in test_set_local:
                    neg_items.append(cand)

            eval_items = test_items + neg_items

            # Get scores
            item_tensor = torch.LongTensor(eval_items).to(DEVICE)
            user_tensor = torch.LongTensor([uid]).to(DEVICE)

            scores = get_scores(model, user_tensor, item_tensor, uid, sequences, adj)
            scores = scores.cpu().numpy()

            # Rank
            ranked_indices = np.argsort(-scores)
            ranked_items = [eval_items[i] for i in ranked_indices]

            for k in k_values:
                topk = ranked_items[:k]
                hits = len(set(topk) & test_set_local)

                metrics[f'Recall@{k}'].append(hits / min(len(test_set_local), k))
                metrics[f'HR@{k}'].append(1.0 if hits > 0 else 0.0)

                dcg = 0.0
                for j, item in enumerate(topk):
                    if item in test_set_local:
                        dcg += 1.0 / np.log2(j + 2)
                ideal = sum(1.0 / np.log2(j + 2) for j in range(min(len(test_set_local), k)))
                metrics[f'NDCG@{k}'].append(dcg / ideal if ideal > 0 else 0.0)

    _clear_graph(model)

    result = {k: np.mean(v) for k, v in metrics.items() if v}
    return result


def get_scores(model, user_tensor, item_tensor, uid, sequences, adj):
    """Get prediction scores for a model (handles different model types)."""
    if isinstance(model, AdaFuse):
        seq = sequences.get(uid, [])
        seq_padded = [0] * (MAX_SEQ_LEN - len(seq)) + seq[-MAX_SEQ_LEN:]
        seq_tensor = torch.LongTensor([seq_padded]).to(DEVICE)
        return model.predict(user_tensor, item_tensor, seq_tensor)
    elif isinstance(model, SASRec):
        seq = sequences.get(uid, [])
        seq_padded = [0] * (MAX_SEQ_LEN - len(seq)) + seq[-MAX_SEQ_LEN:]
        seq_tensor = torch.LongTensor([seq_padded]).to(DEVICE)
        return model.predict(seq_tensor, item_tensor)
    elif isinstance(model, SASGCN):
        seq = sequences.get(uid, [])
        seq_padded = [0] * (MAX_SEQ_LEN - len(seq)) + seq[-MAX_SEQ_LEN:]
        seq_tensor = torch.LongTensor([seq_padded]).to(DEVICE)
        return model.predict(user_tensor, item_tensor, seq_tensor)
    elif isinstance(model, (LightGCN, MFBPR)):
        return model.predict(user_tensor, item_tensor)
    else:
        return model.predict(user_tensor, item_tensor)


# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════

class MFBPR(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        pi = self.item_emb(pos_items)
        ni = self.item_emb(neg_items)
        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)
        return pos_scores, neg_scores, u, pi, ni

    def predict(self, users, items):
        u = self.user_emb(users)  # (1, d)
        i = self.item_emb(items)  # (n, d)
        return (u * i).sum(dim=1)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers, adj):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.adj = adj
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        self._cached_emb = None

    def _compute_all_embeddings(self):
        e = self.embedding.weight
        embs = [e]
        for _ in range(self.num_layers):
            e = torch.sparse.mm(self.adj, e)
            embs.append(e)
        return torch.stack(embs, dim=0).mean(dim=0)

    def get_all_embeddings(self):
        if self._cached_emb is not None:
            return self._cached_emb
        return self._compute_all_embeddings()

    def cache_embeddings(self, detach=True):
        emb = self._compute_all_embeddings()
        self._cached_emb = emb.detach() if detach else emb

    def clear_cache(self):
        self._cached_emb = None

    def forward(self, users, pos_items, neg_items):
        all_emb = self.get_all_embeddings()
        u = all_emb[users]
        pi = all_emb[self.num_users + pos_items]
        ni = all_emb[self.num_users + neg_items]
        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)
        # Regularize on layer-0 embeddings
        u0 = self.embedding(users)
        pi0 = self.embedding(self.num_users + pos_items)
        ni0 = self.embedding(self.num_users + neg_items)
        return pos_scores, neg_scores, u0, pi0, ni0

    def predict(self, users, items):
        all_emb = self.get_all_embeddings()
        u = all_emb[users]
        i = all_emb[self.num_users + items]
        return (u * i).sum(dim=1)

    def get_user_embedding(self, users):
        all_emb = self.get_all_embeddings()
        return all_emb[users]

    def get_item_embedding(self, items):
        all_emb = self.get_all_embeddings()
        return all_emb[self.num_users + items]


class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim, max_len, num_heads, num_layers, dropout):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)  # 0 = padding
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        nn.init.xavier_normal_(self.item_emb.weight[1:])  # skip padding

    def forward(self, seq):
        """seq: (B, L) item IDs, 0 = padding"""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.layer_norm(self.dropout(x))

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()
        # Padding mask
        pad_mask = (seq == 0)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return x  # (B, L, d)

    def get_seq_embedding(self, seq):
        """Get the last-position output as user sequential embedding."""
        out = self.forward(seq)  # (B, L, d)
        # Find last non-padding position
        lengths = (seq != 0).sum(dim=1) - 1  # (B,)
        lengths = lengths.clamp(min=0)
        return out[torch.arange(out.size(0), device=out.device), lengths]  # (B, d)

    def predict(self, seq, items):
        """Score items given a sequence."""
        user_emb = self.get_seq_embedding(seq)  # (1, d)
        item_emb = self.item_emb(items + 1)  # +1 because 0 is padding; (n, d)
        return (user_emb * item_emb).sum(dim=1)


class SASGCN(nn.Module):
    """SASGCN comparator: vanilla GCN + SASRec with concat+MLP fusion."""
    def __init__(self, num_users, num_items, embed_dim, adj, max_len,
                 num_heads, sasrec_layers, gcn_layers, dropout):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.adj = adj

        # GCN branch (vanilla with transforms)
        self.gcn_embedding = nn.Embedding(num_users + num_items, embed_dim)
        self.gcn_layers_list = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(gcn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(gcn_layers)
        ])

        # SASRec branch
        self.sasrec = SASRec(num_items, embed_dim, max_len, num_heads, sasrec_layers, dropout)

        # Fusion: concat + MLP
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        nn.init.xavier_normal_(self.gcn_embedding.weight)
        self._cached_gcn = None

    def _compute_gcn_embeddings(self):
        e = self.gcn_embedding.weight
        for linear, norm in zip(self.gcn_layers_list, self.gcn_norms):
            e = torch.sparse.mm(self.adj, e)
            e = norm(F.relu(linear(e)))
        return e

    def get_gcn_embeddings(self):
        if self._cached_gcn is not None:
            return self._cached_gcn
        return self._compute_gcn_embeddings()

    def cache_gcn_embeddings(self):
        self._cached_gcn = self._compute_gcn_embeddings().detach()

    def clear_gcn_cache(self):
        self._cached_gcn = None

    def forward(self, users, pos_items, neg_items, seq):
        gcn_all = self.get_gcn_embeddings()
        gcn_user = gcn_all[users]
        gcn_pos = gcn_all[self.num_users + pos_items]
        gcn_neg = gcn_all[self.num_users + neg_items]

        seq_emb = self.sasrec.get_seq_embedding(seq)
        fused_user = self.fusion(torch.cat([seq_emb, gcn_user], dim=1))

        pos_scores = (fused_user * gcn_pos).sum(dim=1)
        neg_scores = (fused_user * gcn_neg).sum(dim=1)
        # L2 reg on raw embeddings (not propagated) so gradients flow even with cached GCN
        u0 = self.gcn_embedding(users)
        pi0 = self.gcn_embedding(self.num_users + pos_items)
        ni0 = self.gcn_embedding(self.num_users + neg_items)
        return pos_scores, neg_scores, u0, pi0, ni0

    def predict(self, users, items, seq):
        gcn_all = self.get_gcn_embeddings()
        gcn_user = gcn_all[users]
        gcn_items = gcn_all[self.num_users + items]

        seq_emb = self.sasrec.get_seq_embedding(seq)
        fused_user = self.fusion(torch.cat([seq_emb, gcn_user], dim=1))
        return (fused_user * gcn_items).sum(dim=1)


class AdaFuse(nn.Module):
    """AdaFuse: LightGCN + SASRec with attention fusion gate."""
    def __init__(self, num_users, num_items, embed_dim, adj, num_gcn_layers,
                 max_len, num_heads, sasrec_layers, dropout):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # LightGCN branch
        self.lightgcn = LightGCN(num_users, num_items, embed_dim, num_gcn_layers, adj)

        # SASRec branch (shares item embeddings with LightGCN conceptually,
        # but SASRec uses padding_idx=0 so we keep separate and let gradients flow)
        self.sasrec = SASRec(num_items, embed_dim, max_len, num_heads, sasrec_layers, dropout)

        # Attention fusion gate: [e_seq; e_col; e_seq * e_col] -> scalar alpha
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def cache_graph_embeddings(self, detach=True):
        self.lightgcn.cache_embeddings(detach=detach)

    def clear_graph_cache(self):
        self.lightgcn.clear_cache()

    def fuse(self, e_seq, e_col):
        """Compute fusion: alpha * e_seq + (1 - alpha) * e_col"""
        gate_input = torch.cat([e_seq, e_col, e_seq * e_col], dim=1)
        alpha = self.gate(gate_input)  # (B, 1)
        fused = alpha * e_seq + (1 - alpha) * e_col
        return fused, alpha

    def forward(self, users, pos_items, neg_items, seq):
        # LightGCN embeddings
        lgcn_all = self.lightgcn.get_all_embeddings()
        e_col_user = lgcn_all[users]
        e_col_pos = lgcn_all[self.num_users + pos_items]
        e_col_neg = lgcn_all[self.num_users + neg_items]

        # SASRec embedding
        e_seq = self.sasrec.get_seq_embedding(seq)

        # Fuse user representation
        e_user, alpha = self.fuse(e_seq, e_col_user)

        # Score using LightGCN item embeddings
        pos_scores = (e_user * e_col_pos).sum(dim=1)
        neg_scores = (e_user * e_col_neg).sum(dim=1)

        # Regularization embeddings (layer-0)
        u0 = self.lightgcn.embedding(users)
        pi0 = self.lightgcn.embedding(self.num_users + pos_items)
        ni0 = self.lightgcn.embedding(self.num_users + neg_items)

        return pos_scores, neg_scores, u0, pi0, ni0, alpha

    def predict(self, users, items, seq):
        lgcn_all = self.lightgcn.get_all_embeddings()
        e_col_user = lgcn_all[users]
        e_col_items = lgcn_all[self.num_users + items]

        e_seq = self.sasrec.get_seq_embedding(seq)
        e_user, alpha = self.fuse(e_seq, e_col_user)
        return (e_user * e_col_items).sum(dim=1)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

class BPRDataset(Dataset):
    def __init__(self, pairs, user_items, num_items, sequences=None, max_len=MAX_SEQ_LEN):
        self.pairs = pairs
        self.user_items = user_items
        self.num_items = num_items
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, pos_iid = self.pairs[idx]
        # Sample negative
        neg_iid = random.randint(0, self.num_items - 1)
        while neg_iid in self.user_items[uid]:
            neg_iid = random.randint(0, self.num_items - 1)

        result = [uid, pos_iid, neg_iid]

        if self.sequences is not None:
            seq = self.sequences.get(uid, [])
            seq_padded = [0] * (self.max_len - len(seq)) + seq[-self.max_len:]
            result.append(seq_padded)

        return result


def collate_bpr(batch):
    users = torch.LongTensor([b[0] for b in batch])
    pos = torch.LongTensor([b[1] for b in batch])
    neg = torch.LongTensor([b[2] for b in batch])
    if len(batch[0]) > 3:
        seqs = torch.LongTensor([b[3] for b in batch])
        return users, pos, neg, seqs
    return users, pos, neg


def _cache_graph(model):
    """Cache graph embeddings (detached) for models with GCN branches."""
    if isinstance(model, AdaFuse):
        model.cache_graph_embeddings(detach=True)
    elif isinstance(model, SASGCN):
        model.cache_gcn_embeddings()
    elif isinstance(model, LightGCN):
        model.cache_embeddings(detach=True)


def _clear_graph(model):
    """Clear cached graph embeddings."""
    if isinstance(model, AdaFuse):
        model.clear_graph_cache()
    elif isinstance(model, SASGCN):
        model.clear_gcn_cache()
    elif isinstance(model, LightGCN):
        model.clear_cache()


def train_bpr_model(model, train_pairs, user_items, num_items, val_dict,
                    train_user_items, sequences, adj, name, seed):
    """Generic BPR training loop with early stopping."""
    set_seed(seed)
    has_seq = isinstance(model, (SASRec, SASGCN, AdaFuse))
    is_pure_lightgcn = isinstance(model, LightGCN) and not isinstance(model, (SASGCN, AdaFuse))
    has_cached_graph = isinstance(model, (SASGCN, AdaFuse))

    # Subsample training pairs for CPU speed
    if MAX_TRAIN_SAMPLES and len(train_pairs) > MAX_TRAIN_SAMPLES:
        epoch_pairs = random.sample(train_pairs, MAX_TRAIN_SAMPLES)
        log(f"  [{name}] Subsampled {MAX_TRAIN_SAMPLES}/{len(train_pairs)} training pairs")
    else:
        epoch_pairs = train_pairs

    # LightGCN uses full-batch (one graph propagation per epoch with gradients)
    effective_batch = len(epoch_pairs) if is_pure_lightgcn else BATCH_SIZE

    dataset = BPRDataset(epoch_pairs, user_items, num_items,
                         sequences=sequences if has_seq else None)
    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True,
                        collate_fn=collate_bpr, num_workers=0, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    best_ndcg = 0.0
    best_state = None
    patience_counter = 0
    train_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0

        # Cache graph embeddings (detached) for hybrid models (SASGCN/AdaFuse)
        # to avoid recomputing sparse matmul on every batch.
        # Pure LightGCN uses full-batch with gradients instead.
        if has_cached_graph:
            _cache_graph(model)

        for batch in loader:
            if has_seq:
                users, pos, neg, seqs = [b.to(DEVICE) for b in batch]
            else:
                users, pos, neg = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()

            if isinstance(model, AdaFuse):
                pos_scores, neg_scores, u0, pi0, ni0, alpha = model(users, pos, neg, seqs)
            elif isinstance(model, SASGCN):
                pos_scores, neg_scores, u0, pi0, ni0 = model(users, pos, neg, seqs)
            elif isinstance(model, SASRec):
                seq_emb = model.get_seq_embedding(seqs)
                pi_emb = model.item_emb(pos + 1)
                ni_emb = model.item_emb(neg + 1)
                pos_scores = (seq_emb * pi_emb).sum(dim=1)
                neg_scores = (seq_emb * ni_emb).sum(dim=1)
                u0 = seq_emb
                pi0 = pi_emb
                ni0 = ni_emb
            else:
                pos_scores, neg_scores, u0, pi0, ni0 = model(users, pos, neg)

            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = L2_REG * (u0.norm(2).pow(2) + pi0.norm(2).pow(2) + ni0.norm(2).pow(2)) / u0.size(0)
            loss = bpr_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        if has_cached_graph:
            _clear_graph(model)

        avg_loss = total_loss / max(n_batch, 1)
        train_losses.append(avg_loss)

        # Validation
        if epoch % EVAL_EVERY == 0:
            val_metrics = evaluate_sampled(model, val_dict, train_user_items,
                                           num_items, [10], sequences, adj,
                                           num_neg=99, max_users=VAL_SAMPLE_USERS)
            val_ndcg = val_metrics.get('NDCG@10', 0.0)
            log(f"  [{name}] Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f}")

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += EVAL_EVERY

            if patience_counter >= PATIENCE:
                log(f"  [{name}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, best_ndcg, train_losses


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run_experiment(seed, train_dict, val_dict, test_dict, num_users, num_items):
    log(f"\n{'='*60}")
    log(f"SEED: {seed}")
    log(f"{'='*60}")
    set_seed(seed)

    train_pairs = build_train_set(train_dict)
    user_items = build_user_item_set(train_dict)
    sequences = build_sequences(train_dict)
    adj = build_adj_matrix(train_dict, num_users, num_items)

    results = {}

    # ── 1. MF-BPR ──
    log("\n--- Training MF-BPR ---")
    model_mf = MFBPR(num_users, num_items, EMBED_DIM).to(DEVICE)
    model_mf, _, _ = train_bpr_model(
        model_mf, train_pairs, user_items, num_items,
        val_dict, user_items, sequences, adj, "MF-BPR", seed)
    results['MF-BPR'] = evaluate_sampled(
        model_mf, test_dict, user_items, num_items, K_VALUES, sequences, adj,
        max_users=TEST_SAMPLE_USERS)
    log(f"  MF-BPR test: {results['MF-BPR']}")
    del model_mf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 2. LightGCN ──
    log("\n--- Training LightGCN ---")
    model_lgcn = LightGCN(num_users, num_items, EMBED_DIM, LIGHTGCN_LAYERS, adj).to(DEVICE)
    model_lgcn, _, _ = train_bpr_model(
        model_lgcn, train_pairs, user_items, num_items,
        val_dict, user_items, sequences, adj, "LightGCN", seed)
    results['LightGCN'] = evaluate_sampled(
        model_lgcn, test_dict, user_items, num_items, K_VALUES, sequences, adj,
        max_users=TEST_SAMPLE_USERS)
    log(f"  LightGCN test: {results['LightGCN']}")
    del model_lgcn

    # ── 3. SASRec ──
    log("\n--- Training SASRec ---")
    model_sas = SASRec(num_items, EMBED_DIM, MAX_SEQ_LEN, SASREC_HEADS,
                       SASREC_LAYERS, DROPOUT).to(DEVICE)
    model_sas, _, _ = train_bpr_model(
        model_sas, train_pairs, user_items, num_items,
        val_dict, user_items, sequences, adj, "SASRec", seed)
    results['SASRec'] = evaluate_sampled(
        model_sas, test_dict, user_items, num_items, K_VALUES, sequences, adj,
        max_users=TEST_SAMPLE_USERS)
    log(f"  SASRec test: {results['SASRec']}")
    del model_sas

    # ── 4. SASGCN ──
    log("\n--- Training SASGCN ---")
    model_sasgcn = SASGCN(num_users, num_items, EMBED_DIM, adj, MAX_SEQ_LEN,
                          SASREC_HEADS, SASREC_LAYERS, 2, DROPOUT).to(DEVICE)
    model_sasgcn, _, _ = train_bpr_model(
        model_sasgcn, train_pairs, user_items, num_items,
        val_dict, user_items, sequences, adj, "SASGCN", seed)
    results['SASGCN'] = evaluate_sampled(
        model_sasgcn, test_dict, user_items, num_items, K_VALUES, sequences, adj,
        max_users=TEST_SAMPLE_USERS)
    log(f"  SASGCN test: {results['SASGCN']}")
    del model_sasgcn

    # ── 5. AdaFuse ──
    log("\n--- Training AdaFuse ---")
    model_af = AdaFuse(num_users, num_items, EMBED_DIM, adj, LIGHTGCN_LAYERS,
                       MAX_SEQ_LEN, SASREC_HEADS, SASREC_LAYERS, DROPOUT).to(DEVICE)
    model_af, _, _ = train_bpr_model(
        model_af, train_pairs, user_items, num_items,
        val_dict, user_items, sequences, adj, "AdaFuse", seed)
    results['AdaFuse'] = evaluate_sampled(
        model_af, test_dict, user_items, num_items, K_VALUES, sequences, adj,
        max_users=TEST_SAMPLE_USERS)
    log(f"  AdaFuse test: {results['AdaFuse']}")

    # Extract alpha values for analysis
    model_af.eval()
    model_af.cache_graph_embeddings(detach=True)
    alphas = {}
    with torch.no_grad():
        lgcn_all = model_af.lightgcn.get_all_embeddings()
        for uid in list(test_dict.keys())[:2000]:
            seq = sequences.get(uid, [])
            seq_padded = [0] * (MAX_SEQ_LEN - len(seq)) + seq[-MAX_SEQ_LEN:]
            seq_t = torch.LongTensor([seq_padded]).to(DEVICE)

            e_col = lgcn_all[uid].unsqueeze(0)
            e_seq = model_af.sasrec.get_seq_embedding(seq_t)
            _, alpha = model_af.fuse(e_seq, e_col)
            alphas[uid] = alpha.item()
    model_af.clear_graph_cache()

    del model_af

    return results, alphas


def main():
    raise SystemExit(
        "src/run_all.py is an invalidated prototype. Run "
        "src/prepare_amazon_beauty.py and src/run_beauty.py instead."
    )
    log("AdaFuse — Full Training Pipeline")
    log(f"Device: {DEVICE}")
    log(f"Seeds: {SEEDS}")

    train_dict, val_dict, test_dict, num_users, num_items = load_amazon_book()

    all_results = {s: {} for s in SEEDS}
    all_alphas = {}

    for seed in SEEDS:
        results, alphas = run_experiment(seed, train_dict, val_dict, test_dict, num_users, num_items)
        all_results[seed] = results
        all_alphas[seed] = alphas

    # ── Aggregate results ──
    log(f"\n{'='*60}")
    log("FINAL RESULTS (mean +/- std across seeds)")
    log(f"{'='*60}")

    models = ['MF-BPR', 'LightGCN', 'SASRec', 'SASGCN', 'AdaFuse']
    metrics = [f'{m}@{k}' for m in ['Recall', 'NDCG', 'HR'] for k in K_VALUES]

    header = f"{'Model':<12}" + "".join(f"{m:<18}" for m in metrics)
    log(header)
    log("-" * len(header))

    result_rows = []
    for model_name in models:
        row = [model_name]
        for metric in metrics:
            vals = [all_results[s].get(model_name, {}).get(metric, 0.0) for s in SEEDS]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            row.append(f"{mean_val:.4f}+/-{std_val:.4f}")
        result_rows.append(row)
        log(f"{row[0]:<12}" + "".join(f"{v:<18}" for v in row[1:]))

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'tables', 'main_results.txt')
    with open(results_path, 'w') as f:
        f.write(header + '\n')
        f.write("-" * len(header) + '\n')
        for row in result_rows:
            f.write(f"{row[0]:<12}" + "".join(f"{v:<18}" for v in row[1:]) + '\n')

    # Save alpha analysis
    log("\n--- Alpha Analysis (AdaFuse attention weights) ---")
    user_items_count = {uid: len(items) for uid, items in train_dict.items()}
    for seed in SEEDS:
        alphas = all_alphas[seed]
        if alphas:
            groups = {'1-5': [], '6-15': [], '16-30': [], '30+': []}
            for uid, alpha in alphas.items():
                n = user_items_count.get(uid, 0)
                if n <= 5: groups['1-5'].append(alpha)
                elif n <= 15: groups['6-15'].append(alpha)
                elif n <= 30: groups['16-30'].append(alpha)
                else: groups['30+'].append(alpha)

            log(f"  Seed {seed}:")
            for g, vals in groups.items():
                if vals:
                    log(f"    Users [{g:>5}] ({len(vals):>4} users): alpha = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save all results as pickle for notebook analysis
    save_path = os.path.join(RESULTS_DIR, 'all_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'results': all_results, 'alphas': all_alphas}, f)
    log(f"\nResults saved to {save_path}")
    log("Done!")


if __name__ == '__main__':
    main()
