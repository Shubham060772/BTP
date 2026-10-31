"""Train reproducible recommendation baselines and fusion models on Beauty.

This is a pilot runner. It uses timestamp-ordered sequences, a training-only
graph, fixed sampled-ranking candidates, validation NDCG@10 for selection, and
checkpoints after every model. Fusion heads are trained over frozen pretrained
LightGCN and SASRec encoders so graph representations are not silently detached
during supposed end-to-end training.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    PROJECT_ROOT
    / "roadmap"
    / "data"
    / "amazon-beauty"
    / "processed"
    / "beauty_5core.pkl"
)
RESULTS_ROOT = PROJECT_ROOT / "results" / "beauty"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.file = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument(
        "--ranking",
        choices=("full", "sampled"),
        default="full",
        help="Full eligible-item ranking or one-positive sampled ranking.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("popularity", "mf", "lightgcn", "sasrec", "sasgcn", "concat", "adafuse"),
        default=("popularity", "mf", "lightgcn", "sasrec", "sasgcn", "concat", "adafuse"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--alpha-reg",
        type=float,
        default=0.0,
        help="L1 penalty on gate alpha to push toward graph-only (residual gate).",
    )
    parser.add_argument(
        "--graph-loss-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary BPR loss on pure graph embeddings (multi-task).",
    )
    parser.add_argument(
        "--seq-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers in the sequence encoder.",
    )
    parser.add_argument(
        "--graph-enriched",
        action="store_true",
        help="Feed propagated (graph-enriched) embeddings into the Transformer.",
    )
    parser.add_argument(
        "--length-aware",
        action="store_true",
        help="Modulate gate alpha by learned confidence based on sequence length.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help=(
            "Epochs to train only the fusion gate (encoders frozen) before "
            "joint end-to-end training. Helps on dense datasets where joint "
            "training collapses (e.g. Amazon-Book)."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    required = {"num_users", "num_items", "train", "validation", "test"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"processed dataset is missing keys: {sorted(missing)}")
    return data


def build_graph(train: dict[int, list[int]], num_users: int, num_items: int):
    rows: list[int] = []
    columns: list[int] = []
    for user, items in train.items():
        for item in items:
            item_node = num_users + item
            rows.extend((user, item_node))
            columns.extend((item_node, user))

    interaction_count = sum(map(len, train.values()))
    assert len(rows) == 2 * interaction_count
    node_count = num_users + num_items
    adjacency = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(node_count, node_count),
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = np.power(degrees[nonzero], -0.5)
    normalized = diags(inverse_sqrt) @ adjacency @ diags(inverse_sqrt)
    coo = normalized.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, check_invariants=False
    ).coalesce().to(DEVICE)


def pad_sequence(items: list[int], max_len: int) -> tuple[list[int], int]:
    """Right-pad shifted item IDs; zero is reserved for padding."""
    items = items[-max_len:]
    shifted = [item + 1 for item in items]
    length = len(shifted)
    return shifted + [0] * (max_len - length), length


class GraphDataset(Dataset):
    def __init__(self, pairs, known_items, num_items, seed):
        self.pairs = pairs
        self.known_items = known_items
        self.num_items = num_items
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        user, positive = self.pairs[index]
        rng = random.Random(self.seed * 1_000_003 + index)
        negative = rng.randrange(self.num_items)
        while negative in self.known_items[user]:
            negative = rng.randrange(self.num_items)
        return user, positive, negative


class SequenceDataset(Dataset):
    def __init__(self, train, known_items, num_items, max_len, seed):
        self.train = train
        self.known_items = known_items
        self.num_items = num_items
        self.max_len = max_len
        self.seed = seed
        self.instances = [
            (user, target_index)
            for user, items in train.items()
            for target_index in range(1, len(items))
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        user, target_index = self.instances[index]
        history = self.train[user][:target_index]
        positive = self.train[user][target_index]
        rng = random.Random(self.seed * 1_000_003 + index)
        negative = rng.randrange(self.num_items)
        while negative in self.known_items[user]:
            negative = rng.randrange(self.num_items)
        sequence, length = pad_sequence(history, self.max_len)
        return user, torch.tensor(sequence, dtype=torch.long), length, positive, negative


def subset_dataset(dataset, maximum: int | None, seed: int):
    if maximum is None or len(dataset) <= maximum:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return torch.utils.data.Subset(dataset, indices)


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, dimension):
        super().__init__()
        self.user = nn.Embedding(num_users, dimension)
        self.item = nn.Embedding(num_items, dimension)
        nn.init.xavier_normal_(self.user.weight)
        nn.init.xavier_normal_(self.item.weight)

    def triplet(self, users, positives, negatives):
        return self.user(users), self.item(positives), self.item(negatives)

    def score(self, user, items, sequence=None, length=None):
        return (self.user(user) * self.item(items)).sum(-1)


class Popularity(nn.Module):
    """Non-personalized item-frequency sanity baseline."""

    def __init__(self, train, num_items):
        super().__init__()
        counts = torch.zeros(num_items, dtype=torch.float32)
        for items in train.values():
            for item in items:
                counts[item] += 1
        self.register_buffer("counts", counts)

    def score(self, user, items, sequence=None, length=None):
        return self.counts[items]


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, dimension, adjacency, layers=3):
        super().__init__()
        self.num_users = num_users
        self.adjacency = adjacency
        self.layers = layers
        self.embedding = nn.Embedding(num_users + num_items, dimension)
        nn.init.xavier_normal_(self.embedding.weight)

    def propagate(self):
        current = self.embedding.weight
        embeddings = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            embeddings.append(current)
        return torch.stack(embeddings).mean(0)

    def triplet(self, users, positives, negatives):
        all_embeddings = self.propagate()
        return (
            all_embeddings[users],
            all_embeddings[self.num_users + positives],
            all_embeddings[self.num_users + negatives],
        )

    def score(self, user, items, sequence=None, length=None, propagated=None):
        all_embeddings = self.propagate() if propagated is None else propagated
        return (
            all_embeddings[user]
            * all_embeddings[self.num_users + items]
        ).sum(-1)


class SASRec(nn.Module):
    def __init__(self, num_items, dimension, max_len, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.item = nn.Embedding(num_items + 1, dimension, padding_idx=0)
        self.position = nn.Embedding(max_len, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=2,
            dim_feedforward=4 * dimension,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dimension)
        nn.init.xavier_normal_(self.item.weight[1:])

    def encode(self, sequence, length):
        positions = torch.arange(self.max_len, device=sequence.device)[None, :]
        hidden = self.item(sequence) + self.position(positions)
        causal = torch.triu(
            torch.ones(self.max_len, self.max_len, device=sequence.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal,
            src_key_padding_mask=sequence.eq(0),
        )
        indices = (length - 1).clamp_min(0)
        output = hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        return self.norm(output)

    def triplet(self, sequence, length, positives, negatives):
        user = self.encode(sequence, length)
        return user, self.item(positives + 1), self.item(negatives + 1)

    def score(self, user, items, sequence, length):
        encoded = self.encode(sequence, length)
        return (encoded * self.item(items + 1)).sum(-1)


class SharedSASRecEncoder(nn.Module):
    """SASRec encoder whose item inputs come from a shared graph table.

    When ``graph_enriched=True``, ``encode`` expects a pre-built propagated
    embedding table and indexes into it instead of the raw node_embedding.
    This feeds multi-hop neighbourhood information into the Transformer.
    """

    def __init__(self, node_embedding, num_users, dimension, max_len,
                 dropout=0.2, num_layers=2, graph_enriched=False):
        super().__init__()
        self.node_embedding = node_embedding
        self.num_users = num_users
        self.max_len = max_len
        self.graph_enriched = graph_enriched
        self.position = nn.Embedding(max_len, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=2,
            dim_feedforward=4 * dimension,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dimension)

    def encode(self, sequence, length, propagated=None):
        # sequence uses 0 for padding and item_id + 1 otherwise.
        item_nodes = self.num_users + (sequence - 1).clamp_min(0)
        if self.graph_enriched and propagated is not None:
            hidden = propagated[item_nodes]
        else:
            hidden = self.node_embedding(item_nodes)
        hidden = hidden.masked_fill(sequence.eq(0).unsqueeze(-1), 0.0)
        positions = torch.arange(self.max_len, device=sequence.device)[None, :]
        hidden = hidden + self.position(positions)
        causal = torch.triu(
            torch.ones(self.max_len, self.max_len, device=sequence.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal,
            src_key_padding_mask=sequence.eq(0),
        )
        indices = (length - 1).clamp_min(0)
        output = hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        return self.norm(output)

    def initialize_from(self, sasrec):
        self.position.load_state_dict(sasrec.position.state_dict())
        if len(list(self.encoder.parameters())) == len(list(sasrec.encoder.parameters())):
            self.encoder.load_state_dict(sasrec.encoder.state_dict())
        else:
            # Layer count differs (e.g. pretrained 2-layer → 1-layer encoder).
            # Copy only the first layer's weights.
            src = dict(sasrec.encoder.named_parameters())
            for name, param in self.encoder.named_parameters():
                if name in src and param.shape == src[name].shape:
                    param.data.copy_(src[name].data)
        self.norm.load_state_dict(sasrec.norm.state_dict())


class JointHybrid(nn.Module):
    """Shared-embedding SASRec + graph encoder with trainable fusion.

    ``graph_kind='lightgcn', fusion_kind='gate'`` is AdaFuse. The
    ``gcn/concat`` configuration is the controlled SASGCN-style comparator.
    """

    def __init__(
        self,
        num_users,
        num_items,
        dimension,
        max_len,
        adjacency,
        graph_kind,
        fusion_kind,
        graph_layers=3,
        seq_layers=2,
        graph_enriched=False,
        length_aware=False,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.adjacency = adjacency
        self.graph_kind = graph_kind
        self.fusion_kind = fusion_kind
        self.graph_layers = graph_layers
        self.graph_enriched = graph_enriched
        self.length_aware = length_aware
        self.node_embedding = nn.Embedding(num_users + num_items, dimension)
        nn.init.xavier_normal_(self.node_embedding.weight)
        self.sequence_encoder = SharedSASRecEncoder(
            self.node_embedding, num_users, dimension, max_len,
            num_layers=seq_layers, graph_enriched=graph_enriched,
        )
        if graph_kind == "gcn":
            self.gcn_linears = nn.ModuleList(
                nn.Linear(dimension, dimension) for _ in range(graph_layers)
            )
            self.gcn_norms = nn.ModuleList(
                nn.LayerNorm(dimension) for _ in range(graph_layers)
            )
        elif graph_kind != "lightgcn":
            raise ValueError(graph_kind)

        if fusion_kind == "gate":
            # Exact plan equation: sigmoid(w^T [seq; col; seq*col] + b).
            self.fusion = nn.Sequential(nn.Linear(3 * dimension, 1), nn.Sigmoid())
            if length_aware:
                self.length_scale = nn.Parameter(torch.tensor(1.0))
                self.length_bias = nn.Parameter(torch.tensor(0.0))
        elif fusion_kind == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(2 * dimension, dimension),
                nn.ReLU(),
                nn.Linear(dimension, dimension),
            )
        else:
            raise ValueError(fusion_kind)
        self.propagated = None

    def initialize_from_baselines(self, lightgcn=None, sasrec=None):
        if lightgcn is not None:
            self.node_embedding.weight.data.copy_(lightgcn.embedding.weight.data)
        if sasrec is not None:
            self.sequence_encoder.initialize_from(sasrec)

    def propagate(self):
        current = self.node_embedding.weight
        if self.graph_kind == "lightgcn":
            layers = [current]
            for _ in range(self.graph_layers):
                current = torch.sparse.mm(self.adjacency, current)
                layers.append(current)
            return torch.stack(layers).mean(0)
        for linear, norm in zip(self.gcn_linears, self.gcn_norms):
            current = torch.sparse.mm(self.adjacency, current)
            current = norm(F.relu(linear(current)))
        return current

    def cache_graph(self):
        with torch.no_grad():
            self.propagated = self.propagate().detach()

    def clear_graph(self):
        self.propagated = None

    def fuse(self, sequence_user, graph_user, length=None):
        if self.fusion_kind == "concat":
            return self.fusion(torch.cat((sequence_user, graph_user), dim=-1)), None
        alpha = self.fusion(
            torch.cat((sequence_user, graph_user, sequence_user * graph_user), dim=-1)
        )
        if self.length_aware and length is not None:
            confidence = torch.sigmoid(
                self.length_scale * torch.log1p(length.float()).unsqueeze(-1)
                + self.length_bias
            )
            alpha = alpha * confidence
        # Residual: graph is the base, sequence corrects it. alpha=0 → pure graph.
        return graph_user + alpha * (sequence_user - graph_user), alpha

    def encode_user(self, users, sequence, length, propagated=None):
        graph = self.propagated if propagated is None else propagated
        if graph is None:
            graph = self.propagate()
        sequence_user = self.sequence_encoder.encode(
            sequence, length, propagated=graph if self.graph_enriched else None,
        )
        return self.fuse(sequence_user, graph[users], length)

    def triplet_with_graph(self, graph, users, sequence, length, positives, negatives):
        user, alpha = self.encode_user(users, sequence, length, graph)
        graph_user = graph[users]
        positive = graph[self.num_users + positives]
        negative = graph[self.num_users + negatives]
        return user, positive, negative, graph_user, alpha

    def score(self, user, items, sequence, length):
        encoded, _ = self.encode_user(user, sequence, length)
        return (encoded * self.propagated[self.num_users + items]).sum(-1)

    def alpha(self, users, sequence, length):
        _, alpha = self.encode_user(users, sequence, length)
        return alpha


class FusionModel(nn.Module):
    def __init__(self, lightgcn, sasrec, dimension, kind):
        super().__init__()
        self.lightgcn = lightgcn
        self.sasrec = sasrec
        self.kind = kind
        for parameter in self.lightgcn.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.sasrec.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if kind == "concat":
            self.head = nn.Sequential(
                nn.Linear(2 * dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension)
            )
        elif kind == "adafuse":
            self.head = nn.Sequential(
                nn.Linear(3 * dimension, dimension),
                nn.ReLU(),
                nn.Linear(dimension, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(kind)
        self.propagated = None

    def cache_graph(self):
        self.lightgcn.eval()
        self.sasrec.eval()
        with torch.no_grad():
            self.propagated = self.lightgcn.propagate().detach()

    def clear_graph(self):
        self.propagated = None

    def encode_user(self, users, sequence, length):
        if self.propagated is None:
            self.cache_graph()
        # Keep the frozen encoder deterministic even when the fusion head is
        # in training mode (SASRec contains dropout).
        self.lightgcn.eval()
        self.sasrec.eval()
        with torch.no_grad():
            graph_user = self.propagated[users]
            sequence_user = self.sasrec.encode(sequence, length)
        if self.kind == "concat":
            return self.head(torch.cat((sequence_user, graph_user), dim=-1))
        alpha = self.head(
            torch.cat((sequence_user, graph_user, sequence_user * graph_user), dim=-1)
        )
        return graph_user + alpha * (sequence_user - graph_user)

    def triplet(self, users, sequence, length, positives, negatives):
        encoded = self.encode_user(users, sequence, length)
        positive = self.propagated[self.lightgcn.num_users + positives]
        negative = self.propagated[self.lightgcn.num_users + negatives]
        return encoded, positive, negative

    def score(self, user, items, sequence, length):
        encoded = self.encode_user(user, sequence, length)
        item_embeddings = self.propagated[self.lightgcn.num_users + items]
        return (encoded * item_embeddings).sum(-1)


def bpr_loss(user, positive, negative, weight_decay):
    ranking = -F.logsigmoid((user * positive).sum(-1) - (user * negative).sum(-1)).mean()
    regularization = weight_decay * (
        user.square().sum() + positive.square().sum() + negative.square().sum()
    ) / user.size(0)
    return ranking + regularization


def known_items(data: dict) -> dict[int, set[int]]:
    return {
        user: set(data["train"][user] + data["validation"][user] + data["test"][user])
        for user in range(data["num_users"])
    }


def evaluation_users(data, maximum, seed):
    users = list(range(data["num_users"]))
    if maximum is not None and len(users) > maximum:
        users = random.Random(seed).sample(users, maximum)
    return sorted(users)


def sampled_candidates(user, positive, excluded, num_items, count, seed):
    rng = random.Random(seed * 1_000_003 + user)
    available = num_items - len(excluded)
    if count > available:
        raise ValueError("requested more unique negatives than are available")
    negatives: set[int] = set()
    while len(negatives) < count:
        item = rng.randrange(num_items)
        if item not in excluded:
            negatives.add(item)
    return [positive, *sorted(negatives)]


def evaluate(model, data, split, args, candidate_seed):
    model.eval()
    users = evaluation_users(data, args.max_eval_users, candidate_seed)
    all_known = known_items(data)
    per_user = {}
    propagated = None
    if isinstance(model, LightGCN):
        with torch.no_grad():
            propagated = model.propagate()
    if isinstance(model, (FusionModel, JointHybrid)):
        model.cache_graph()

    with torch.no_grad():
        for user in users:
            positive = data[split][user][0]
            if args.ranking == "full":
                candidates = [positive] + [
                    item for item in range(data["num_items"]) if item not in all_known[user]
                ]
            else:
                candidates = sampled_candidates(
                    user,
                    positive,
                    all_known[user],
                    data["num_items"],
                    args.num_negatives,
                    candidate_seed,
                )
            if split == "validation":
                history = data["train"][user]
            else:
                history = data["train"][user] + data["validation"][user]
            padded, length = pad_sequence(history, args.max_seq_len)
            user_tensor = torch.tensor([user], device=DEVICE)
            item_tensor = torch.tensor(candidates, device=DEVICE)
            sequence = torch.tensor([padded], device=DEVICE)
            length_tensor = torch.tensor([length], device=DEVICE)

            if isinstance(model, LightGCN):
                scores = model.score(user_tensor, item_tensor, propagated=propagated)
            else:
                scores = model.score(user_tensor, item_tensor, sequence, length_tensor)
            order = torch.argsort(scores, descending=True).tolist()
            rank = order.index(0) + 1
            row = {}
            for k in (10, 20):
                row[f"Recall@{k}"] = float(rank <= k)
                row[f"NDCG@{k}"] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
                row[f"HR@{k}"] = float(rank <= k)
            per_user[user] = row

    if isinstance(model, (FusionModel, JointHybrid)):
        model.clear_graph()
    aggregate = {
        metric: float(np.mean([row[metric] for row in per_user.values()]))
        for metric in (
            "Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "HR@10", "HR@20"
        )
    }
    return aggregate, per_user


def extract_alphas(model, data, args):
    """Return interpretable AdaFuse weights and activity counts per user."""
    model.eval()
    model.cache_graph()
    values = {}
    users = evaluation_users(data, args.max_eval_users, args.seed + 30_000)
    with torch.no_grad():
        for user in users:
            history = data["train"][user] + data["validation"][user]
            padded, length = pad_sequence(history, args.max_seq_len)
            alpha = model.alpha(
                torch.tensor([user], device=DEVICE),
                torch.tensor([padded], device=DEVICE),
                torch.tensor([length], device=DEVICE),
            )
            values[user] = {
                "alpha": float(alpha.item()),
                "train_interactions": len(data["train"][user]),
            }
    model.clear_graph()
    return values


def train_model(model, loader, data, args, name, checkpoint, sequence_model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    best_score = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(model, FusionModel):
            model.cache_graph()
        total = 0.0
        batches = 0
        for batch in loader:
            batch = [tensor.to(DEVICE) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            if sequence_model:
                users, sequences, lengths, positives, negatives = batch
                if isinstance(model, FusionModel):
                    embeddings = model.triplet(users, sequences, lengths, positives, negatives)
                else:
                    embeddings = model.triplet(sequences, lengths, positives, negatives)
            else:
                users, positives, negatives = batch
                embeddings = model.triplet(users, positives, negatives)
            loss = bpr_loss(*embeddings, args.weight_decay)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        if isinstance(model, FusionModel):
            model.clear_graph()

        if epoch % args.eval_every != 0:
            continue
        metrics, _ = evaluate(model, data, "validation", args, args.seed + 10_000)
        score = metrics["NDCG@10"]
        log(f"{name} epoch {epoch:3d}: loss={total/max(batches,1):.5f}, val={metrics}")
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not isinstance(model, FusionModel) or key.startswith("head.")
            }
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += args.eval_every
            if stale_epochs >= args.patience:
                log(f"{name} early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced; lower --eval-every")
    if isinstance(model, FusionModel):
        model.head.load_state_dict(
            {key.removeprefix("head."): value.to(DEVICE) for key, value in best_state.items()}
        )
    else:
        model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})
    return model


def train_joint_model(model, loader, data, args, name, checkpoint):
    """End-to-end hybrid optimization with one GNN propagation per epoch.

    Mini-batch losses accumulate gradients on a detached leaf representing the
    propagated graph. The aggregate leaf gradient is then sent through the GNN
    once. This is the exact chain-rule gradient for the epoch objective while
    avoiding one full sparse propagation per sequence mini-batch.

    When args.pretrain_epochs > 0, Phase 1 trains only the fusion gate with
    encoders frozen. Phase 2 unfreezes everything for joint fine-tuning.
    This prevents joint-training collapse on dense datasets.
    """
    pretrain_epochs = getattr(args, "pretrain_epochs", 0)
    alpha_reg = getattr(args, "alpha_reg", 0.0)
    graph_loss_weight = getattr(args, "graph_loss_weight", 0.0)
    example_count = len(loader.dataset)

    # --- Phase 1: gate-only warm-up (encoders frozen) ---
    if pretrain_epochs > 0:
        log(f"{name} phase-1: training fusion gate only for {pretrain_epochs} epochs")
        # Freeze all encoder params; only fusion gate trains
        for param_name, param in model.named_parameters():
            param.requires_grad_(param_name.startswith("fusion"))
        gate_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
        )
        model.cache_graph()
        for epoch in range(1, pretrain_epochs + 1):
            model.train()
            gate_optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for batch in loader:
                users, sequences, lengths, positives, negatives = [
                    t.to(DEVICE) for t in batch
                ]
                fused_user, alpha = model.encode_user(users, sequences, lengths)
                positive = model.propagated[model.num_users + positives]
                negative = model.propagated[model.num_users + negatives]
                loss = bpr_loss(fused_user, positive, negative, args.weight_decay)
                if alpha_reg > 0 and alpha is not None:
                    loss = loss + alpha_reg * alpha.mean()
                weight = users.size(0) / example_count
                (loss * weight).backward()
                total += loss.item() * users.size(0)
            gate_optimizer.step()
            if epoch % args.eval_every == 0:
                metrics, _ = evaluate(model, data, "validation", args, args.seed + 10_000)
                log(f"{name} pretrain epoch {epoch:3d}: loss={total/example_count:.5f}, val={metrics}")
        model.clear_graph()
        # Unfreeze everything for Phase 2
        for param in model.parameters():
            param.requires_grad_(True)
        log(f"{name} phase-2: joint end-to-end fine-tuning")

    # --- Phase 2: full joint training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_score = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        propagated = model.propagate()
        graph_leaf = propagated.detach().requires_grad_(True)
        total = 0.0

        for batch in loader:
            users, sequences, lengths, positives, negatives = [
                tensor.to(DEVICE) for tensor in batch
            ]
            fused_user, positive, negative, graph_user, alpha = (
                model.triplet_with_graph(
                    graph_leaf, users, sequences, lengths, positives, negatives
                )
            )
            loss = bpr_loss(fused_user, positive, negative, args.weight_decay)
            if graph_loss_weight > 0:
                loss = loss + graph_loss_weight * bpr_loss(
                    graph_user, positive, negative, args.weight_decay
                )
            if alpha_reg > 0 and alpha is not None:
                loss = loss + alpha_reg * alpha.mean()
            weight = users.size(0) / example_count
            (loss * weight).backward()
            total += loss.item() * users.size(0)

        if graph_leaf.grad is None:
            raise RuntimeError("hybrid graph branch received no ranking gradient")
        propagated.backward(graph_leaf.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue
        metrics, _ = evaluate(model, data, "validation", args, args.seed + 10_000)
        score = metrics["NDCG@10"]
        log(f"{name} epoch {epoch:3d}: loss={total/example_count:.5f}, val={metrics}")
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            # Runtime graph caches are plain attributes and are not serialized.
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += args.eval_every
            if stale_epochs >= args.patience:
                log(f"{name} early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced; lower --eval-every")
    model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})
    return model


def train_lightgcn(model, loader, data, args, name, checkpoint):
    """Train LightGCN with one propagation and bounded BPR batches per epoch.

    Keeping the propagated table as a temporary leaf preserves the exact
    gradient while avoiding both repeated sparse propagation and a single
    multi-million-example GPU batch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_score = -1.0
    best_state = None
    stale_epochs = 0
    example_count = len(loader.dataset)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        propagated = model.propagate()
        graph_leaf = propagated.detach().requires_grad_(True)
        total = 0.0

        for users, positives, negatives in loader:
            users = users.to(DEVICE)
            positives = positives.to(DEVICE)
            negatives = negatives.to(DEVICE)
            embeddings = (
                graph_leaf[users],
                graph_leaf[model.num_users + positives],
                graph_leaf[model.num_users + negatives],
            )
            loss = bpr_loss(*embeddings, args.weight_decay)
            weight = users.size(0) / example_count
            (loss * weight).backward()
            total += loss.item() * users.size(0)

        if graph_leaf.grad is None:
            raise RuntimeError("LightGCN propagated embeddings received no gradient")
        propagated.backward(graph_leaf.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % args.eval_every != 0:
            continue
        metrics, _ = evaluate(model, data, "validation", args, args.seed + 10_000)
        score = metrics["NDCG@10"]
        log(f"{name} epoch {epoch:3d}: loss={total/example_count:.5f}, val={metrics}")
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += args.eval_every
            if stale_epochs >= args.patience:
                log(f"{name} early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced; lower --eval-every")
    model.load_state_dict({key: value.to(DEVICE) for key, value in best_state.items()})
    return model


def load_checkpoint(model, path, fusion=False):
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if fusion:
        model.head.load_state_dict({key.removeprefix("head."): value for key, value in state.items()})
    else:
        model.load_state_dict(state)
    return model


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / f"seed_{args.seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(run_dir / "training.log")
    set_seed(args.seed)
    log(f"device={DEVICE}; arguments={vars(args)}")

    data = load_data(args.data)
    log(
        f"dataset users={data['num_users']:,}, items={data['num_items']:,}, "
        f"train={sum(map(len, data['train'].values())):,}"
    )
    adjacency = build_graph(data["train"], data["num_users"], data["num_items"])
    all_known = known_items(data)
    graph_pairs = [
        (user, item) for user, items in data["train"].items() for item in items
    ]
    graph_data = GraphDataset(graph_pairs, all_known, data["num_items"], args.seed)
    sequence_data = SequenceDataset(
        data["train"], all_known, data["num_items"], args.max_seq_len, args.seed
    )
    graph_data = subset_dataset(graph_data, args.max_train_samples, args.seed)
    sequence_data = subset_dataset(sequence_data, args.max_train_samples, args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    graph_loader = DataLoader(graph_data, **loader_options)
    sequence_loader = DataLoader(sequence_data, **loader_options)

    results = {}
    per_user_scores = {}

    def fit_or_load(model, key, loader, is_sequence):
        checkpoint = checkpoint_dir / f"{key}.pt"
        if args.resume and checkpoint.is_file():
            log(f"loading {key} from {checkpoint}")
            return load_checkpoint(model, checkpoint, isinstance(model, FusionModel))
        log(f"training {key}")
        if isinstance(model, LightGCN):
            return train_lightgcn(model, loader, data, args, key, checkpoint)
        if isinstance(model, JointHybrid):
            return train_joint_model(model, loader, data, args, key, checkpoint)
        return train_model(model, loader, data, args, key, checkpoint, is_sequence)

    if "popularity" in args.models:
        popularity = Popularity(data["train"], data["num_items"]).to(DEVICE)
        results["popularity"], per_user_scores["popularity"] = evaluate(
            popularity, data, "test", args, args.seed + 20_000
        )
        log(f"popularity test={results['popularity']}")
        del popularity

    if "mf" in args.models:
        mf = MatrixFactorization(data["num_users"], data["num_items"], args.embedding_dim).to(DEVICE)
        mf = fit_or_load(mf, "mf", graph_loader, False)
        results["mf"], per_user_scores["mf"] = evaluate(
            mf, data, "test", args, args.seed + 20_000
        )
        log(f"mf test={results['mf']}")
        del mf

    needs_graph = any(
        name in args.models for name in ("lightgcn", "sasgcn", "concat", "adafuse")
    )
    needs_sequence = any(
        name in args.models for name in ("sasrec", "sasgcn", "concat", "adafuse")
    )
    lightgcn = None
    sasrec = None
    if needs_graph:
        lightgcn = LightGCN(
            data["num_users"], data["num_items"], args.embedding_dim, adjacency
        ).to(DEVICE)
        lightgcn = fit_or_load(lightgcn, "lightgcn", graph_loader, False)
        if "lightgcn" in args.models:
            results["lightgcn"], per_user_scores["lightgcn"] = evaluate(
                lightgcn, data, "test", args, args.seed + 20_000
            )
            log(f"lightgcn test={results['lightgcn']}")

    if needs_sequence:
        sasrec = SASRec(
            data["num_items"], args.embedding_dim, args.max_seq_len
        ).to(DEVICE)
        sasrec = fit_or_load(sasrec, "sasrec", sequence_loader, True)
        if "sasrec" in args.models:
            results["sasrec"], per_user_scores["sasrec"] = evaluate(
                sasrec, data, "test", args, args.seed + 20_000
            )
            log(f"sasrec test={results['sasrec']}")

    # Frozen LightGCN+SASRec concatenation is retained as an explicit ablation.
    if "concat" in args.models:
        kind = "concat"
        fusion = FusionModel(lightgcn, sasrec, args.embedding_dim, kind).to(DEVICE)
        fusion = fit_or_load(fusion, kind, sequence_loader, True)
        results[kind], per_user_scores[kind] = evaluate(
            fusion, data, "test", args, args.seed + 20_000
        )
        log(f"{kind} test={results[kind]}")
        del fusion

    if "sasgcn" in args.models:
        sasgcn = JointHybrid(
            data["num_users"],
            data["num_items"],
            args.embedding_dim,
            args.max_seq_len,
            adjacency,
            graph_kind="gcn",
            fusion_kind="concat",
            graph_layers=2,
            seq_layers=args.seq_layers,
        ).to(DEVICE)
        sasgcn.initialize_from_baselines(sasrec=sasrec)
        sasgcn = fit_or_load(sasgcn, "sasgcn", sequence_loader, True)
        results["sasgcn"], per_user_scores["sasgcn"] = evaluate(
            sasgcn, data, "test", args, args.seed + 20_000
        )
        log(f"sasgcn test={results['sasgcn']}")
        del sasgcn

    alpha_values = {}
    if "adafuse" in args.models:
        adafuse = JointHybrid(
            data["num_users"],
            data["num_items"],
            args.embedding_dim,
            args.max_seq_len,
            adjacency,
            graph_kind="lightgcn",
            fusion_kind="gate",
            graph_layers=3,
            seq_layers=args.seq_layers,
            graph_enriched=args.graph_enriched,
            length_aware=getattr(args, "length_aware", False),
        ).to(DEVICE)
        adafuse.initialize_from_baselines(lightgcn=lightgcn, sasrec=sasrec)
        adafuse = fit_or_load(adafuse, "adafuse", sequence_loader, True)
        results["adafuse"], per_user_scores["adafuse"] = evaluate(
            adafuse, data, "test", args, args.seed + 20_000
        )
        log(f"adafuse test={results['adafuse']}")
        alpha_values = extract_alphas(adafuse, data, args)
        del adafuse

    payload = {
        "protocol": {
            "dataset": data.get("dataset"),
            "split": data.get("protocol"),
            "ranking": (
                "full eligible-item ranking with interacted-item masking"
                if args.ranking == "full"
                else f"1 positive + {args.num_negatives} fixed sampled negatives"
            ),
            "selection": "validation NDCG@10",
            "seed": args.seed,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        "aggregate": results,
        "per_user": per_user_scores,
        "alphas": alpha_values,
    }
    with (run_dir / "results.pkl").open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol": payload["protocol"], "aggregate": results}, handle, indent=2)
    log(f"saved results to {run_dir}")


if __name__ == "__main__":
    main()
