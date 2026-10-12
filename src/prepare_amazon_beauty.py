"""Prepare timestamp-ordered Amazon All Beauty data for recommendation.

The raw file is streamed once. Repeated user/product reviews are collapsed to
the latest timestamp, an iterative k-core filter is applied, and each user's
last two interactions become validation and test respectively.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "roadmap"
    / "data"
    / "amazon-beauty"
    / "raw"
    / "review_categories"
    / "All_Beauty.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "roadmap"
    / "data"
    / "amazon-beauty"
    / "processed"
    / "beauty_5core.pkl"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Debug only: stop after this many JSONL rows.",
    )
    return parser.parse_args()


def load_and_deduplicate(path: Path, max_lines: int | None = None):
    """Return integer-coded (user, item, timestamp) interactions.

    The original IDs are mapped while streaming to avoid retaining JSON rows.
    For duplicate user-item pairs, the latest timestamp is kept.
    """
    user_to_raw: list[str] = []
    item_to_raw: list[str] = []
    raw_to_user: dict[str, int] = {}
    raw_to_item: dict[str, int] = {}
    latest: dict[tuple[int, int], int] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_lines is not None and line_number > max_lines:
                break
            row = json.loads(line)
            raw_user = row.get("user_id")
            raw_item = row.get("parent_asin")
            timestamp = row.get("timestamp")
            if raw_user is None or raw_item is None or timestamp is None:
                continue

            user = raw_to_user.get(raw_user)
            if user is None:
                user = len(user_to_raw)
                raw_to_user[raw_user] = user
                user_to_raw.append(raw_user)

            item = raw_to_item.get(raw_item)
            if item is None:
                item = len(item_to_raw)
                raw_to_item[raw_item] = item
                item_to_raw.append(raw_item)

            key = (user, item)
            timestamp = int(timestamp)
            if timestamp > latest.get(key, -1):
                latest[key] = timestamp

            if line_number % 500_000 == 0:
                log(
                    f"read {line_number:,} rows; "
                    f"retained {len(latest):,} unique user-item pairs"
                )

    interactions = [(u, i, ts) for (u, i), ts in latest.items()]
    return interactions, user_to_raw, item_to_raw


def iterative_k_core(
    interactions: list[tuple[int, int, int]], min_interactions: int
) -> list[tuple[int, int, int]]:
    """Repeatedly remove users and items below the requested degree."""
    iteration = 0
    while True:
        iteration += 1
        user_counts = Counter(user for user, _, _ in interactions)
        item_counts = Counter(item for _, item, _ in interactions)
        filtered = [
            row
            for row in interactions
            if user_counts[row[0]] >= min_interactions
            and item_counts[row[1]] >= min_interactions
        ]
        log(
            f"k-core iteration {iteration}: {len(interactions):,} -> "
            f"{len(filtered):,} interactions"
        )
        if len(filtered) == len(interactions):
            return filtered
        interactions = filtered


def remap_and_split(interactions, original_users, original_items, min_interactions):
    surviving_users = sorted({user for user, _, _ in interactions})
    surviving_items = sorted({item for _, item, _ in interactions})
    user_remap = {old: new for new, old in enumerate(surviving_users)}
    item_remap = {old: new for new, old in enumerate(surviving_items)}

    histories: list[list[tuple[int, int]]] = [[] for _ in surviving_users]
    for old_user, old_item, timestamp in interactions:
        histories[user_remap[old_user]].append((timestamp, item_remap[old_item]))

    train: dict[int, list[int]] = {}
    validation: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    timestamps: dict[int, list[int]] = {}
    for user, history in enumerate(histories):
        # Dense item ID is a deterministic tie-break for equal timestamps.
        history.sort(key=lambda pair: (pair[0], pair[1]))
        if len(history) < min_interactions:
            raise RuntimeError("k-core invariant violated after remapping")
        ordered_items = [item for _, item in history]
        timestamps[user] = [timestamp for timestamp, _ in history]
        train[user] = ordered_items[:-2]
        validation[user] = [ordered_items[-2]]
        test[user] = [ordered_items[-1]]

    return {
        "format_version": 1,
        "dataset": "Amazon Reviews 2023 - All Beauty",
        "protocol": "iterative-k-core, temporal leave-two-out",
        "min_interactions": min_interactions,
        "num_users": len(surviving_users),
        "num_items": len(surviving_items),
        "train": train,
        "validation": validation,
        "test": test,
        "timestamps": timestamps,
        "raw_user_ids": [original_users[index] for index in surviving_users],
        "raw_item_ids": [original_items[index] for index in surviving_items],
    }


def validate(data: dict) -> None:
    num_users = data["num_users"]
    num_items = data["num_items"]
    assert set(data["train"]) == set(range(num_users))
    assert set(data["validation"]) == set(range(num_users))
    assert set(data["test"]) == set(range(num_users))
    for user in range(num_users):
        train = data["train"][user]
        validation = data["validation"][user]
        test = data["test"][user]
        assert len(train) >= data["min_interactions"] - 2
        assert len(validation) == len(test) == 1
        assert not (set(train) & set(validation))
        assert not (set(train) & set(test))
        assert 0 <= min(train + validation + test)
        assert max(train + validation + test) < num_items


def main() -> None:
    args = parse_args()
    if args.min_interactions < 3:
        raise ValueError("--min-interactions must be at least 3 for leave-two-out")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    log(f"reading {args.input}")
    interactions, users, items = load_and_deduplicate(args.input, args.max_lines)
    log(f"loaded {len(interactions):,} unique interactions")
    interactions = iterative_k_core(interactions, args.min_interactions)
    data = remap_and_split(interactions, users, items, args.min_interactions)
    validate(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_count = sum(map(len, data["train"].values()))
    log(
        f"saved {data['num_users']:,} users, {data['num_items']:,} items, "
        f"and {train_count:,} training interactions to {args.output}"
    )


if __name__ == "__main__":
    main()
