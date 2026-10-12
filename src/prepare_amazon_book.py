"""Prepare Amazon-Book (LightGCN format) for the run_beauty.py pipeline.

The pre-split train.txt / test.txt files have no timestamps. Item list order
is used as a synthetic sequence, which is standard practice when timestamps
are unavailable (LightGCN, SASGCN papers do this on Amazon-Book).

Split strategy:
  - train: all items from train.txt except the last per user
  - validation: last item from train.txt per user
  - test: first item from test.txt per user
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "roadmap" / "data" / "amazon-book"
DEFAULT_OUTPUT = DEFAULT_DIR / "amazon_book.pkl"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DIR,
        help="Directory containing train.txt and test.txt",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_interactions(path: Path) -> dict[int, list[int]]:
    data: dict[int, list[int]] = {}
    with path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            items = [int(p) for p in parts[1:]]
            data[user] = items
    return data


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.txt"
    test_path = args.data_dir / "test.txt"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if not test_path.is_file():
        raise FileNotFoundError(test_path)

    log(f"reading {train_path}")
    raw_train = read_interactions(train_path)
    log(f"reading {test_path}")
    raw_test = read_interactions(test_path)

    users = sorted(raw_train.keys())
    assert users == list(range(len(users))), "user IDs must be 0-indexed and contiguous"
    missing_test = set(users) - set(raw_test.keys())
    if missing_test:
        log(f"dropping {len(missing_test)} users with no test items")

    all_items: set[int] = set()
    for items in raw_train.values():
        all_items.update(items)
    for items in raw_test.values():
        all_items.update(items)
    num_items = max(all_items) + 1
    assert all_items == set(range(num_items)), "item IDs must be 0-indexed and contiguous"

    num_users = len(users)
    train: dict[int, list[int]] = {}
    validation: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    timestamps: dict[int, list[int]] = {}
    skipped = 0

    for user in users:
        train_items = raw_train[user]
        test_items = raw_test.get(user, [])
        if len(train_items) < 2 or len(test_items) < 1:
            skipped += 1
            continue
        train[user] = train_items[:-1]
        validation[user] = [train_items[-1]]
        test[user] = [test_items[0]]
        timestamps[user] = list(range(len(train_items) - 1))

    if skipped:
        log(f"skipped {skipped} users with < 2 training items; remapping")
        old_to_new = {}
        for new_id, old_id in enumerate(sorted(train.keys())):
            old_to_new[old_id] = new_id
        train = {old_to_new[u]: v for u, v in train.items()}
        validation = {old_to_new[u]: v for u, v in validation.items()}
        test = {old_to_new[u]: v for u, v in test.items()}
        timestamps = {old_to_new[u]: v for u, v in timestamps.items()}
        num_users = len(train)

    train_count = sum(len(v) for v in train.values())
    log(f"users={num_users:,}, items={num_items:,}, train={train_count:,}")
    log(f"avg train items/user={train_count/num_users:.1f}")

    data = {
        "format_version": 1,
        "dataset": "Amazon-Book (LightGCN format, synthetic sequence order)",
        "protocol": "pre-split train/test, last-train-item validation, position-order sequence",
        "min_interactions": 1,
        "num_users": num_users,
        "num_items": num_items,
        "train": train,
        "validation": validation,
        "test": test,
        "timestamps": timestamps,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"saved to {args.output}")


if __name__ == "__main__":
    main()
