# Repository Guide

## Current State

- This is currently a roadmap/data repository, not a runnable Python project. The human entrypoint is `roadmap/README.md`; the implementation plan starts at `roadmap/04-btp-implementation/00-btp-master-plan.md`.
- There is no project source tree, dependency manifest, lockfile, test suite, lint/typecheck configuration, or CI. Python in the Markdown files is illustrative; do not report it as implemented or invent verification commands.
- Do not use or modify `roadmap/.venv`: its `pyvenv.cfg` records an external Linux path, so it is not a portable environment for this Windows workspace. The PyG install snippets in `roadmap/01-gnn-study/07-pytorch-geometric.md` are not a complete environment specification.
- The directory trees in the implementation documents are proposals and conflict (`experiments/` vs `results/`, `report/` vs `thesis/`, YAML vs JSON). Ask which direction to implement rather than treating one as established architecture.
- The proposed research direction also differs by document: the master plan recommends DualCL, while `07-final-model.md` presents metadata-augmented LightGCN as an example contribution. Neither exists in code.

## Data

- Actual datasets live under `roadmap/data/`; examples using `data/...` therefore assume `roadmap/` as the working directory.
- `roadmap/data/amazon-book/{train,test}.txt` are already split LightGCN-format files: each line is `user_id item_id...`. They contain no timestamps and there is no validation file, so the roadmap's temporal leave-one-out procedure cannot be reconstructed from these files alone.
- Amazon-Beauty reviews are JSONL at `roadmap/data/amazon-beauty/raw/review_categories/All_Beauty.jsonl`; relevant fields are `user_id`, `parent_asin`, `rating`, and `timestamp`. Metadata is a separate Parquet file under `raw_meta_All_Beauty/`.
- `data/processed/amazon_book_processed.pkl` and all documented configs, logs, results, figures, and thesis outputs are planned artifacts, not checked-in outputs.

## Experiment Invariants

- Build recommendation graphs from training interactions only. Offset item node IDs by `num_users`, add both edge directions, and expect `edge_index.shape == [2, 2 * num_train_interactions]`.
- During ranking evaluation, mask each user's training items, compute metrics per user, then average across users. Full-item ranking is preferred; if using 100 sampled negatives, record that protocol because it changes comparability.
- Report Recall and NDCG at K=10 and 20. Select hyperparameters and early stopping on validation NDCG@10, then evaluate the test set once.
- Comparable experiments use embedding dimension 64 and seeds `[42, 123, 456, 789, 1024]`, reporting mean and standard deviation. Preserve per-user scores when performing paired significance tests.
