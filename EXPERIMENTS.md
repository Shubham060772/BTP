# Reproducible experiment workflow

The earlier `src/run_all.py` experiment is invalidated. It treated the
untimestamped Amazon-Book file order as a behavioral sequence, leaked targets
into SASRec inputs, and detached the hybrid graph branch during training. Its
printed metrics must not be used in the thesis or compared with paper tables.

## 1. Prepare Amazon-Beauty

From `B:\projects\BTP` with the existing local environment activated:

```powershell
.\venv\Scripts\python.exe src\prepare_amazon_beauty.py 2>&1 |
  Tee-Object -FilePath results\logs\prepare_amazon_beauty.log
```

This streams `All_Beauty.jsonl`, keeps the latest event for each user-product
pair, performs iterative 5-core filtering, and uses temporal leave-two-out:
the penultimate interaction is validation and the last interaction is test.

The checked-in Amazon Reviews 2023 All Beauty data collapses to zero users
under iterative 10-core filtering. The 5-core choice is therefore an empirical
dataset constraint and must be reported; the approximate 10-core statistics in
the revised plan describe a different Amazon-Beauty version/protocol.

## 2. Run a CPU pilot

Start with one seed and a bounded workload:

```powershell
.\venv\Scripts\python.exe src\run_beauty.py `
  --seed 42 `
  --max-train-samples 100000 `
  --max-eval-users 3000 `
  --epochs 60 `
  --patience 15
```

The command writes its live log, checkpoints, aggregate metrics, and per-user
metrics under `results/beauty/seed_42/`. If it stops, rerun the identical
command with `--resume`.

The default protocol now uses full eligible-item ranking with interacted-item
masking and records that protocol in `summary.json`. Use `--ranking sampled`
only for larger debugging runs and report it separately.

## 3. Final comparable runs on GPU

After the pilot passes, remove both workload caps and run the required seeds:

```powershell
$seeds = 42, 123, 456, 789, 1024
foreach ($seed in $seeds) {
  python src\run_beauty.py --seed $seed --epochs 100 --patience 20
}
```

Run each seed in a fresh process, then aggregate the saved per-user scores:

```powershell
python src\aggregate_beauty_results.py
```

This reports mean and sample standard deviation across five seeds. Paired tests
first average each user over seeds and then test across users, avoiding treating
repeated measurements of one user as independent observations.

## Model interpretation

- `mf`, `lightgcn`, and `sasrec` are local baseline implementations under one
  controlled protocol; they are not claimed as exact reproductions of every
  original-paper preprocessing choice.
- `concat` is a controlled frozen-encoder concatenation ablation.
- `sasgcn` is the trainable vanilla-GCN + SASRec static-fusion comparator; it is
  SASGCN-style, not a claim of exact reproduction of the authors' code.
- `adafuse` uses shared item embeddings, end-to-end graph/sequence learning,
  and the scalar gate specified in the revised research plan.
