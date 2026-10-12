# AdaFuse recommendation experiments

Reproducible implementations of MF-BPR, LightGCN, SASRec, a SASGCN-style
hybrid, static concatenation, and AdaFuse under one ranking protocol.

## Run directly in Kaggle

1. Enable a GPU accelerator.
2. Add either Amazon Reviews 2023 `All_Beauty.jsonl`, or an Amazon-Book
   dataset containing `train.txt` and `test.txt`, as notebook input.
3. Clone this repository and run:

```bash
bash run_kaggle.sh beauty
```

For Amazon-Book:

```bash
bash run_kaggle.sh book /kaggle/input/<amazon-book-folder>
```

The defaults run seeds `42 123 456 789 1024`, embedding dimension 64, every
eligible training instance, and every evaluation user. Beauty uses full-item
ranking. Amazon-Book uses one positive plus 99 fixed negatives because full
ranking over 91,599 items is costly.

To run one seed first:

```bash
SEEDS="42" EPOCHS=50 bash run_kaggle.sh beauty
```

Outputs are written to `/kaggle/working/adafuse`. Re-running the command skips
completed seeds and resumes available per-model checkpoints.

## Dataset caveat

Beauty contains timestamps and supports temporal sequential evaluation.
Amazon-Book's checked-in LightGCN-format split has no timestamps; its list
order is synthetic, so SASRec/SASGCN/AdaFuse results must not be presented as
a valid temporal recommendation experiment.

Raw datasets, model checkpoints, and generated results are intentionally not
stored in Git. Attach datasets through Kaggle and regenerate all outputs with
the script above.

