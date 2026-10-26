# Results

The satellite numbers below are the mean ± standard deviation over seeds from the original experiment logs (2024) and are not a freshly re-run benchmark. The UCI Credit baseline was re-run on the public code with the real dataset.

| Setting | Parties | Seeds | Accuracy | AUC |
|---|---:|---:|---:|---:|
| UCI Credit Card, vertical | 2 | 5 | 81.89 ± 0.07 | 76.79 ± 0.30 |
| Sentinel L2A features, vertical | 16 | 5 | 84.66 ± 1.22 | 95.02 ± 0.51 |
| Sentinel L2A features, horizontal | 1 (random view) | 5 | 94.57 ± 0.09 | 98.84 ± 0.09 |
| SPOT features | 1 | 20 | 88.40 ± 0.88 | 96.83 ± 0.30 |
| SPOT + Sentinel, aggregation | 17 | 5 | 98.17 ± 0.47 | 99.99 ± 0.00 |
| VertiBench, vertical | 16 | 5 | 83.62 ± 0.76 | 94.39 ± 0.30 |
| VertiBench, horizontal | 1 (random view) | 5 | 80.76 ± 0.30 | 92.81 ± 0.08 |
| VertiBench, isolated party | 1 | 5 | 78.85 ± 0.90 | 91.32 ± 0.54 |

## How to reproduce the aggregation

Hydra writes one `main_pipeline.log` per run under `outputs/`. Aggregate the test metrics with:

```bash
python scripts/aggregate_results.py outputs/multirun/CreditDataset
```

## Verified UCI Credit baseline

Run on the public code with the real UCI Credit Card dataset (30,000 rows, 22.1% positive class), 5 seeds:

```bash
simple-vfl -m --config-name=credit runtime.seed=0,1,2,3,4
```

Result: **81.89% ± 0.07 accuracy**, **76.79% ± 0.30 AUC** on the held-out test set.

## Caveats

- The image normalisation statistics were computed over the full dataset (train + test) in the original feature-extraction scripts. For a strict protocol, recompute them on the training split only.
- The very high SPOT + Sentinel score reflects the complementary views of the same POIs and the strength of the DINOv2 features; it is not a claim of state-of-the-art performance.
- The splits are grouped by POI, so images from the same POI do not appear in both train and test.
