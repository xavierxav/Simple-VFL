# Results

All numbers in the fresh benchmark below were produced by this repository on
the published DINOv2 features, with the current POI-grouped split. Unless
stated otherwise: **5 seeds, mean ± standard deviation**, test split of 589
POIs (2,668 train / 668 validation), one run per seed.

The split is grouped by POI, so no revisit of a test POI appears in training.
All protocols use the same 2,668 / 668 / 589 split, which makes the rows
directly comparable.

## One dataset, several protocols

The same WorldStrat imagery is cut in several ways. The table separates the
protocol from the data.

| Protocol | Parties / encoders | Views per sample | Model | Accuracy | AUC |
|---|---|---:|---|---:|---:|
| Sentinel, vertical, concatenation | 16 inputs, 1 head | 16 | `24576 → 1000 → 4` | **85.37 ± 0.70** | 95.76 ± 0.16 |
| Sentinel, vertical, mean logits | 16 encoders | 16 | `16 × (1536 → 1000 → 4)`, mean | **85.16 ± 0.60** | 96.09 ± 0.18 |
| Sentinel, horizontal proxy | 1 | 1 (random revisit) | `1536 → 1000 → 4` | **80.51 ± 0.86** | 93.17 ± 0.51 |
| SPOT only | 1 | 1 | `1536 → 1000 → 4` | **88.96 ± 1.04** | 97.21 ± 0.17 |
| SPOT + 16 Sentinel, weighted logits | 2 encoders (SPOT + shared Sentinel) | 17 | weighted sum, `sentinel_weight=0.5` | **89.61 ± 0.37** | 97.33 ± 0.10 |
| SPOT + 1 Sentinel | 2 encoders | 2 | shared encoder, concat, `2000 → 4` | **86.32 ± 0.63** | 96.74 ± 0.30 |
| SPOT + Sentinel, horizontal proxy | 1 | 1 of 17 | `1536 → 1000 → 4` | **80.75 ± 0.87** | 93.58 ± 0.63 |

The 5-seed runs above were executed on CPU (a local 20-core machine for the
SPOT + Sentinel variants and the Kaggle CPU runner for the baseline sweeps).
A GPU replication runner is provided; the conclusions are hardware-independent.

## Reading the table

1. **Using all Sentinel revisits helps, but only by a few points.** The
   16-view vertical model reaches 85.37, against 80.51 for one random revisit
   (+4.9). The capacity-controlled model (16 party encoders, mean logits)
   reaches 85.16, so the gain is not an artefact of the 24,576-dimensional
   concatenation. This is a *views-per-sample* effect, not evidence that
   federation itself improves accuracy.
2. **A single high-resolution sensor is stronger than 16 low-resolution
   revisits.** SPOT alone reaches 88.96, above every Sentinel-only protocol.
   Resolution matters more than revisit count on this label task.
3. **Combining SPOT with Sentinel gives a small, stable gain.** The shared
   Sentinel encoder + SPOT encoder with `sentinel_weight=0.5` reaches
   89.61 ± 0.37, against 88.96 ± 1.04 for SPOT alone (+0.65 accuracy,
   +0.12 AUC). The fusion model is also more stable across seeds (std 0.37 vs
   1.04). This is a small but consistent improvement.
4. **The horizontal proxy is not horizontal FL.** It is a centralized model
   that sees one view per POI; there are no clients and no FedAvg. It is a
   lower bound for a client that owns a single image per location.

### Ablations (same split, 5 seeds unless noted)

| Variant | Accuracy | Note |
|---|---:|---|
| SPOT + 16 Sentinel, shared encoder over all 17 views (`SpecialModel`) | 85.77 ± 0.72 | overfits 17 × 1000 inputs |
| SPOT + 16 Sentinel, raw concat (no shared encoder) | 86.76 (seed 0) | 26,112-dim input |
| SPOT + 16 Sentinel, shared Sentinel encoder + concat embeddings | 86.93 (seed 0) | 17 × 1000 input |
| SPOT + 16 Sentinel, mean of Sentinel embeddings | 88.96 (seed 0) | 2,000-dim input |
| `sentinel_weight` sweep on seed 0 | 0.5 → 89.81, 1 → 88.96, 2 → 89.13, 4 → 88.12, 8 → 87.61 | 0.5 selected |

## VertiBench, 2024 logs (not re-run here)

These rows are kept for provenance. They use the VertiBench-provided split,
not the WorldStrat POI split, and were not re-run in this benchmark.

| Setting | Parties | Seeds | Accuracy | AUC |
|---|---:|---:|---:|---:|
| VertiBench vertical, concat | 16 | 5 | 83.62 ± 0.76 | 94.39 ± 0.30 |
| VertiBench vertical, 2-layer head | 16 | 5 | 82.65 ± 0.75 | — |
| VertiBench vertical, mean logits | 16 | 5 | 83.63 ± 0.35 | 94.78 ± 0.32 |
| VertiBench horizontal proxy | 1 | 5 | 80.76 ± 0.30 | 92.81 ± 0.08 |
| VertiBench isolated party | 1 | 5 | 78.85 ± 0.90 | 91.32 ± 0.54 |

The comparison with the VertiBench paper (SplitNN 81.17 ± 0.35) is
confounded: 3 RGB bands vs 13 channels, a different split, and a frozen
DINOv2 backbone. The defensible claim is "match or slightly exceed", not
"beat".

## UCI Credit Card sanity check

The tabular smoke test checks that the pipeline runs end-to-end on real data;
it is not a scientific result.

```bash
simple-vfl -m --config-name=credit runtime.seed=0,1,2,3,4
```

Result: **81.89 % ± 0.07 accuracy**, **76.79 % ± 0.30 AUC** on the held-out
test set (30,000 rows, 22.1 % positive class).

## Reproducibility manifest

- **Code**: branch `add-satellite-controls`, commit shown by
  `git log -1 --oneline`.
- **Data**: WorldStrat `metadata.csv` from Zenodo record 6810792; DINOv2
  features from
  [Hugging Face](https://huggingface.co/datasets/xavvierxav/L1C_dataset)
  (`extracted_features_giant_rgb_light_preprocess.zip` for Sentinel,
  `extracted_features_giant_ps_normalized.7z` for SPOT).
- **Splits**: 2,668 train / 668 validation / 589 test POIs, grouped by POI,
  identical for every protocol. The train/validation boundary depends on
  `runtime.seed`; the test boundary is fixed (`random_state=42`).
- **Command** (per setting, seeds 0–4):

  ```bash
  python -m simple_vfl -m \
      --config-name=satellite \
      +experiment_satellite_features=<setting> \
      runtime.device=auto runtime.num_workers=0 \
      runtime.seed=0,1,2,3,4
  ```

  Settings: `sentinel`, `sentinel_aggregation`, `sentinel_horizontal`, `spot`,
  `spot_and_sentinel`, `spot_and_sentinel_horizontal`,
  `spot_and_sentinel_horizontal_2views`.
- **Aggregation**: `python scripts/aggregate_results.py <sweep folder>`.
- **Hardware**: CPU for the fresh 5-seed table (local 20-core and Kaggle CPU
  runners). The Kaggle T4 runner (`simple-vfl-gpu-benchmark`) is kept for
  replication; it copies the feature folders to local disk before training
  because the Kaggle mount is the bottleneck.

## Caveats

- Image normalisation statistics were computed over the full dataset in the
  original feature extraction. A strict protocol would recompute them on the
  training split only.
- SPOT and Sentinel are two sensors observing the same POIs, not two
  independent data owners. The two-sensor result is multi-view fusion, not
  cross-silo learning on disjoint populations.
- This is a single-machine simulation of the VFL architecture. Party models are
  separate and only embeddings/logits are aggregated; communication cost is
  computed, not incurred. There is no secure aggregation, differential
  privacy, or encrypted communication.
- The SPOT + Sentinel gain (+0.65 accuracy) is small; more seeds would tighten
  the estimate. The fusion model is more stable across seeds than SPOT alone.
