# Results

## Main result: frozen foundation features for VFL

The one-shot result is the **concatenation** protocol: each party's frozen
features are uploaded once, then the coordinator trains its head on the collected
feature matrix. The 16-view Sentinel concat row below follows this design. The
late-fusion rows are additional iterative comparisons that exchange logits per
batch; they are not one-shot results. All runs are simulated in one process, and
communication is estimated from payload sizes.

**Scope of the claim.** On 30 October 2026 we searched arXiv (full text and
API), OpenAlex title/abstract, GitHub repository search and the Hugging Face Hub
for public work that evaluates a vertical federated split on a satellite
benchmark using frozen foundation-model embeddings as the party-side
representation. We found none. This is a negative search result, not a proof of
absence, and it claims no novelty for any component: frozen pretrained
backbones for federated image classification (Hu et al., ICCV 2023, a
horizontal-FL study), the 16-party satellite split (VertiBench, ICLR 2024),
open-source split learning of large pretrained models (VFLAIR-LLM, KDD 2025)
and frozen satellite foundation features (DINOv3-Sat, 2025) are each published
separately. A **frozen DINOv2 ViT-g/14 backbone**, run
locally at each party, gives a compact 1536-d embedding that is competitive with
end-to-end SplitNN training. In the concatenation models the coordinator trains
the fusion head and no gradient goes back to the party; in the late-fusion
models the forward pass sends 4-d logits and the backward pass returns 4-d logit
gradients. Raw images never leave the party. No secure aggregation, DP or
encryption is implemented.

| Setting | Parties | Accuracy |
|---|---:|---:|
| VertiBench SplitNN (paper) | 16 | 81.17 ± 0.35 |
| Frozen DINOv2 + one-hidden-layer MLP, vertical | 16 | 82.65 ± 0.75 |
| Frozen DINOv2 + one-hidden-layer MLP, concat | 16 | 83.62 ± 0.76 |

The two DINOv2 rows differ only in the head over the concatenated 16 × 1536
features (two-layer MLP vs a single linear layer); the 2024 archive does not
record which run used which head, so the re-run replaces both rows.

VertiBench's 16 parties are 16 revisits of the same AOI (same sensor, same
location, different dates): multi-view fusion, not cross-silo learning over
disjoint populations, and no entity-resolution test. No PSI or alignment step
is performed.

These two DINOv2 rows are 2024 runs on the VertiBench partition. This is a
**2024 preliminary** result: the feature archive and raw logs are not yet in the
public repo, and a fresh re-run is pending. The comparison is confounded: 3 RGB
bands vs 13 channels, a different train/val split and optimizer, and a frozen
vs end-to-end backbone. We make no ranking claim until the re-run; the numbers
are in the same range. The important point is that a frozen foundation encoder plus a light
head is competitive without sharing raw images.

### Communication budget

One-time upload payload, float32: a DINOv2 class token is 1536 × 4
bytes = **6 KiB**. A raw Sentinel L2A patch
(12 × 158 × 158 × uint16) is about **0.571 MiB**, and a SPOT pansharpened patch
(4 × 1050 × 1050 × uint16) about **8.4 MiB**. The 17-view SPOT + Sentinel
payload is therefore ~102 KiB of embeddings instead of ~17.55 MiB of raw
patches, a ~100× reduction for 16 Sentinel views and ~180× for SPOT + 16.
DINOv2 uses a 3-band RGB subset, so the like-for-like reduction is ~24× and
~86×.
These are embedding-level upper bounds: the iterative late-fusion models send 4-d logits
(256 B for 16 parties, 272 B for SPOT + 16 Sentinel on the 4-class tasks;
448 B / 476 B on the 7-class IPCC task). The full table is in the
[technical guide](docs/TECHNICAL.md); protocol overhead and encryption are not included.

### Related work

- Hu et al., *FL Over Images: Vertical Decompositions and Pre-Trained Backbones
  Are Difficult to Beat* (ICCV 2023, arXiv:2309.03237): horizontal FL over
  images, frozen ResNet/DenseNet backbones. Its "vertical decomposition" splits
  the network, not the features across parties; it constrains the
  frozen-backbone claim, not the VFL protocol.
- VFLAIR (ICLR 2024, arXiv:2310.09827) and VertiBench (ICLR 2024): the open VFL
  library and the 16-revisit satellite split used here.
- VFLAIR-LLM (KDD 2025, doi:10.1145/3711896.3737411): open-source split learning
  of LLMs, so open VFL + large models is not new.
- FeT (NeurIPS 2024, arXiv:2410.17986) and LASER-VFL (ICLR 2025): fuzzy linkage
  and missing parties; we assume exact alignment and full participation.
- DINOv3-Sat (2025, arXiv:2508.10104): frozen satellite features are the
  standard strong baseline in Earth observation.
- Wu et al., *VFL in Practice* (2025, arXiv:2502.08160): the deployment gap;
  our benchmark is in the aligned, complete-partition regime.

WorldStrat (NeurIPS 2022) provides the SPOT + Sentinel imagery; DINOv2 (2023)
provides the backbone. We do not claim the first VFL benchmark, the first
foundation-model VFL method, or the first frozen-backbone VFL result.

## Exploratory WorldStrat benchmark

The table below shows how the same frozen features behave under different
vertical, horizontal and multi-sensor protocols. Unless stated otherwise:
**5 seeds, mean ± standard deviation**, test split of 589 POIs (2,668 train /
668 validation), one run per seed.

| Protocol | Parties / encoders | Views per sample | Model | Accuracy | AUC |
|---|---|---:|---|---:|---:|
| Sentinel, vertical, concatenation | 16 inputs, 1 head | 16 | `24576 → 1000 → 4` | **85.37 ± 0.70** | 95.76 ± 0.16 |
| Sentinel, vertical, mean logits | 16 encoders | 16 | `16 × (1536 → 1000 → 4)`, mean | **85.16 ± 0.60** | 96.09 ± 0.18 |
| Sentinel, horizontal proxy | 1 | 1 (random revisit) | `1536 → 1000 → 4` | **80.51 ± 0.86** | 93.17 ± 0.51 |
| SPOT only | 1 | 1 | `1536 → 1000 → 4` | **88.96 ± 1.04** | 97.21 ± 0.17 |
| SPOT + 16 Sentinel, weighted logits | 2 encoders (SPOT + shared Sentinel) | 17 | weighted sum, `sentinel_weight=0.5` | **89.61 ± 0.36** | 97.33 ± 0.09 |
| SPOT + 1 Sentinel | 1 shared encoder | 2 | shared encoder, concat, `2000 → 4` | **86.32 ± 0.63** | 96.74 ± 0.30 |
| SPOT + Sentinel, horizontal proxy | 1 | 1 of 17 | `1536 → 1000 → 4` | **80.75 ± 0.87** | 93.58 ± 0.63 |

The 5-seed runs above were executed on CPU (a local 20-core machine for the
SPOT + Sentinel variants and the Kaggle CPU runner for the baseline sweeps).
The five-seed IPCC and simplified-IPCC runs were executed on a Kaggle T4.

### What the exploratory table says

1. **Using all Sentinel revisits helps, but only by a few points.** The
   16-view vertical model reaches 85.37, against 80.51 for one random revisit
   (+4.9). The mean-logit control (16 party encoders, 24.6M parameters)
   reaches 85.16, so the fusion design is not the driver. The 16-view model
   still has 16x the head parameters of the 1-view model, so this is a
   *views-plus-capacity* effect, not evidence that federation itself improves
   accuracy.
2. **A single high-resolution sensor is stronger than 16 low-resolution
   revisits.** SPOT alone reaches 88.96, above every Sentinel-only protocol.
3. **Combining SPOT with Sentinel gives a small gain; its strength depends on
   the label.** On `sources`, the five-seed fusion reaches 89.61 ± 0.36,
   against 88.96 ± 1.04 for SPOT alone (+0.65 accuracy, +0.12 AUC), but the
   paired differences are within seed noise (paired t-test: p ≈ 0.39 accuracy,
   p ≈ 0.28 AUC; the fusion loses on seed 3). It has the **best image-model AUC
   (97.33)**, though the location-only baseline is higher (98.00). On IPCC
   (five seeds), fusion reaches 78.00 ± 0.55 / 95.04 ± 0.14, vs 76.20 ± 0.77 /
   94.21 ± 0.11 for SPOT alone (+1.80 accuracy, +0.83 AUC). Paired t-tests give
   p ≈ 0.008 and p < 0.001; exact two-sided Wilcoxon is p = 0.0625 for both
   metrics (n=5), so treat the evidence cautiously. The fusion has ~3.1M
   parameters and ensembles 17 logit vectors (68 scalars for 4 classes) vs
   ~1.5M for SPOT-only, so part of the gain may be capacity/ensemble; a
   capacity-matched SPOT control is future work.
4. **The horizontal proxy is not horizontal FL.** It is a centralized model
   that sees one view per POI; there are no clients and no FedAvg. It is a
   lower bound for a client that owns a single image per location.

### Ablations (same split, 5 seeds unless noted)

| Variant | Accuracy | Note |
|---|---:|---|
| SPOT + 16 Sentinel, shared encoder over all 17 views (`SpecialModel`) | 85.77 ± 0.71 | overfits 17 × 1000 inputs |
| SPOT + 16 Sentinel, raw concat (no shared encoder) | 86.76 (seed 0) | 26,112-dim input |
| SPOT + 16 Sentinel, shared Sentinel encoder + concat embeddings | 86.93 (seed 0) | 17 × 1000 input |
| SPOT + 16 Sentinel, mean of Sentinel embeddings | 88.96 (seed 0) | 2,000-dim input |
| `sentinel_weight` sweep on seed 0 | 0.1 → 89.13, 0.25 → 89.47, 0.5 → 89.81, 2 → 89.13, 4 → 88.12, 8 → 87.61 | 0.5 selected |

## Discussion: dataset challenges (work in progress)

The WorldStrat benchmark is **not finished**. The foundation-feature pipeline
is stable; the dataset protocol is the open work.

VFL image benchmarks are scarce because the natural ones (health, finance) are
sensitive. Most public simulations either split a single image into
patches/channels or treat revisits as parties; neither reflects real data
ownership. VertiBench's 16 Sentinel revisits as 16 parties is the clearest
example: same sensor, same AOI, different dates.

WorldStrat lets us test a sensor-level split: one SPOT party and one Sentinel
party holding 16 revisits. On the semantic IPCC label, the one-shot 16-view
Sentinel model reaches **77.93 ± 0.52 accuracy / 94.13 ± 0.53 AUC** (five
seeds), vs **70.66 / 86.49** for the location-only random forest. The separate
SPOT + Sentinel late-fusion comparison reaches **78.00 ± 0.55 / 95.04 ± 0.14**.
Adding SPOT to Sentinel gives **+1.80 accuracy / +0.83 AUC** over SPOT alone
on IPCC (+0.65 / +0.12 on `sources`),
and using all 16 revisits gives **~+4.3** over one revisit on IPCC (+4.9 on
`sources`). That is a real signal, but a modest one: the dataset is
**label-limited**, the revisits are likely highly correlated, and `sources` is a geographic
shortcut. We use WorldStrat for an exploratory two-sensor benchmark; it is not a
mature benchmark, and So2Sat LCZ42 and BigEarthNet are public multi-sensor
alternatives.

**The labels are geographic shortcuts.** The 4 `sources` classes are provider
names (Amnesty, ASMSpotter, Landcover, UNHCR), and they are spatially
clustered. A random forest on `lon`/`lat` alone reaches **91.71%** accuracy on
the same test split, above the best image model (89.61%); the image numbers are
therefore not yet a clean measure of remote-sensing value on that label.

Across the three supported labels, the AUC gap between the best image model and
the location-only baseline flips sign:

| Label | Classes | Location-only AUC | Best image AUC | Gap |
|---|---:|---:|---:|---:|
| `sources` | 4 | 98.00 | 97.33 | **-0.67** |
| `simplified_ipcc` | 4 | 87.09 | 93.70 | **+6.61** |
| `ipcc` | 7 | 86.49 | 95.04 | **+8.55** |

(Seed-matched: location-only RF and image models use 5 seeds for all three labels. POIs with a missing
`IPCC Class` are mapped to `Other`, matching the image pipeline.)

On `sources`, geography is a better ranking signal than the images. On the
semantic labels the image model is ahead, and the gap is larger on the
land-cover labels; this is suggestive, not a controlled test of semantic content. This is why the dataset work is moving to IPCC.

On `simplified_ipcc`, the SPOT + Sentinel fusion is also the best image model
(**78.98 ± 0.39%** accuracy, **93.70 ± 0.22 AUC**, five seeds), ahead of SPOT
(78.24 ± 1.04% / 92.85 ± 0.20), one-shot Sentinel 16 views
(78.20 ± 0.41% / 93.17 ± 0.15) and Sentinel 1 view
(74.63 ± 1.22% / 91.31 ± 0.55). The paired fusion-vs-SPOT accuracy difference
is not clear at five seeds (t-test p ≈ 0.31); all five AUC differences are
positive (paired t-test p < 0.001; exact Wilcoxon p = 0.0625).

A semantic label changes the picture. On the 7-class IPCC task (5 seeds,
Kaggle T4), the same frozen pipeline beats the location-only baseline:

| IPCC setting | Accuracy | AUC |
|---|---:|---:|
| Location-only RF (`lon`/`lat`) | 70.66 ± 0.63 | 86.49 ± 0.86 |
| Sentinel, 16 views, concat (one-shot) | **77.93 ± 0.52** | 94.13 ± 0.53 |
| Sentinel, 1 view | 73.75 ± 1.14 | 92.86 ± 0.54 |
| SPOT only | 76.20 ± 0.77 | 94.21 ± 0.11 |
| SPOT + 16 Sentinel (late fusion) | 78.00 ± 0.55 | **95.04 ± 0.14** |

IPCC class counts are Settlement 1,700, Forest 733, Agriculture 699, Other 489
(including 71 POIs with missing `IPCC Class`), Grassland 205, Water 58 and
Wetland 41; a majority-class predictor gets about 44% accuracy, so accuracy
alone is not enough and AUC is the primary metric.

The one-shot 16-view Sentinel model adds **+7.27 accuracy points / +7.64 AUC
points** over the location-only RF. The separate SPOT + Sentinel fusion adds
**+8.55 AUC points** (95.04 vs 86.49) and has the **best image-model AUC
(95.04 ± 0.14)**; its accuracy is 78.00 vs 77.93 for the one-shot model. On an imbalanced
7-class label, AUC is the more informative summary. This is the direction the
dataset work is taking: a semantic label, then a geographically disjoint split
to remove the remaining location overlap.

A `sentinel_weight` sweep on IPCC (0.25, 0.5, 1, 2; 3 seeds, separate from the
five-seed main runs) leaves AUC at
94.92–95.06 and accuracy at 77.65–78.21. Weights 0.5 and 1.0 tie at the best
AUC (95.06), so 0.5 remains the default; the fusion is robust to this knob.

**The revisits are likely highly correlated.** On `sources`, 16 Sentinel views add
+4.9 points over one view, but SPOT alone (88.96%) beats every Sentinel-only
protocol; on IPCC the 16-vs-1 gain is +4.3. The concat model also has 16x the
head parameters of the 1-view model, so the gain is views-plus-capacity; a
capacity-matched 1-view control is future work. The interesting heterogeneity
is across sensors and regions, not across revisits of the same AOI.

Next steps for the dataset work:

- a geographically disjoint split (hold out regions/countries) so test POIs
  are not in the same regions as training POIs;
- add a five-seed SPOT + Sentinel concat run to directly test the one-shot
  transfer on the sensor-level split;
- a harder semantic label or additional sensors, to move past the
  label-limited regime.

## Reproducibility manifest

- **Code**: `main` is the single public commit `58267cb`. The 2024 VertiBench
  numbers predate this repository. The `sources` runs and original ablations
  were produced on pre-publication trees (`1f1a343` for IPCC, `3cf04fc` for the
  weight sweep, `d884c29` for simplified-IPCC). The five-seed IPCC and
  simplified-IPCC runs were freshly reproduced on `58267cb` with
  `scripts/kaggle/repro_ipcc5.py` on a Kaggle T4; raw logs are stored locally in
  the git-excluded `experiments/logs/ipcc_5seed/` folder.
- **Data**: WorldStrat `metadata.csv` from Zenodo record 6810792; DINOv2
  features from
  [Hugging Face](https://huggingface.co/datasets/xavvierxav/L1C_dataset)
  (`extracted_features_giant_rgb_light_preprocess.zip` for Sentinel,
  `extracted_features_giant_ps_normalized.7z` for SPOT).
- **Splits**: 3,925 POIs after dropping `ASMSpotter-1-1-1` (no Sentinel
  features), `Landcover-1062091` and `UNHCR-HUNs002665` (already excluded in
  the original normalisation scripts); 2,668 train / 668 validation / 589 test,
  grouped by POI,
  identical within each label. The train/validation boundary depends on
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
  `spot_and_sentinel_horizontal_2views`. The IPCC and simplified-IPCC runs use
  the same command with `dataset.label_type=ipcc dataset.num_classes=7` (or
  `dataset.label_type=simplified_ipcc dataset.num_classes=4`) and the same
  five seeds.
- **Location baseline**:

  ```bash
  DATA_ROOT=/path/to/data \
  python scripts/location_baseline.py --label sources --seeds 0,1,2,3,4
  DATA_ROOT=/path/to/data \
  python scripts/location_baseline.py --label ipcc --seeds 0,1,2,3,4
  DATA_ROOT=/path/to/data \
  python scripts/location_baseline.py --label simplified_ipcc --seeds 0,1,2,3,4
  ```
- **Aggregation**: `python scripts/aggregate_results.py <sweep folder>`.
- **Hardware**: the five-seed `sources` table was run on CPU (local 20-core and
  Kaggle CPU runners); five-seed IPCC and simplified-IPCC runs were run on a
  Kaggle T4 (`xavierdesprez/simple-vfl-ipcc5`). The GPU runner copies a single
  feature tar to local disk, so the Kaggle mount is not the bottleneck.

## Caveats

- Image normalisation statistics were computed over the full dataset in the
  original feature extraction. A strict protocol would recompute them on the
  training split only.
- SPOT and Sentinel are two sensors observing the same POIs, not two
  independent data owners. The two-sensor result is multi-view fusion, not
  cross-silo learning on disjoint populations.
- The WorldStrat table does not include an end-to-end raw-image SplitNN on the
  same split; the frozen-feature claim is supported by the preliminary
  VertiBench comparison.
- This is a single-machine simulation of the VFL architecture. Party models are
  separate and only embeddings/logits are aggregated; communication cost is
  computed, not incurred. There is no secure aggregation, differential
  privacy, or encrypted communication. A secure version would need MPC-based
  aggregation with output DP (Jin et al., PoPETs 2026, arXiv:2604.13474).
- The SPOT + Sentinel gain is small (+0.65 accuracy on `sources`, +1.80 on
  IPCC). On `sources` the paired differences are within seed noise (paired
  t-test: p ≈ 0.39 accuracy, p ≈ 0.28 AUC; the fusion loses on seed 3). The
  fusion has the best AUC among image models on every label, but on `sources`
  the location-only baseline is higher (98.00 vs 97.33). On IPCC (paired
  t-test, n=5), the gain over SPOT alone is +1.80 accuracy (p ≈ 0.008) and
  +0.83 AUC (p < 0.001); exact two-sided Wilcoxon is p = 0.0625 for both.
  On simplified-IPCC, the +0.74 accuracy difference is not clear (p ≈ 0.31),
  while AUC is higher across all five seeds (paired t-test p < 0.001; exact
  Wilcoxon p = 0.0625). The fusion has ~3.1M parameters and ensembles 17 logit
  vectors (68 scalars for 4 classes) vs ~1.5M for SPOT-only, so part of the
  gain may be capacity/ensemble.
- The 16-view concat model has 24.6M parameters vs 1.5M for the 1-view model,
  so the +4.9 gain is views-plus-capacity; a capacity-matched 1-view control is
  future work.
- `sentinel_weight=0.5` was selected on seed 0's test accuracy, so the seed
  standard deviation does not include model-selection variance.
