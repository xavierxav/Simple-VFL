# Simple-VFL

[![CI](https://github.com/xavierxav/Simple-VFL/actions/workflows/ci.yml/badge.svg)](https://github.com/xavierxav/Simple-VFL/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**One-shot vertical federated learning for satellite imagery.** Each party
extracts features from its own images with a frozen foundation model, uploads
its complete feature matrix to a coordinator **once**, and keeps the raw pixels.
The coordinator trains a small prediction head on the received features; its
later training epochs require no further communication with the parties.

## The idea

```text
Party A: local images ── frozen DINOv2 ──┐
                                         ├── one feature upload ──> Coordinator: train head
Party B: local images ── frozen DINOv2 ──┘
```

For DINOv2 ViT-g/14, one image becomes a 1536-dimensional feature (6 KiB in
float32). Sixteen Sentinel views therefore mean **96 KiB per location sent
once**, rather than repeatedly exchanging activations during head training.
That is about **100× smaller** than sending the full 12-band patches, or ~24×
smaller against an equivalent 3-band RGB input.
The central training loop in this repository simulates the post-upload stage;
the communication payload is calculated from tensor sizes, not measured over a
network.

To our knowledge, this one-shot, frozen-feature setup has not yet been evaluated
on an open satellite VFL benchmark.

## Why a new image-VFL benchmark?

Real datasets with vertically held images are hard to obtain. Many image-VFL
benchmarks work around this by cutting one image into patches or channel groups
and assigning the pieces to artificial parties. That is useful for testing
algorithms, but it is not a natural model of separate data owners.

**VertiBench** is the closest public satellite VFL benchmark we know: it treats
16 Sentinel revisits of the same WorldStrat locations as parties. We use it as
a reference, then build a sensor-level benchmark on WorldStrat: one party holds
high-resolution **SPOT** imagery and another holds **Sentinel-2** revisits.
Different sensors make a more natural division of views between parties.

We also use WorldStrat's **IPCC land-cover labels**, rather than relying only on
provider-name labels. The distinction matters: on `sources`, a random forest
using just longitude and latitude reaches **91.71% accuracy / 98.00 AUC**, above
the best image model. On IPCC, the same location-only baseline is **70.66 ±
0.63% / 86.49 ± 0.86 AUC**; the one-shot 16-view Sentinel model reaches
**77.93 ± 0.52% / 94.13 ± 0.53 AUC** (5 seeds). This makes the semantic task
a more informative test of what the image features add beyond geography.

## Results at a glance

| Experiment | Result | Comparison |
|---|---:|---|
| VertiBench, frozen DINOv2 | 82.65–83.62% accuracy | 81.17 ± 0.35% for the paper's SplitNN; **2024 preliminary runs, re-run pending** |
| WorldStrat IPCC, one-shot Sentinel features | 77.93 ± 0.52% accuracy / 94.13 ± 0.53 AUC | Location-only: 70.66 ± 0.63% / 86.49 ± 0.86 AUC (5 seeds) |
| WorldStrat IPCC, SPOT + Sentinel late fusion | 78.00 ± 0.55% accuracy / **95.04 ± 0.14 AUC** | Additional iterative protocol; not the one-shot method (5 seeds) |

The VertiBench result is a promising historical comparison, not a controlled
claim of beating SplitNN: the original runs used a different setup and are
being re-run. The WorldStrat sensor-fusion row is also reported separately
because late fusion exchanges logits during training; it is not one-shot.

## Quickstart

```bash
git clone https://github.com/xavierxav/Simple-VFL.git
cd Simple-VFL
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
cp .env.example .env  # set DATA_ROOT

simple-vfl --config-name=satellite \
    +experiment_satellite_features=sentinel
```

The smoke test needs no data:

```bash
simple-vfl --config-name=credit epochs=2 batch_size=128
```

Pre-extracted WorldStrat features and VertiBench arrays are available on
[Hugging Face](https://huggingface.co/datasets/xavvierxav/L1C_dataset) (~8.6 GB).
See the [data and setup guide](docs/TECHNICAL.md#data) before running satellite
experiments.

## In this repository

- Frozen DINOv2 feature extraction and satellite VFL experiments.
- Concatenation, late-fusion, mean-logit and shared-encoder models.
- WorldStrat location-only baseline and POI-grouped evaluation splits.
- Hydra configs, multi-seed runs, TensorBoard, tests and CI (Python 3.10–3.12).

| Read next | |
|---|---|
| [RESULTS.md](RESULTS.md) | Full tables, ablations, IPCC analysis and caveats |
| [Technical guide](docs/TECHNICAL.md) | One-shot protocol, architecture, data, configs and reproduction |
| [ATTRIBUTION.md](ATTRIBUTION.md) | Code and data attribution |

The project is research code: experiments run in a single process and the
one-time upload is represented by an analytical payload estimate, not a
network transport or a security guarantee.

## License and thanks

The original code is MIT-licensed; third-party code and data retain their own
terms. See [ATTRIBUTION.md](ATTRIBUTION.md) for details. Thanks to the VFLAIR,
WorldStrat, DINOv2, VertiBench and UCI Credit Card teams.

Xavier Desprez — [github.com/xavierxav](https://github.com/xavierxav)
