# WorldStrat Downloading and Features Extraction

Scripts and instructions for downloading and then extracting features from the WorldStrat dataset using DINOv2.

## Overview

The WorldStrat dataset includes Sentinel (low resolution) and SPOT (high resolution) images. This folder provides scripts to preprocess these images and extract features using the DINOv2 model. We only used Sentinel L2A images and SPOT pansharpened images, and only with RGB channels (Sentinel L2A have 12 channels).

## Setup

Follow the main README instructions:
1. Clone the repository.
2. Install the environment.
3. Create a `.env` file containing `DATA_ROOT`, the path to the folder that holds `satellite_dataset` (for example `DATA_ROOT=/path/to/data`).

All Python scripts accept `--data-root` and fall back to the `DATA_ROOT` environment variable when it is not provided. If neither is set, the script exits with an error. Some scripts expect `--data-root` to be the folder that contains `satellite_dataset`, while others expect `satellite_dataset` itself or the `Vertibench` folder; see each script's `--help` output.

## Dataset Download and Extraction

1. Open `Dataset_download_and_extract.ipynb`.
2. Set `DOWNLOAD` to `True` and run the cell to download the dataset to `$DATA_ROOT/satellite_dataset`.
3. Verify file integrity by running the MD5 hash check cell.
4. Set `EXTRACT` to `True` and run the cell to extract the dataset into `$DATA_ROOT/satellite_dataset`.

## Calculating Mean and Standard Deviation

1. Ensure the dataset is downloaded and extracted.
2. Run the appropriate script to calculate mean and standard deviation for each channel:
   - Sentinel L2A: `python Satellite_mean_std.py --data-root /path/to/data`
   - SPOT Pansharpened: `python Satellite_PS_mean_std.py --data-root /path/to/data`
   - Vertibench: `python Vertibench_mean_std.py --data-root /path/to/data/satellite_dataset/Vertibench`

## Feature Extraction

1. Ensure the dataset is downloaded and extracted.
2. Run the appropriate script to preprocess images and extract class tokens. `--data-root` is the `satellite_dataset` folder; `--output-dir` and `--batch-size` are optional:
   - SPOT Pansharpened: `python normalize_and_extract_features_PS.py --data-root /path/to/data/satellite_dataset`
   - Sentinel L2A: `python normalize_and_extract_features_sentinel.py --data-root /path/to/data/satellite_dataset`
   - Vertibench: `python extract_features_vertibench.py --data-root /path/to/data/satellite_dataset`

## Third-Party Code

`dataset_download_tools.py` is derived from the [WorldStrat repository](https://github.com/worldstrat/worldstrat), Copyright (c) 2022 WhyHow Ltd., and is distributed under the BSD 3-Clause License. See `THIRD_PARTY_LICENSES/WorldStrat-BSD-3-Clause.txt` for the full license text.

## References

- [WorldStrat Dataset](https://github.com/worldstrat/worldstrat/tree/main)
- [Vertibench Dataset](https://vertibench.xtra.science/datasets/1/)
- [DINOv2](https://github.com/facebookresearch/dino)
