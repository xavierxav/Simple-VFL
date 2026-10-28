# Configuration parameters

All experiments use [Hydra](https://hydra.cc/). Defaults live in `simple_vfl/config/standard/standard.yaml`; dataset-specific values live in the other YAML files.

## Training parameters

- **epochs**
  Number of training epochs. Default: `10`.
- **lr**
  Initial learning rate. Default: `0.001`.
- **batch_size**
  Batch size for training and evaluation. Default: `512`.
- **k**
  Number of parties (or local models) in the federated setting. Default: `2`.
- **weight_decay**
  Adam weight decay. If omitted or `null`, no weight decay is used.
- **hidden_dim**
  Hidden dimension for models that use an MLP head.
- **class_weight**
  Optional class weights for the cross-entropy loss. If omitted or `null`, class weighting is disabled.
- **label_smoothing**
  Optional label-smoothing factor. If omitted or `null`, label smoothing is disabled.
- **model**
  Model class to instantiate, for example `CreditModel`, `SatelliteFeaturesCentralized2layers`, `AggregationModel`, or `VertibenchReproduction`.
- **continue_training**
  Path to a checkpoint to resume from. If omitted or `null`, training starts from scratch.
- **party_id**
  Party index for isolated experiments such as `VertibenchIsolated`.
- **sentinel_weight**
  Weight applied to Sentinel outputs relative to SPOT in `SPOTandSentinelAggregationModel`.

## Dataset

- **dataset.data_root**
  Dataset root. Default: `${oc.env:DATA_ROOT}`, read from the `.env` file.
- **dataset.dataset_name**
  Dataset class to use, for example `CreditDataset`, `SentinelDataset`, `SPOTDataset`, or `VertibenchData`.
- **dataset.num_classes**
  Number of target classes.
- **dataset.label_type**
  Label mapping for satellite data: `sources`, `ipcc`, or `simplified_ipcc`.
- **dataset.features_instead**
  Folder containing pre-extracted features to use instead of raw Sentinel/VertiBench images.
- **dataset.spot_features_instead**
  Folder containing pre-extracted SPOT features. Defaults to `dataset.features_instead` when unset.
- **dataset.flips**
  Apply random horizontal and vertical flips. Do not use with pre-extracted features. If omitted or `null`, flips are disabled.
- **dataset.rgb**
  Keep only the RGB channels of raw images. Sentinel L2A images have 12 channels. Do not use with pre-extracted features.

## Runtime

- **runtime.device**
  `auto`, `cuda`, or `cpu`. Default: `auto` (CUDA when available, CPU otherwise).
- **runtime.gpu**
  CUDA device index used when running on GPU. Default: `0`.
- **runtime.seed**
  Random seed for reproducibility. Default: `0`.
- **runtime.save_model**
  Save the final model as `trained.pt`. Default: `False`.
- **runtime.num_workers**
  Number of PyTorch data-loader workers. Default: `4`. Use `0` on machines where multiprocessing is problematic.

## Early stopping

- **early_stopping.patience**
  Number of epochs without validation-loss improvement before stopping. Default: `10`.
- **early_stopping.delta**
  Minimum validation-loss change considered an improvement. Default: `0.0`.

Set `early_stopping: null` to disable early stopping.

## Scheduler

- **scheduler.type**
  `ReduceLROnPlateau` or `StepLR`. Default: `ReduceLROnPlateau`.
- **scheduler.factor**
  Learning-rate reduction factor. Default: `0.3`.
- **scheduler.patience**
  Patience for `ReduceLROnPlateau`. Default: `4`.
- **scheduler.step_size**
  Step size for `StepLR`.

Set `scheduler: null` to disable the scheduler.

## Hydra

Output folders and logging are configured in `simple_vfl/config/standard/standard_output_folder_naming.yaml` and `simple_vfl/config/hydra/`. By default Hydra writes to `outputs/` and changes the working directory to the run folder.
