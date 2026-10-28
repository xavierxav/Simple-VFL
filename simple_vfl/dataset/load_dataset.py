"""
Load and preprocess datasets based on the specified configuration.

This function loads different datasets and prepares them for training, validation, and testing
depending on the dataset name provided in the configuration arguments. It supports multiple
datasets (from Datasets.py file) and performs necessary preprocessing steps,
such as scaling features or mapping labels.

Key steps:
- Selects the appropriate dataset class based on the dataset name.
- Reads data from CSV files or other data sources.
- Splits the data into training, validation, and test sets, ensuring stratification if needed.
- Applies data preprocessing, such as feature scaling or label mapping.
- Returns the datasets as instances of PyTorch's Dataset or Subset, ready for use with DataLoader.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset

from ..utils.labels import LabelMappings
from . import datasets as Datasets

# Use the same logger as in the main file
logger = logging.getLogger(__name__)

# Feature order expected by the UCI Credit Card dataset (22 columns).
CREDIT_FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


SATELLITE_EXCLUDED_POIS = ["ASMSpotter-1-1-1", "Landcover-1062091", "UNHCR-HUNs002665"]


def load_satellite_metadata(data_root: str, label_type: str) -> pd.DataFrame:
    """Build the one-row-per-POI frame used by every satellite experiment.

    Keeping this in one place guarantees that the location-only baseline in
    ``scripts/location_baseline.py`` uses the same POIs and labels as the
    image models.
    """
    csv_file = os.path.join(data_root, "satellite_dataset", "metadata.csv")
    # The public WorldStrat metadata.csv (Zenodo) is comma-separated, while
    # some local exports use semicolons. Sniff the delimiter so both load.
    df = pd.read_csv(os.path.normpath(csv_file), sep=None, engine="python")
    df = df.rename(columns={df.columns[0]: "POI"})
    df = df.loc[~df["POI"].isin(SATELLITE_EXCLUDED_POIS)]

    columns = ["POI", "IPCC Class"] + [column for column in ("lon", "lat") if column in df.columns]
    metadata = df[columns].drop_duplicates().reset_index(drop=True)

    label_mappings = LabelMappings(label_type)

    def label_of(row):
        source = row["POI"].split("-")[0] if label_type == "sources" else row["IPCC Class"]
        return label_mappings.get_label(source)

    metadata["label"] = metadata.apply(label_of, axis=1)
    return metadata


def split_satellite_metadata(metadata: pd.DataFrame, seed: int):
    """Return the POI-grouped train/validation/test indices for a seed."""
    train_val_pois, test_pois = train_test_split(
        range(len(metadata)), test_size=0.15, random_state=42, stratify=metadata["label"]
    )
    train_pois, val_pois = train_test_split(
        train_val_pois,
        test_size=0.2,
        random_state=seed,
        stratify=metadata.iloc[train_val_pois]["label"],
    )
    return train_pois, val_pois, test_pois


def load_datasets(args):
    logger.info("load_datasets")

    # The dataset class is selected from `dataset.dataset_name` in the config.
    dataset_class = getattr(Datasets, args.dataset.dataset_name)

    if "credit" in args.dataset.dataset_name.lower():
        csv_file = os.path.join(args.dataset.data_root, "tabledata", "UCI_Credit_Card.csv")
        df = pd.read_csv(csv_file)
        logger.info("credit dataset loaded")

        X = df[CREDIT_FEATURE_COLUMNS].values
        y = df["default.payment.next.month"].values
        train_val, test = train_test_split(range(len(y)), test_size=0.2, random_state=42, stratify=y)
        train, val = train_test_split(train_val, test_size=0.25, random_state=args.runtime.seed, stratify=y[train_val])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_val = scaler.transform(X[val])
        X_test = scaler.transform(X[test])

        train_data = dataset_class(data=X_train, labels=y[train])
        val_data = dataset_class(data=X_val, labels=y[val])
        test_data = dataset_class(data=X_test, labels=y[test])

        return train_data, val_data, test_data

    elif "sentinel" in args.dataset.dataset_name.lower() or "spot" in args.dataset.dataset_name.lower():
        metadata = load_satellite_metadata(args.dataset.data_root, args.dataset.label_type)
        train_pois, val_pois, test_pois = split_satellite_metadata(metadata, args.runtime.seed)

        dataset = dataset_class(dataset_dict=args.dataset, metadata=metadata, k=args.k)

        return Subset(dataset, train_pois), Subset(dataset, val_pois), Subset(dataset, test_pois)

    elif "vertibench" in args.dataset.dataset_name.lower():
        test_dataset = dataset_class(args, split="test")
        train_val_dataset = dataset_class(args, split="train")

        features_instead = args.dataset.get("features_instead") or "Vertibench"
        data_path = os.path.join(args.dataset.data_root, "satellite_dataset", features_instead)
        train_val_labels = np.load(os.path.join(data_path, "label_train.npy"))
        train_indices, val_indices = train_test_split(
            range(len(train_val_labels)), test_size=0.15, random_state=args.runtime.seed, stratify=train_val_labels
        )

        return Subset(train_val_dataset, train_indices), Subset(train_val_dataset, val_indices), test_dataset
