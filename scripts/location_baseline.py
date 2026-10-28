#!/usr/bin/env python3
"""Location-only baseline on the WorldStrat labels.

Uses the exact metadata frame and POI-grouped split from
``simple_vfl/dataset/load_dataset.py``, so the baseline is trained on the same
2,668 POIs and evaluated on the same 589 test POIs as the image models.

If the coordinates alone predict the label, an image benchmark on that label is
measuring geography as much as remote sensing.
"""

import argparse
import os
import statistics

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simple_vfl.dataset.load_dataset import load_satellite_metadata, split_satellite_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=os.environ.get("DATA_ROOT"), help="Folder containing satellite_dataset/."
    )
    parser.add_argument("--label", choices=["sources", "ipcc", "simplified_ipcc"], default="sources")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds for the train/validation split.")
    args = parser.parse_args()

    if not args.data_root:
        raise SystemExit("Set DATA_ROOT or pass --data-root.")

    metadata = load_satellite_metadata(args.data_root, args.label)
    coordinates = metadata[["lon", "lat"]].to_numpy()
    target = metadata["label"].to_numpy()
    seeds = [int(seed) for seed in args.seeds.split(",")]

    print(f"label={args.label}  classes={len(np.unique(target))}  POIs={len(metadata)}")
    print("class counts:", np.bincount(target).tolist())

    models = {
        "random forest": lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }

    for name, factory in models.items():
        accuracies = []
        aucs = []
        for seed in seeds:
            train, _, test = split_satellite_metadata(metadata, seed)
            model = factory()
            model.fit(coordinates[train], target[train])
            probabilities = model.predict_proba(coordinates[test])
            accuracies.append(accuracy_score(target[test], probabilities.argmax(axis=1)))
            aucs.append(roc_auc_score(target[test], probabilities, multi_class="ovr"))
        print(
            f"{name:20s} accuracy {statistics.mean(accuracies):.4f} +/- {statistics.pstdev(accuracies):.4f}  "
            f"auc {statistics.mean(aucs):.4f} +/- {statistics.pstdev(aucs):.4f}  "
            f"(n={len(seeds)} seeds, test={len(test)})"
        )


if __name__ == "__main__":
    main()
