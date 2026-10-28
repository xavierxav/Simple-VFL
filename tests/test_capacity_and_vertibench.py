import numpy as np
import torch
from addict import Dict

from simple_vfl.dataset.datasets import VertibenchData
from simple_vfl.dataset.load_dataset import load_datasets
from simple_vfl.models.complete_model import (
    SatelliteFeaturesCentralized2layers,
    SentinelSplitNN,
    VertibenchReproduction,
)


def test_vertibench_rgb_slice_keeps_three_channels(tmp_path):
    root = tmp_path
    (root / "satellite_dataset" / "Vertibench").mkdir(parents=True)
    data_dir = root / "satellite_dataset" / "Vertibench"
    np.save(data_dir / "label_train.npy", np.array([0], dtype=np.uint8))
    image = np.zeros((13, 4, 4), dtype=np.uint8)
    image[1] = 11
    image[2] = 22
    image[3] = 33
    np.save(data_dir / "satellite_party0_train_0.npy", image)

    args = Dict(
        {
            "dataset": {"data_root": str(root), "dataset_name": "VertibenchData", "rgb": True},
            "k": 1,
        }
    )
    dataset = VertibenchData(args, split="train")
    views, _ = dataset[0]

    assert views[0].shape == (3, 4, 4)
    assert views[0][0, 0, 0].item() == image[3, 0, 0]
    assert views[0][1, 0, 0].item() == image[2, 0, 0]
    assert views[0][2, 0, 0].item() == image[1, 0, 0]


def test_vertibench_model_accepts_three_channels():
    args = Dict(
        {
            "dataset": {"num_classes": 4, "data_root": "/tmp"},
            "k": 1,
            "hidden_dim": 1000,
            "lr": 1e-3,
            "channels": 3,
        }
    )
    model = VertibenchReproduction(args)
    views = [torch.zeros(1, 3, 158, 158)]
    _, output = model(views)
    assert output.shape == (1, 4)


def test_one_view_capacity_matches_sixteen_view_head():
    base = Dict(
        {
            "dataset": {"num_classes": 4},
            "k": 16,
            "hidden_dim": 1000,
            "lr": 1e-3,
        }
    )
    wide = Dict(
        {
            "dataset": {"num_classes": 4},
            "k": 1,
            "hidden_dim": 15932,
            "lr": 1e-3,
        }
    )
    sixteen = sum(p.numel() for p in SatelliteFeaturesCentralized2layers(base).parameters())
    one = sum(p.numel() for p in SatelliteFeaturesCentralized2layers(wide).parameters())
    assert abs(sixteen - one) / sixteen < 0.001


def test_vertibench_train_on_all_keeps_every_training_sample(tmp_path):
    data_dir = tmp_path / "satellite_dataset" / "Vertibench"
    data_dir.mkdir(parents=True)
    for split, count in (("train", 10), ("test", 2)):
        np.save(data_dir / f"label_{split}.npy", np.zeros(count, dtype=np.uint8))
        for index in range(count):
            np.save(data_dir / f"satellite_party0_{split}_{index}.npy", np.zeros((13, 2, 2), dtype=np.uint8))

    args = Dict(
        {
            "dataset": {
                "data_root": str(tmp_path),
                "dataset_name": "VertibenchData",
                "num_classes": 4,
                "label_type": "sources",
                "train_on_all": True,
            },
            "runtime": {"seed": 0},
            "k": 1,
        }
    )
    train, val, test = load_datasets(args)

    assert len(train) == 10
    assert len(val) > 0
    assert len(test) == 2


def test_sentinel_splitnn_matches_the_concatenation_capacity():
    args = Dict(
        {
            "dataset": {"num_classes": 4},
            "k": 16,
            "hidden_dim": 1000,
            "lr": 1e-3,
        }
    )
    splitnn = SentinelSplitNN(args)
    concat = SatelliteFeaturesCentralized2layers(args)
    splitnn_params = sum(p.numel() for p in splitnn.parameters())
    concat_params = sum(p.numel() for p in concat.parameters())
    assert abs(splitnn_params - concat_params) / concat_params < 0.01

    _, output = splitnn([torch.zeros(2, 1536) for _ in range(16)])
    assert output.shape == (2, 4)
