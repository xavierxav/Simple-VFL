import numpy as np
import pandas as pd
import torch

from addict import Dict

from simple_vfl.main_task_vfl import MainTaskVFL
from simple_vfl.dataset.datasets import SatelliteDataset
from simple_vfl.models.complete_model import AggregationModel, SPOTandSentinelAggregationModel


FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def _write_credit_csv(root, n=100, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(0, 100, n) for col in FEATURE_COLUMNS}
    data["LIMIT_BAL"] = rng.integers(10000, 500000, n)
    data["default.payment.next.month"] = np.arange(n) % 2
    target = root / "tabledata"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(target / "UCI_Credit_Card.csv", index=False)


def _credit_config(data_root):
    return Dict({
        "k": 2,
        "epochs": 1,
        "lr": 1e-3,
        "batch_size": 16,
        "hidden_dim": 8,
        "model": "CreditModel",
        "runtime": {"device": "cpu", "gpu": 0, "seed": 0, "save_model": False, "num_workers": 0},
        "dataset": {"dataset_name": "CreditDataset", "num_classes": 2, "data_root": str(data_root)},
        "early_stopping": {"patience": 2, "delta": 0.0},
    })


def test_credit_smoke(tmp_path, monkeypatch):
    """Run one full epoch on a synthetic Credit dataset (CPU)."""
    monkeypatch.chdir(tmp_path)
    _write_credit_csv(tmp_path)
    vfl = MainTaskVFL(_credit_config(tmp_path))
    vfl.train()


def test_satellite_feature_instead_is_optional():
    """Missing features_instead must not select the feature branch (addict returns an empty Dict)."""
    dataset = SatelliteDataset(Dict({"data_root": "/tmp", "dataset_name": "SentinelDataset"}), metadata=[])
    assert dataset.feature_instead is None
    assert dataset.spot_features_instead is None


def test_satellite_spot_features_instead_fallback():
    """spot_features_instead falls back to features_instead when not set."""
    dataset = SatelliteDataset(
        Dict({"data_root": "/tmp", "dataset_name": "SentinelDataset", "features_instead": "sentinel_feats"}),
        metadata=[],
    )
    assert dataset.spot_features_instead == "sentinel_feats"


def test_aggregation_averages_logits():
    """The aggregation models average party logits instead of summing them."""
    config = Dict({"k": 3, "lr": 1e-3, "hidden_dim": 8, "dataset": {"num_classes": 2}})
    model = AggregationModel(config)
    features = [torch.randn(4, 1536) for _ in range(3)]
    bottom_outputs, final_output = model(features)
    assert torch.allclose(final_output, torch.stack(bottom_outputs).mean(dim=0))


def test_spot_and_sentinel_aggregation_shapes():
    """The SPOT + Sentinel model handles an arbitrary number of Sentinel views."""
    config = Dict(
        {
            "k": 17,
            "lr": 1e-3,
            "hidden_dim": 8,
            "sentinel_weight": 1.0,
            "dataset": {"num_classes": 2},
        }
    )
    model = SPOTandSentinelAggregationModel(config)
    features = [torch.randn(4, 1536) for _ in range(17)]
    bottom_outputs, final_output = model(features)
    assert len(bottom_outputs) == 17
    assert final_output.shape == (4, 2)
