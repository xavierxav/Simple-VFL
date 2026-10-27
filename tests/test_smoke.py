import numpy as np
import pandas as pd
import torch
from addict import Dict

from simple_vfl.dataset.datasets import HorizontalSPOTAndOneSentinelDataset, SatelliteDataset
from simple_vfl.main_task_vfl import MainTaskVFL
from simple_vfl.models.complete_model import (
    AggregationModel,
    SpecialModel,
    SPOTandSentinelAggregationModel,
    SPOTandSentinelMeanEmbeddingModel,
    SPOTandSentinelSharedConcatModel,
)

FEATURE_COLUMNS = [
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


def _write_credit_csv(root, n=100, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(0, 100, n) for col in FEATURE_COLUMNS}
    data["LIMIT_BAL"] = rng.integers(10000, 500000, n)
    data["default.payment.next.month"] = np.arange(n) % 2
    target = root / "tabledata"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(target / "UCI_Credit_Card.csv", index=False)


def _credit_config(data_root):
    return Dict(
        {
            "k": 2,
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 16,
            "hidden_dim": 8,
            "model": "CreditModel",
            "runtime": {"device": "cpu", "gpu": 0, "seed": 0, "save_model": False, "num_workers": 0},
            "dataset": {"dataset_name": "CreditDataset", "num_classes": 2, "data_root": str(data_root)},
            "early_stopping": {"patience": 2, "delta": 0.0},
        }
    )


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


def test_horizontal_spot_and_one_sentinel_returns_two_views():
    """The SPOT + 1 Sentinel proxy returns exactly the two sensor views."""
    dataset = HorizontalSPOTAndOneSentinelDataset(
        Dict({"data_root": "/tmp", "dataset_name": "HorizontalSPOTAndOneSentinelDataset"}),
        metadata=pd.DataFrame({"POI": ["A"], "label": [0]}),
        k=2,
    )
    dataset.get_spot_image = lambda poi: torch.zeros(1536)
    dataset.get_sentinel_image = lambda poi, n: torch.ones(1536)
    images, label = dataset[0]
    assert len(images) == 2
    assert images[0].shape == (1536,)
    assert label == 0


def test_special_model_handles_seventeen_views():
    """The shared-encoder model concatenates one hidden vector per view."""
    config = Dict({"k": 17, "lr": 1e-3, "hidden_dim": 8, "dataset": {"num_classes": 4}})
    model = SpecialModel(config)
    features = [torch.randn(4, 1536) for _ in range(17)]
    bottom_outputs, final_output = model(features)
    assert len(bottom_outputs) == 17
    assert final_output.shape == (4, 4)


def test_spot_and_sentinel_fusion_variants():
    """Mean-embedding and shared-concat fusion both return four logits per sample."""
    config = Dict({"k": 17, "lr": 1e-3, "hidden_dim": 8, "dataset": {"num_classes": 4}})
    features = [torch.randn(4, 1536) for _ in range(17)]
    for model_class in (SPOTandSentinelMeanEmbeddingModel, SPOTandSentinelSharedConcatModel):
        _, final_output = model_class(config)(features)
        assert final_output.shape == (4, 4)
