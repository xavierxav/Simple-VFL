"""Optional configuration keys must accept ``null`` and stay device-safe."""

import torch
import torch.nn as nn

from addict import Dict

from simple_vfl.main_task_vfl import MainTaskVFL
from simple_vfl.models.complete_model import CompleteModel


class _TaskStub:
    """Minimal stand-in exposing only what ``_initialize_criterion`` needs."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device


def _model_args(**overrides) -> Dict:
    args = Dict({"lr": 1e-3, "weight_decay": 0.0, "dataset": {"num_classes": 2}})
    args.update(overrides)
    return args


def test_null_scheduler_disables_scheduler() -> None:
    model = CompleteModel([nn.Linear(2, 2)], _model_args(scheduler=None))
    assert not hasattr(model, "scheduler")


def test_null_continue_training_does_not_load_checkpoint() -> None:
    model = CompleteModel([nn.Linear(2, 2)], _model_args(continue_training=None))
    assert isinstance(model, CompleteModel)


def test_null_class_weight_disables_weighting() -> None:
    args = _model_args(class_weight=None, label_smoothing=None, runtime={"device": "cpu"})
    criterion = MainTaskVFL._initialize_criterion(_TaskStub(), args)
    assert criterion.weight is None


def test_class_weight_tensor_follows_requested_device() -> None:
    args = _model_args(class_weight=[1, 3], runtime={"device": "meta"})
    criterion = MainTaskVFL._initialize_criterion(_TaskStub(device="meta"), args)
    assert criterion.weight.device.type == "meta"


def test_class_weight_is_normalised() -> None:
    args = _model_args(class_weight=[1, 3], runtime={"device": "cpu"})
    criterion = MainTaskVFL._initialize_criterion(_TaskStub(), args)
    assert torch.allclose(criterion.weight.sum(), torch.tensor(1.0))
