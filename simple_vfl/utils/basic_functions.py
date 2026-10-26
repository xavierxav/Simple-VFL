import os
import random
import re

import numpy as np
import torch
from omegaconf import ListConfig


def set_seed(seed=0):
    """
    Set the seed for reproducibility across various libraries.
    
    Parameters:
    seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    """Seed a DataLoader worker for reproducibility (``worker_init_fn``)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_name_for_hydra_outputs_folder(
    task_overrides: ListConfig,  # list[str]: overrides passed at command line
    exclude_patterns: ListConfig,  # list[str]: regex patterns to exclude
) -> str:
    """
    Code taken from : https://github.com/facebookresearch/hydra/issues/1873

    Return a string: concatenation of overrides that are not matched by any of the `exclude_patterns`."""

    rets: list[str] = []
    for override in task_overrides:
        should_exclude = any(re.search(exc_pat, override) for exc_pat in exclude_patterns)
        if not should_exclude:
            rets.append(override)

    return "_".join(rets)