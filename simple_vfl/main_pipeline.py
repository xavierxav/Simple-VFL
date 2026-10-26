"""Entry point for the Simple-VFL training pipeline."""

import logging
from pathlib import Path

import hydra
import torch
from addict import Dict
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from .main_task_vfl import MainTaskVFL
from .utils.basic_functions import custom_name_for_hydra_outputs_folder, set_seed

logger = logging.getLogger(__name__)

# Load the .env file from the current directory or the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for env_file in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
    if env_file.is_file():
        load_dotenv(env_file)

OmegaConf.register_new_resolver(
    "custom_name_for_hydra_outputs_folder",
    custom_name_for_hydra_outputs_folder,
    replace=True,
)


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise to CPU."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@hydra.main(version_base="1.3", config_path="config", config_name="credit")
def main(cfg: DictConfig) -> None:
    config = Dict(OmegaConf.to_container(cfg, resolve=True))

    set_seed(config.runtime.seed)

    config.runtime.device = resolve_device(config.runtime.device)
    if config.runtime.device == "cuda":
        torch.cuda.set_device(config.runtime.gpu)
        logger.info("Running on CUDA device %s", torch.cuda.current_device())
    else:
        logger.info("Running on CPU")

    vfl = MainTaskVFL(config)
    vfl.train()


if __name__ == "__main__":
    main()
