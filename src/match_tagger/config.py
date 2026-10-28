"""Loading of YAML stage configs into pydantic models."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path("configs")

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_config(name: str, model: type[ModelT], config_dir: Path = CONFIG_DIR) -> ModelT:
    """Load ``<config_dir>/<name>.yaml`` and validate it against ``model``.

    A missing or empty file yields the model's defaults.
    """
    path = config_dir / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return model.model_validate(raw or {})


class PathsConfig(BaseModel):
    """Project-wide directories, read from the ``paths`` section of default.yaml."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")


class DefaultConfig(BaseModel):
    """Contents of configs/default.yaml."""

    paths: PathsConfig = PathsConfig()
