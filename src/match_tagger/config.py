"""Loading of YAML stage configs into pydantic models."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path("configs")

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return raw or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """``overlay`` over ``base``: nested dicts are merged key by key, other values replaced."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(name: str, model: type[ModelT], config_dir: Path = CONFIG_DIR) -> ModelT:
    """Load ``<config_dir>/<name>.yaml`` and validate it against ``model``.

    An optional ``<name>.private.yaml`` next to it (gitignored) is merged over it: it holds
    entries that must not be published, such as per-clip metadata of licensed footage.
    A missing or empty file yields the model's defaults.
    """
    public = _read_yaml(config_dir / f"{name}.yaml")
    private = _read_yaml(config_dir / f"{name}.private.yaml")
    return model.model_validate(deep_merge(public, private))


class PathsConfig(BaseModel):
    """Project-wide directories, read from the ``paths`` section of default.yaml."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")


class DefaultConfig(BaseModel):
    """Contents of configs/default.yaml."""

    paths: PathsConfig = PathsConfig()
