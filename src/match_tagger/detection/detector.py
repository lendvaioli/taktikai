"""Detector interface and backends, selected by ``backend`` in configs/detection.yaml."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from match_tagger.detection.config import DetectionConfig
from match_tagger.detection.schema import CLASSES
from match_tagger.detection.weights import ensure_weights

if TYPE_CHECKING:
    import supervision as sv


class Detector(Protocol):
    """Detects objects in one BGR frame; ``class_id`` indexes ``schema.CLASSES``."""

    def detect(self, image: np.ndarray) -> "sv.Detections":
        """Return detections in source-resolution pixel coordinates."""
        ...


def resolve_device(device: str) -> str:
    """Map ``auto`` to ``cuda`` when a GPU is available, else ``cpu``."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def class_lookup(model_names: dict[int, str], class_map: dict[str, str]) -> np.ndarray:
    """Return an array mapping model class index -> ``CLASSES`` index (-1 = ignored).

    Raises if a pipeline class has no model class mapped to it.
    """
    lookup = np.full(max(model_names) + 1, -1, dtype=int)
    for idx, name in model_names.items():
        if name in class_map:
            lookup[idx] = CLASSES.index(class_map[name])
    missing = set(range(len(CLASSES))) - set(lookup.tolist())
    if missing:
        names = sorted(CLASSES[i] for i in missing)
        raise ValueError(f"Model classes {model_names} do not provide {names}")
    return lookup


class UltralyticsDetector:
    """YOLO model loaded with ultralytics (AGPL-3.0)."""

    def __init__(self, cfg: DetectionConfig, models_dir: Path, device: str) -> None:
        """Load the weights, downloading them if needed."""
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = device
        self.model = YOLO(str(ensure_weights(cfg.weights, models_dir)))
        self.lookup = class_lookup(self.model.names, cfg.class_map)

    def detect(self, image: np.ndarray) -> "sv.Detections":
        """Run the model and remap classes to ``schema.CLASSES`` indices."""
        import supervision as sv

        result = self.model.predict(
            image, imgsz=self.cfg.imgsz, conf=self.cfg.conf, device=self.device, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections.class_id = self.lookup[detections.class_id]
        return detections[detections.class_id >= 0]


BACKENDS: dict[str, Callable[[DetectionConfig, Path, str], Detector]] = {
    "ultralytics_yolo": UltralyticsDetector,
}


def build_detector(cfg: DetectionConfig, models_dir: Path, device: str) -> Detector:
    """Instantiate the configured backend."""
    if cfg.ball_model.enabled:
        raise NotImplementedError("Separate ball model with sliced inference is not implemented")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"Unknown detector backend {cfg.backend!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[cfg.backend](cfg, models_dir, device)
