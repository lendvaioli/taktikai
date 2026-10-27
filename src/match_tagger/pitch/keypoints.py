"""Pitch keypoint detector interface and backends, selected by ``backend`` in pitch.yaml."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from match_tagger.detection.weights import ensure_weights
from match_tagger.pitch.config import PitchConfig

N_KEYPOINTS = 32


class KeypointDetector(Protocol):
    """Detects the pitch landmarks in one BGR frame."""

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return keypoint pixels (32, 2) and confidences (32,), ordered as geometry.py."""
        ...


class UltralyticsPoseDetector:
    """YOLOv8-pose pitch model loaded with ultralytics (AGPL-3.0)."""

    def __init__(self, cfg: PitchConfig, models_dir: Path, device: str) -> None:
        """Load the weights, downloading them if needed, and check the keypoint count."""
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = device
        self.model = YOLO(str(ensure_weights(cfg.weights, models_dir)))
        kpt_shape = self.model.model.yaml.get("kpt_shape")
        if not kpt_shape or kpt_shape[0] != N_KEYPOINTS:
            raise ValueError(f"Expected {N_KEYPOINTS} keypoints, model has {kpt_shape}")

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the model; with no pitch instance all confidences are 0."""
        results = self.model.predict(image, imgsz=self.cfg.imgsz, device=self.device, verbose=False)
        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return np.zeros((N_KEYPOINTS, 2)), np.zeros(N_KEYPOINTS)
        best = int(result.boxes.conf.argmax())
        xy = result.keypoints.xy[best].cpu().numpy().astype(np.float64)
        conf = result.keypoints.conf[best].cpu().numpy().astype(np.float64)
        return xy, conf


BACKENDS: dict[str, Callable[[PitchConfig, Path, str], KeypointDetector]] = {
    "ultralytics_pose": UltralyticsPoseDetector,
}


def build_keypoint_detector(cfg: PitchConfig, models_dir: Path, device: str) -> KeypointDetector:
    """Instantiate the configured backend."""
    if cfg.backend not in BACKENDS:
        raise ValueError(f"Unknown keypoint backend {cfg.backend!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[cfg.backend](cfg, models_dir, device)
