"""Projection of detection rows onto the pitch with the per-frame homography."""

from collections.abc import Sequence

import numpy as np

from match_tagger.detection.schema import DetectionRecord
from match_tagger.pitch.geometry import PitchDims, outside_distance
from match_tagger.pitch.homography import project
from match_tagger.pitch.schema import PositionRecord
from match_tagger.pitch.sequence import HQuality


def anchor(record: DetectionRecord) -> tuple[float, float]:
    """Image point to project: ball centre, otherwise the bottom-centre (foot point)."""
    cx = (record.x1 + record.x2) / 2
    if record.cls == "ball":
        return cx, (record.y1 + record.y2) / 2
    return cx, record.y2


def project_frame(
    records: Sequence[DetectionRecord],
    matrix: np.ndarray | None,
    quality: HQuality,
    dims: PitchDims,
    out_margin_m: float,
) -> list[PositionRecord]:
    """Project all records of one frame; positions are never clipped, only flagged."""
    base = [r.model_dump(include={"frame", "t_sec", "track_id", "cls", "team"}) for r in records]
    if matrix is None:
        return [PositionRecord(**b, h_quality="missing") for b in base]
    xy = project(matrix, np.array([anchor(r) for r in records]).reshape(-1, 2))
    flagged = outside_distance(xy, dims) > out_margin_m
    return [
        PositionRecord(
            **b, x_m=float(p[0]), y_m=float(p[1]), h_quality=quality, out_of_pitch=bool(f)
        )
        for b, p, f in zip(base, xy, flagged, strict=True)
    ]
