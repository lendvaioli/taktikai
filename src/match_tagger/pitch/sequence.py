"""Temporal handling of per-frame homographies: smoothing, reuse and quality labels."""

from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np

from match_tagger.pitch.geometry import PitchDims
from match_tagger.pitch.homography import project

HQuality = Literal["ok", "reused", "missing"]
_EPS = 1e-9


def pitch_grid(d: PitchDims, n: int = 5) -> np.ndarray:
    """Return an ``n`` x ``n`` grid of pitch points covering the whole pitch."""
    xs, ys = np.linspace(0, d.length_m, n), np.linspace(0, d.width_m, n)
    return np.array([(x, y) for x in xs for y in ys])


def smooth_homographies(
    times: Sequence[float],
    matrices: Sequence[np.ndarray | None],
    window_sec: float,
    grid: np.ndarray,
) -> list[np.ndarray | None]:
    """Smooth valid homographies over a centred time window; frames without one stay None.

    Matrices are not averaged elementwise. Each valid neighbour within ``window_sec`` maps the
    pitch ``grid`` into the image; those image points are averaged and the homography is refit
    from them. Only real estimates contribute.
    """
    valid = [(t, m) for t, m in zip(times, matrices, strict=True) if m is not None]
    image_pts = {id(m): project(np.linalg.inv(m), grid) for _, m in valid}
    out: list[np.ndarray | None] = []
    for t, matrix in zip(times, matrices, strict=True):
        if matrix is None:
            out.append(None)
            continue
        near = [image_pts[id(m)] for s, m in valid if abs(s - t) <= window_sec + _EPS]
        smoothed, _ = cv2.findHomography(np.mean(near, axis=0), grid, 0)
        out.append(smoothed if smoothed is not None else matrix)
    return out


def assign_quality(
    times: Sequence[float], matrices: Sequence[np.ndarray | None], max_reuse_sec: float
) -> list[tuple[np.ndarray | None, HQuality]]:
    """Label each frame ok / reused / missing.

    A frame without its own homography reuses the last earlier valid one if that is at most
    ``max_reuse_sec`` old; otherwise it is missing. Nothing is extrapolated.
    """
    out: list[tuple[np.ndarray | None, HQuality]] = []
    last: tuple[float, np.ndarray] | None = None
    for t, matrix in zip(times, matrices, strict=True):
        if matrix is not None:
            last = (t, matrix)
            out.append((matrix, "ok"))
        elif last is not None and t - last[0] <= max_reuse_sec + _EPS:
            out.append((last[1], "reused"))
        else:
            out.append((None, "missing"))
    return out
