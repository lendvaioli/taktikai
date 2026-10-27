"""Per-frame homography from image pixels to pitch metres (OpenCV, RANSAC)."""

from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, Field


class HomographyConfig(BaseModel):
    """Keypoint selection and homography acceptance thresholds."""

    kp_conf: float = Field(0.5, ge=0, le=1)
    min_points: int = Field(4, ge=4)
    reproj_threshold_m: float = Field(1.0, gt=0, description="RANSAC inlier threshold.")
    max_reproj_error_m: float = Field(1.0, gt=0, description="Max mean inlier error.")


@dataclass(frozen=True)
class HomographyFit:
    """An accepted image -> pitch homography and its fit statistics."""

    matrix: np.ndarray
    n_points: int
    n_inliers: int
    mean_error_m: float


def select_keypoints(
    xy: np.ndarray, conf: np.ndarray, image_wh: tuple[int, int], cfg: HomographyConfig
) -> np.ndarray:
    """Return a boolean mask of keypoints that are confident and inside the image."""
    width, height = image_wh
    inside = (xy[:, 0] > 1) & (xy[:, 1] > 1) & (xy[:, 0] < width - 1) & (xy[:, 1] < height - 1)
    return (conf >= cfg.kp_conf) & inside


def project(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    if len(xy) == 0:
        return np.empty((0, 2))
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def fit_homography(
    image_xy: np.ndarray, pitch_xy: np.ndarray, cfg: HomographyConfig
) -> HomographyFit | None:
    """Fit image -> pitch with RANSAC; None unless enough inliers fit well enough.

    No homography is returned when fewer than ``min_points`` correspondences or inliers exist
    or when the mean inlier error exceeds ``max_reproj_error_m``.
    """
    if len(image_xy) < cfg.min_points:
        return None
    src = np.asarray(image_xy, dtype=np.float64)
    dst = np.asarray(pitch_xy, dtype=np.float64)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.reproj_threshold_m)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < cfg.min_points:
        return None
    errors = np.linalg.norm(project(matrix, src[inliers]) - dst[inliers], axis=1)
    if errors.mean() > cfg.max_reproj_error_m:
        return None
    return HomographyFit(matrix, len(src), int(inliers.sum()), float(errors.mean()))
