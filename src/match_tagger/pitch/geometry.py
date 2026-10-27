"""Pitch model in metres and the 32 landmark keypoints of the roboflow/sports pitch model.

Coordinates: origin at the bottom-left corner as seen from the main camera (near touchline,
left goal line); x along the touchlines up to ``length_m``, y towards the far touchline up to
``width_m``. Markings use IFAB sizes; only length and width vary between pitches.
"""

import numpy as np
from pydantic import BaseModel, Field


class PitchDims(BaseModel):
    """Pitch size and markings in metres."""

    length_m: float = Field(105.0, gt=0)
    width_m: float = Field(68.0, gt=0)
    penalty_area_length_m: float = 16.5
    penalty_area_width_m: float = 40.32
    goal_area_length_m: float = 5.5
    goal_area_width_m: float = 18.32
    penalty_spot_m: float = 11.0
    centre_circle_radius_m: float = 9.15


def keypoint_vertices(d: PitchDims) -> np.ndarray:
    """Return the (32, 2) pitch positions of the model's keypoints, in keypoint order.

    The order follows ``SoccerPitchConfiguration.vertices`` in roboflow/sports, whose y axis
    starts at the far touchline; it is flipped here so that y starts at the near touchline.
    """
    length, width = d.length_m, d.width_m
    pb, gb = d.penalty_area_width_m / 2, d.goal_area_width_m / 2
    mid, r = width / 2, d.centre_circle_radius_m
    far_to_near = [0.0, mid - pb, mid - gb, mid + gb, mid + pb, width]  # y from far touchline
    pa, ga, ps = d.penalty_area_length_m, d.goal_area_length_m, d.penalty_spot_m
    rf = [
        *[(0.0, y) for y in far_to_near],  # 1-6 left goal line
        (ga, mid - gb), (ga, mid + gb), (ps, mid),  # 7-9
        *[(pa, y) for y in far_to_near[1:5]],  # 10-13
        *[(length / 2, y) for y in (0.0, mid - r, mid + r, width)],  # 14-17
        *[(length - pa, y) for y in far_to_near[1:5]],  # 18-21
        (length - ps, mid), (length - ga, mid - gb), (length - ga, mid + gb),  # 22-24
        *[(length, y) for y in far_to_near],  # 25-30 right goal line
        (length / 2 - r, mid), (length / 2 + r, mid),  # 31-32
    ]  # fmt: skip
    xy = np.array(rf, dtype=np.float64)
    xy[:, 1] = width - xy[:, 1]
    return xy


# Line segments between keypoints (1-based, as in roboflow/sports), for drawing.
EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13), (14, 15),
    (15, 16), (16, 17), (18, 19), (19, 20), (20, 21), (23, 24), (25, 26), (26, 27), (27, 28),
    (28, 29), (29, 30), (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26),
    (23, 27), (24, 28), (21, 29), (17, 30),
)  # fmt: skip


def outside_distance(xy: np.ndarray, d: PitchDims) -> np.ndarray:
    """Return how far each point lies outside the pitch rectangle (0 inside), in metres."""
    dx = np.maximum(np.maximum(-xy[:, 0], xy[:, 0] - d.length_m), 0.0)
    dy = np.maximum(np.maximum(-xy[:, 1], xy[:, 1] - d.width_m), 0.0)
    return np.hypot(dx, dy)
