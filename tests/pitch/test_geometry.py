import numpy as np
import pytest

from match_tagger.pitch.config import PitchConfig, PitchSize
from match_tagger.pitch.geometry import PitchDims, keypoint_vertices, outside_distance


def _kp(vertices, number):
    return tuple(vertices[number - 1].round(2))


def test_keypoints_in_bottom_left_origin_frame():
    v = keypoint_vertices(PitchDims())
    assert v.shape == (32, 2)
    assert _kp(v, 1) == (0.0, 68.0)  # far-left corner
    assert _kp(v, 6) == (0.0, 0.0)  # near-left corner = origin
    assert _kp(v, 9) == (11.0, 34.0)  # left penalty spot
    assert _kp(v, 10) == (16.5, 54.16)  # 34 + 40.32 / 2
    assert _kp(v, 14) == (52.5, 68.0)  # halfway line, far touchline
    assert _kp(v, 17) == (52.5, 0.0)  # halfway line, near touchline
    assert _kp(v, 30) == (105.0, 0.0)  # near-right corner
    assert _kp(v, 32) == (61.65, 34.0)  # centre circle, right


def test_custom_pitch_size_moves_dependent_keypoints_only():
    v = keypoint_vertices(PitchDims(length_m=100, width_m=64))
    assert _kp(v, 25) == (100.0, 64.0)
    assert _kp(v, 18) == (83.5, 52.16)  # 100 - 16.5, 32 + 20.16
    assert _kp(v, 22) == (89.0, 32.0)  # 100 - 11


def test_match_size_override():
    cfg = PitchConfig(match_sizes={"m1": PitchSize(length_m=100, width_m=64)})
    assert (cfg.dims_for("m1").length_m, cfg.dims_for("m1").width_m) == (100, 64)
    assert cfg.dims_for("m1").penalty_area_length_m == 16.5
    assert (cfg.dims_for("other").length_m, cfg.dims_for("other").width_m) == (105, 68)


def test_outside_distance():
    pts = np.array([[50.0, 30.0], [-4.9, 34.0], [110.2, 34.0], [-3.0, -4.0]])
    assert outside_distance(pts, PitchDims()) == pytest.approx([0.0, 4.9, 5.2, 5.0])
