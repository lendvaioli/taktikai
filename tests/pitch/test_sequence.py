import numpy as np
import pytest

from match_tagger.pitch.geometry import PitchDims
from match_tagger.pitch.homography import project
from match_tagger.pitch.sequence import assign_quality, pitch_grid, smooth_homographies

GRID = pitch_grid(PitchDims())
# image -> pitch: scale pixels by 0.05 (20 px per metre)
H = np.diag([0.05, 0.05, 1.0])


def _shifted(dx_m):
    m = H.copy()
    m[0, 2] = dx_m
    return m


def test_quality_ok_reused_missing_at_5fps():
    times = [round(i * 0.2, 1) for i in range(21)]  # 0.0 .. 4.0 s
    matrices = [H if t in (0.0, 1.0) else None for t in times]
    labels = {t: q for t, (_, q) in zip(times, assign_quality(times, matrices, 2.0), strict=True)}
    assert labels[0.0] == "ok" and labels[1.0] == "ok"
    assert labels[0.4] == "reused" and labels[2.0] == "reused"
    assert labels[3.0] == "reused"  # exactly 2 s after the last ok frame
    assert labels[3.2] == "missing" and labels[4.0] == "missing"


def test_missing_before_first_valid_homography():
    out = assign_quality([0.0, 0.2, 0.4], [None, None, H], 2.0)
    assert [q for _, q in out] == ["missing", "missing", "ok"]
    assert out[0][0] is None


def test_reused_frames_carry_the_last_valid_matrix():
    out = assign_quality([0.0, 0.2], [H, None], 2.0)
    assert out[1][0] is H


def test_smoothing_keeps_identical_homographies():
    smoothed = smooth_homographies([0.0, 0.2, 0.4], [H, H, H], 0.4, GRID)
    point = np.array([[1000.0, 600.0]])
    for m in smoothed:
        assert project(m, point) == pytest.approx(project(H, point))


def test_smoothing_damps_a_single_outlier_frame():
    matrices = [_shifted(0.0), _shifted(3.0), _shifted(0.0)]  # middle frame off by 3 m
    smoothed = smooth_homographies([0.0, 0.2, 0.4], matrices, 0.2, GRID)
    dx = project(smoothed[1], np.array([[1000.0, 600.0]]))[0, 0] - 50.0
    assert dx == pytest.approx(1.0, abs=1e-6)  # mean of 0, 3, 0


def test_smoothing_leaves_frames_without_homography_empty():
    smoothed = smooth_homographies([0.0, 0.2], [H, None], 0.4, GRID)
    assert smoothed[1] is None


def test_zero_window_disables_smoothing():
    matrices = [_shifted(0.0), _shifted(3.0)]
    smoothed = smooth_homographies([0.0, 0.2], matrices, 0.0, GRID)
    assert project(smoothed[1], np.array([[0.0, 0.0]]))[0, 0] == pytest.approx(3.0)
