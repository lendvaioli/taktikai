import cv2
import numpy as np
import pytest

from match_tagger.detection.schema import DetectionRecord
from match_tagger.pitch.geometry import PitchDims, keypoint_vertices
from match_tagger.pitch.homography import (
    HomographyConfig,
    fit_homography,
    project,
    select_keypoints,
)
from match_tagger.pitch.project import project_frame

CFG = HomographyConfig()
# A synthetic broadcast-like view: pitch corners -> image quadrilateral (far side narrower).
PITCH_CORNERS = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
IMAGE_CORNERS = np.float32([[100, 1000], [1800, 1000], [1400, 300], [500, 300]])
PITCH_TO_IMAGE = cv2.getPerspectiveTransform(PITCH_CORNERS, IMAGE_CORNERS).astype(np.float64)
VERTICES = keypoint_vertices(PitchDims())
IMAGE_KP = project(PITCH_TO_IMAGE, VERTICES)
CHECK_POINTS = np.array([[52.5, 34.0], [30.0, 20.0], [90.0, 60.0], [0.0, 0.0], [105.0, 68.0]])


def _roundtrip(fit):
    return project(fit.matrix, project(PITCH_TO_IMAGE, CHECK_POINTS))


def test_known_correspondences_map_within_10cm():
    idx = [0, 5, 8, 13, 16, 24, 29, 31]  # 8 keypoints spread over the pitch
    fit = fit_homography(IMAGE_KP[idx], VERTICES[idx], CFG)
    assert fit is not None and fit.n_inliers == 8
    assert np.abs(_roundtrip(fit) - CHECK_POINTS).max() < 0.1


def test_ransac_ignores_two_corrupted_keypoints():
    idx = [0, 5, 8, 9, 13, 16, 24, 29, 30, 31]
    image = IMAGE_KP[idx].copy()
    image[[2, 7]] += 50.0  # two keypoints off by 50 px
    fit = fit_homography(image, VERTICES[idx], CFG)
    assert fit is not None and fit.n_inliers == 8
    assert np.abs(_roundtrip(fit) - CHECK_POINTS).max() < 0.1


def test_fewer_than_four_points_gives_no_homography():
    assert fit_homography(IMAGE_KP[:3], VERTICES[:3], CFG) is None


def test_select_keypoints_needs_confidence_and_visibility():
    xy = np.array([[10.0, 10.0], [10.0, 10.0], [0.0, 0.0], [1919.5, 10.0]])
    conf = np.array([0.9, 0.4, 0.9, 0.9])
    assert select_keypoints(xy, conf, (1920, 1080), CFG).tolist() == [True, False, False, False]


def _rec(cls, x, y, team=None, track_id=1):
    # box whose bottom-centre is (x, y) and whose centre is (x, y - 1)
    return DetectionRecord(
        frame=0, t_sec=0.0, track_id=None if cls == "ball" else track_id, cls=cls, team=team,
        x1=x - 0.5, y1=y - 2.0, x2=x + 0.5, y2=y, conf=0.9,
    )  # fmt: skip


def test_out_of_pitch_points_are_flagged_not_clipped():
    rows = [_rec("player", -4.9, 34.0, 0), _rec("player", -5.1, 34.0, 1), _rec("player", 110.2, 34)]
    out = project_frame(rows, np.eye(3), "ok", PitchDims(), out_margin_m=5.0)
    assert [r.out_of_pitch for r in out] == [False, True, True]
    assert [r.x_m for r in out] == pytest.approx([-4.9, -5.1, 110.2])


def test_ball_is_projected_at_box_centre_players_at_foot_point():
    out = project_frame(
        [_rec("ball", 50.0, 30.0), _rec("player", 50.0, 30.0)], np.eye(3), "reused", PitchDims(), 5
    )
    assert [(r.x_m, r.y_m) for r in out] == pytest.approx([(50.0, 29.0), (50.0, 30.0)])
    assert {r.h_quality for r in out} == {"reused"}


def test_missing_homography_gives_no_position():
    out = project_frame([_rec("player", 10.0, 10.0)], None, "missing", PitchDims(), 5.0)
    assert (out[0].x_m, out[0].y_m, out[0].h_quality, out[0].out_of_pitch) == (
        None, None, "missing", False,
    )  # fmt: skip
