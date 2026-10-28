import numpy as np
import pytest

from match_tagger.events.ball import (
    airborne_flags,
    expected_ball_px,
    interpolate_gaps,
    speed_spikes,
)
from match_tagger.events.config import EventsConfig
from match_tagger.events.possession import FramePlayers, frame_candidate
from match_tagger.events.stage import detect_events
from tests.events.synthetic import hold, make_clip

CFG = EventsConfig()
ALL = range(0, 60)


def test_nearest_player_within_radius():
    players = FramePlayers(
        np.array([1, 2]), ["home", "away"], np.array([[10.0, 10.0], [11.0, 10.0]])
    )
    cand = frame_candidate(np.array([10.8, 10.0]), players, 1.5)
    assert (cand.track_id, cand.team, round(cand.dist_m, 2)) == (2, "away", 0.2)
    assert frame_candidate(np.array([13.0, 10.0]), players, 1.5) is None  # 2 m from nearest
    assert frame_candidate(np.array([np.nan, np.nan]), players, 1.5) is None


def test_flicker_shorter_than_min_hold_is_merged_away():
    # A1 holds frames 0-20 except frame 10, where the ball is nearer to B (0.08 s < 0.24 s)
    ball = {**hold(range(0, 21), (10.5, 10)), 10: (11.4, 10)}
    players = {1: ("home", ALL, (10, 10)), 5: ("away", ALL, (11.6, 10))}
    det = detect_events(make_clip(21, players, ball), CFG)
    assert [(s.track_id, s.start, s.end) for s in det.segments] == [(1, 0, 20)]
    assert det.events == []


def test_expected_ball_width_from_homography():
    # image -> pitch scales pixels by 0.05 m (20 px per metre): 0.22 m ball = 4.4 px
    assert expected_ball_px(np.diag([0.05, 0.05, 1.0]), 30.0, 20.0, 0.22) == pytest.approx(4.4)


def test_airborne_by_size_ratio():
    t = np.array([0.0, 0.08, 0.16])
    xy = np.array([[10.0, 10.0], [10.5, 10.0], [11.0, 10.0]])
    flags = airborne_flags(t, xy, np.array([1.0, 2.0, np.nan]), 1.6, 35.0)
    assert flags.tolist() == [False, True, False]


def test_speed_spike_flags_only_the_outlier():
    # 10 m jump in 0.08 s and back: only the middle observation is too fast to both neighbours
    t = np.array([0.0, 0.08, 0.16, 0.24])
    xy = np.array([[10.0, 10.0], [10.2, 10.0], [20.2, 10.0], [10.6, 10.0]])
    assert speed_spikes(t, xy, 35.0).tolist() == [False, False, True, False]


def test_airborne_ball_is_not_possessed():
    # ball 0.5 m from A1 but twice the size expected on the ground -> in the air
    ratio = np.full(13, 2.0)
    det = detect_events(
        make_clip(13, {1: ("home", ALL, (10, 10))}, hold(range(13), (10.5, 10)), size_ratio=ratio),
        CFG,
    )
    assert det.airborne.all() and not det.possessed.any()


def test_ball_at_feet_with_normal_size_is_possessed():
    ratio = np.full(13, 1.0)
    det = detect_events(
        make_clip(13, {1: ("home", ALL, (10, 10))}, hold(range(13), (10.5, 10)), size_ratio=ratio),
        CFG,
    )
    assert not det.airborne.any() and det.possessed.all()


def test_interpolation_fills_short_gaps_only():
    times = np.round(np.arange(12) * 0.08, 2)
    xy = np.full((12, 2), np.nan)
    xy[0], xy[4] = (0.0, 0.0), (4.0, 2.0)  # gap 0.32 s -> filled
    xy[5], xy[11] = (5.0, 0.0), (11.0, 0.0)  # gap 0.48 s -> left empty
    out, filled = interpolate_gaps(times, xy, 0.4)
    assert out[2] == pytest.approx([2.0, 1.0])
    assert filled.tolist() == [False, True, True, True] + [False] * 8
    assert np.isnan(out[8, 0])


def test_no_interpolation_across_airborne_ball():
    # ground ball at frames 0 and 3, airborne (size ratio 2) at 1-2: the airborne rows are
    # removed and the 0.24 s gap is NOT filled, although it is shorter than 0.4 s
    ball = {0: (10.0, 10.0), 1: (12.0, 10.0), 2: (14.0, 10.0), 3: (16.0, 10.0)}
    ratio = np.array([1.0, 2.0, 2.0, 1.0])
    det = detect_events(make_clip(4, {}, ball, size_ratio=ratio), CFG)
    assert det.airborne.tolist() == [False, True, True, False]
    assert not det.interpolated.any()
    assert np.isnan(det.ball_xy[1:3, 0]).all()
