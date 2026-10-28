import numpy as np
import pytest

from match_tagger.events.ball import ground_ratio_baseline
from match_tagger.events.config import EventsConfig
from match_tagger.events.stage import airborne_mask
from tests.events.synthetic import make_clip

ALL = range(0, 60)


def _clip():
    # frames 0-9: ball 0.5 m from the player, ratio 1.4 (ground balls, detector boxes wider)
    # frame 10: far ball, ratio 2.0;  frame 11: far ball, ratio 2.4
    ball = {f: (10.5, 10.0) for f in range(10)}
    ball[10], ball[11] = (30.0, 10.0), (30.2, 10.0)
    ratio = [1.4] * 10 + [2.0, 2.4]
    return make_clip(12, {1: ("home", ALL, (10, 10))}, ball, size_ratio=ratio)


def _mask(mode, **overrides):
    cfg = EventsConfig().with_params({"ball.airborne_mode": mode, **overrides}).ball
    return airborne_mask(_clip(), cfg)


def test_baseline_is_median_ratio_of_balls_near_players():
    clip = _clip()
    players = [p.xy for p in clip.players]
    assert ground_ratio_baseline(clip.ball_xy, clip.ball_size_ratio, players, 1.5, 10) == 1.4
    assert ground_ratio_baseline(clip.ball_xy, clip.ball_size_ratio, players, 1.5, 11) is None


def test_off_flags_nothing():
    mask, baseline = _mask("off")
    assert not mask.any() and baseline is None


def test_fixed_flags_both_large_balls():
    mask, baseline = _mask("fixed")  # 2.0 > 1.6 and 2.4 > 1.6
    assert mask.tolist() == [False] * 10 + [True, True]
    assert baseline is None


def test_calibrated_flags_only_balls_large_relative_to_this_clip():
    mask, baseline = _mask("calibrated")  # 2.0 / 1.4 = 1.43 < 1.6; 2.4 / 1.4 = 1.71 > 1.6
    assert baseline == pytest.approx(1.4)
    assert mask.tolist() == [False] * 10 + [False, True]


def test_calibrated_falls_back_to_fixed_without_enough_ground_balls():
    mask, baseline = _mask("calibrated", **{"ball.calibration_min_samples": 50})
    assert baseline is None
    assert mask.tolist() == [False] * 10 + [True, True]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        EventsConfig().with_params({"ball.airborne_mode": "sometimes"})


def test_no_ball_no_flags():
    clip = make_clip(5, {}, {})
    mask, _ = airborne_mask(clip, EventsConfig().ball)
    assert not mask.any() and len(mask) == 5 and np.isnan(clip.ball_xy).all()
