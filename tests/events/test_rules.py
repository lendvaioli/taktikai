import pytest

from match_tagger.events.config import EventsConfig
from match_tagger.events.stage import detect_events
from tests.events.synthetic import DT, hold, make_clip

CFG = EventsConfig()  # start values: radius 1.5, hold 0.24, transition 3, travel 3, intercept 5
ALL = range(0, 60)


def _summary(det):
    return [(e.type, e.team, e.t_sec) for e in det.events]


def _pass_clip():
    # A1 (home, track 1) at x=10 holds frames 0-12, ball flies frames 13-17, A2 (home) at x=20
    ball = {
        **hold(range(0, 13), (10.5, 10)),
        13: (12, 10),
        14: (13.5, 10),
        15: (15, 10),
        16: (16.5, 10),
        17: (18, 10),
        **hold(range(18, 31), (19.5, 10)),
    }
    players = {1: ("home", ALL, (10, 10)), 2: ("home", ALL, (20, 10))}
    return make_clip(31, players, ball)


def test_pass_between_teammates_at_passers_last_frame():
    det = detect_events(_pass_clip(), CFG)
    assert _summary(det) == [("pass", "home", round(12 * DT, 2))]
    assert [(s.track_id, s.start, s.end) for s in det.segments] == [(1, 0, 12), (2, 18, 30)]


def test_pass_confidence_by_hand():
    # hold 13 frames = 1.04 s >= 2 x 0.24 -> 1.0; gap 0.48 s -> 1 - 0.48/3 = 0.84;
    # mean ball distance 0.5 m -> 1 - 0.5/1.5 = 0.6667; mean = 0.8356
    assert detect_events(_pass_clip(), CFG).events[0].confidence == pytest.approx(0.836)


def test_pass_location_is_ball_at_release():
    event = detect_events(_pass_clip(), CFG).events[0]
    assert (event.x_m, event.y_m) == (10.5, 10.0)


def test_lost_duel_is_ball_loss_only_at_opponents_first_frame():
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(13, 26), (11.2, 10))}
    players = {1: ("home", ALL, (10, 10)), 5: ("away", ALL, (11.5, 10))}
    det = detect_events(make_clip(26, players, ball), CFG)
    assert _summary(det) == [("ball_loss", "home", round(13 * DT, 2))]


def test_intercepted_pass_gives_pass_and_ball_loss():
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(18, 31), (24.5, 10))}
    players = {1: ("home", ALL, (10, 10)), 5: ("away", ALL, (25, 10))}
    det = detect_events(make_clip(31, players, ball), CFG)
    assert _summary(det) == [
        ("pass", "home", round(12 * DT, 2)),
        ("ball_loss", "home", round(18 * DT, 2)),
    ]


def _track_break_clip():
    # one player re-identified: track 1 frames 0-12, track 2 frames 13-25, same spot, ball still
    ball = hold(range(0, 26), (10.5, 10))
    players = {1: ("home", range(0, 13), (10, 10)), 2: ("home", range(13, 26), (10, 10))}
    return make_clip(26, players, ball)


def test_track_break_is_not_a_pass():
    assert detect_events(_track_break_clip(), CFG).events == []


def test_track_break_becomes_a_pass_without_min_travel():
    cfg = CFG.with_params({"rules.pass_min_ball_travel_m": 0.0})
    assert _summary(detect_events(_track_break_clip(), cfg)) == [("pass", "home", 0.96)]


def test_transition_longer_than_limit_gives_no_event():
    # A1 holds frames 0-12, A2 from frame 63: gap = 50 frames = 4.0 s > 3.0 s
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(63, 76), (19.5, 10))}
    players = {1: ("home", range(0, 80), (10, 10)), 2: ("home", range(0, 80), (20, 10))}
    assert detect_events(make_clip(76, players, ball), CFG).events == []


def test_pass_time_fraction_moves_the_pass_into_the_transition():
    # A ends at frame 12 (0.96 s), B starts at frame 18 (1.44 s): 0.96 + 0.25 x 0.48 = 1.08 s
    cfg = CFG.with_params({"rules.pass_time_fraction": 0.25})
    event = detect_events(_pass_clip(), cfg).events[0]
    assert event.t_sec == pytest.approx(1.08)
    assert (event.x_m, event.y_m) == (10.5, 10.0)  # still located at the release point


def test_pass_time_fraction_does_not_move_ball_losses():
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(18, 31), (24.5, 10))}
    players = {1: ("home", ALL, (10, 10)), 5: ("away", ALL, (25, 10))}
    cfg = CFG.with_params({"rules.pass_time_fraction": 0.5})
    det = detect_events(make_clip(31, players, ball), cfg)
    assert _summary(det) == [("pass", "home", 1.2), ("ball_loss", "home", round(18 * DT, 2))]
