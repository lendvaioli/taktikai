import numpy as np

from match_tagger.evals.diagnostics import false_pass_causes, miss_cause, observable
from match_tagger.events.config import DiagnosticsConfig, EventsConfig
from match_tagger.events.schema import Event
from match_tagger.events.stage import detect_events
from tests.events.synthetic import hold, make_clip

DIAG = DiagnosticsConfig()
ALL = range(0, 60)


def _golden(t, team="home"):
    return Event(id="g", t_sec=t, type="pass", team=team)


def _clip(ball, h_quality=None, players=None):
    players = players or {1: ("home", ALL, (10, 10))}
    return make_clip(40, players, ball, h_quality=h_quality)


def _cause(clip, golden, preds=(), mapping_correct=True):
    det = detect_events(clip, EventsConfig())
    return miss_cause(golden, clip, det, list(preds), mapping_correct, 2.0, DIAG)


def test_observable_window():
    times = np.arange(10) * 0.08
    known = np.zeros(10, dtype=bool)
    known[8] = True  # t = 0.64
    assert observable(times, known, 0.2, 0.5)  # |0.64 - 0.2| = 0.44 <= 0.5
    assert not observable(times, known, 0.1, 0.5)  # 0.54 > 0.5


def test_miss_causes_in_order():
    ball_far = hold(range(0, 40), (20.0, 10.0))  # 10 m from the only player
    ball_near = hold(range(0, 40), (10.5, 10.0))
    assert _cause(_clip({}), _golden(1.0)) == "ball_gap"
    assert _cause(_clip(ball_far), _golden(1.0)) == "no_player_near"
    reused = ["ok"] * 10 + ["reused"] * 30
    assert _cause(_clip(ball_near, h_quality=reused), _golden(1.0)) == "homography"
    assert _cause(_clip(ball_near), _golden(1.0), mapping_correct=False) == "team_mapping"
    late = Event(id="p", t_sec=4.0, type="pass", team="home")  # 3 s off: beyond tolerance
    assert _cause(_clip(ball_near), _golden(1.0), preds=[late]) == "timing"
    assert _cause(_clip(ball_near), _golden(1.0)) == "other"


def test_false_pass_from_track_break_is_counted_separately():
    ball = hold(range(0, 26), (10.5, 10))
    players = {1: ("home", range(0, 13), (10, 10)), 2: ("home", range(13, 26), (10, 10))}
    clip = make_clip(26, players, ball)
    det = detect_events(clip, EventsConfig().with_params({"rules.pass_min_ball_travel_m": 0.0}))
    assert false_pass_causes(det.events, clip, det, DIAG) == {"track_break": 1}


def test_real_pass_between_teammates_is_not_a_track_break():
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(18, 31), (19.5, 10))}
    players = {1: ("home", ALL, (10, 10)), 2: ("home", ALL, (20, 10))}
    clip = make_clip(31, players, ball)
    det = detect_events(clip, EventsConfig())
    assert false_pass_causes(det.events, clip, det, DIAG) == {"other": 1}


def test_track_break_found_when_pass_is_timed_inside_the_transition():
    ball = {**hold(range(0, 13), (10.5, 10)), **hold(range(18, 31), (10.5, 10))}
    players = {1: ("home", range(0, 13), (10, 10)), 2: ("home", range(18, 31), (10, 10))}
    clip = make_clip(31, players, ball)
    params = {"rules.pass_min_ball_travel_m": 0.0, "rules.pass_time_fraction": 0.5}
    det = detect_events(clip, EventsConfig().with_params(params))
    assert [e.t_sec for e in det.events] == [1.2]
    assert false_pass_causes(det.events, clip, det, DIAG) == {"track_break": 1}
