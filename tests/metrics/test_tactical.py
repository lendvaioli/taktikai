import pytest

from match_tagger.events.schema import Event
from match_tagger.metrics.config import (
    MetricsConfig,
    PassNetworkConfig,
    PressingConfig,
    ReliabilityConfig,
)
from match_tagger.metrics.context import MatchContext, attack_sign
from match_tagger.metrics.sample import reliability, sample, team_summary
from match_tagger.metrics.stage import compute_metrics
from match_tagger.metrics.tactical import (
    Unavailable,
    ball_losses_by_zone,
    pass_network,
    ppda_proxy,
    shot_map,
)
from match_tagger.pitch.geometry import PitchDims

# 90 x 60 pitch: thirds end at x = 30 / 60, lanes at y = 20 / 40; PPDA area x <= 54.
DIMS = PitchDims(length_m=90, width_m=60)


def _ctx(home_attacks="right", source="pipeline"):
    return MatchContext(
        match_id="m",
        source=source,
        duration_sec=1800,
        teams={"home": "H", "away": "A"},
        has_video=source == "pipeline",
        dims=DIMS,
        home_attacks=home_attacks,
    )


def _ev(id_, type_, team, x=None, y=None, **kw):
    return Event(id=id_, t_sec=1.0, type=type_, team=team, x_m=x, y_m=y, **kw)


# Home attacks right (attack frame = absolute); away attacks left (attack frame turned 180°).
EVENTS = [
    _ev("p1", "pass", "home", 10, 10, end_x_m=40, end_y_m=30, outcome="complete"),
    _ev("p2", "pass", "home", 50, 50, end_x_m=80, end_y_m=50, outcome="incomplete"),
    _ev("p3", "pass", "home", 85, 30, outcome="complete"),
    _ev("l1", "ball_loss", "home", 70, 10),
    _ev("l2", "ball_loss", "home", 20, 50),
    _ev("l3", "ball_loss", "home"),
    _ev("a1", "pass", "away", 80, 10, end_x_m=60, end_y_m=30, outcome="complete"),  # (10, 50)
    _ev("a2", "pass", "away", 20, 30, outcome="incomplete"),  # attack frame (70, 30)
    _ev("b1", "ball_loss", "away", 75, 40),  # attack frame (15, 20)
]


def test_attack_sign():
    assert attack_sign(_ctx(), "home", None) == 1
    assert attack_sign(_ctx(), "away", None) == -1
    assert attack_sign(_ctx({1: "left", 2: "right"}), "home", 2) == 1
    assert attack_sign(_ctx({1: "left", 2: "right"}), "home", None) is None
    assert attack_sign(_ctx(None), "home", 1) is None


def test_ball_losses_by_attack_zone():
    result = ball_losses_by_zone(EVENTS, _ctx(), "home")
    assert (result.frame, result.total, result.located) == ("attack", 3, 2)
    assert {k: (z.count, z.share) for k, z in result.zones.items()} == {
        "final_right": (1, 0.5),
        "own_left": (1, 0.5),
    }
    assert {k: z.count for k, z in result.thirds.items()} == {"final": 1, "own": 1}
    assert {k: z.count for k, z in result.lanes.items()} == {"left": 1, "right": 1}
    away = ball_losses_by_zone(EVENTS, _ctx(), "away")
    assert list(away.zones) == ["own_centre"]
    assert ball_losses_by_zone(EVENTS, _ctx(None), "home").thirds == {}


def test_ball_losses_by_absolute_zone_without_direction():
    result = ball_losses_by_zone(EVENTS, _ctx(None), "home")
    # (70, 10): column 4, row 0 -> 13; (20, 50): column 1, row 2 -> 6
    assert result.frame == "absolute"
    assert set(result.zones) == {"13", "6"}


def test_zone_pass_network():
    net = pass_network(EVENTS, _ctx(), "home", "zone", PassNetworkConfig())
    assert (net.passes, net.complete, net.completion_share) == (3, 2, 0.667)
    assert {(e.source, e.target, e.count) for e in net.edges} == {
        ("own_right", "middle_centre", 1),
        ("middle_left", "final_left", 1),
    }
    assert {n.id: n.passes for n in net.nodes} == {
        "own_right": 1,
        "middle_left": 1,
        "final_centre": 1,
    }
    away = pass_network(EVENTS, _ctx(), "away", "zone", PassNetworkConfig())
    assert [(e.source, e.target) for e in away.edges] == [("own_left", "middle_centre")]


def test_player_pass_network():
    passes = [
        _ev("1", "pass", "home", 10, 10, player="A", recipient="B", outcome="complete"),
        _ev("2", "pass", "home", 30, 20, player="A", recipient="B", outcome="complete"),
        _ev("3", "pass", "home", 20, 30, player="A", recipient="C", outcome="incomplete"),
        _ev("4", "pass", "home", 50, 40, player="B", recipient="A", outcome="complete"),
    ]
    net = pass_network(passes, _ctx(source="statsbomb"), "home", "player", PassNetworkConfig())
    assert [(e.source, e.target, e.count) for e in net.edges] == [("A", "B", 2), ("B", "A", 1)]
    assert [(n.id, n.passes, n.mean_x_m, n.mean_y_m) for n in net.nodes] == [
        ("A", 3, 20.0, 20.0),
        ("B", 1, 50.0, 40.0),
    ]
    capped = pass_network(passes, _ctx(), "home", "player", PassNetworkConfig(min_edge_count=2))
    assert len(capped.edges) == 1
    assert isinstance(
        pass_network(EVENTS, _ctx(), "home", "player", PassNetworkConfig()), Unavailable
    )


def test_ppda_proxy():
    cfg = PressingConfig()
    # home presses: away passes in away's own 60 % -> a1 (x 10) yes, a2 (x 70) no; b1 (x 15)
    home = ppda_proxy(EVENTS, _ctx(), "home", cfg)
    assert (home.opponent_passes, home.opponent_ball_losses, home.value) == (1, 1, 1.0)
    # away presses: home passes p1 (10), p2 (50) yes, p3 (85) no; losses l2 (20) yes, l1 (70) no
    away = ppda_proxy(EVENTS, _ctx(), "away", cfg)
    assert (away.opponent_passes, away.opponent_ball_losses, away.value) == (2, 1, 2.0)
    assert isinstance(ppda_proxy(EVENTS, _ctx(None), "home", cfg), Unavailable)
    no_losses = ppda_proxy(EVENTS[:3], _ctx(), "away", cfg)
    assert (no_losses.opponent_passes, no_losses.value) == (2, None)


def test_shot_map():
    shots = [
        _ev("s1", "shot", "home", 80, 30, xg=0.1, outcome="goal"),
        _ev("s2", "shot", "home", 75, 20, xg=0.25, outcome="saved"),
        _ev("s3", "shot", "away", 10, 30, xg=0.3, outcome="off_target"),
    ]
    home = shot_map(shots, _ctx(source="statsbomb"), "home")
    assert (home.shots, home.goals, home.xg_total) == (2, 1, 0.35)
    away = shot_map(shots, _ctx(source="statsbomb"), "away")
    assert (away.items[0].x_m, away.items[0].y_m) == (80, 30)  # turned into away's attack frame
    assert isinstance(shot_map(shots, _ctx(), "home"), Unavailable)


def test_sample_and_reliability():
    cfg = ReliabilityConfig()
    block = sample(_ctx(), EVENTS, cfg)
    assert (block.duration_min, block.n_events, block.reliability) == (30.0, 9, "very_low")
    assert block.events_per_type == {"ball_loss": 4, "pass": 5}
    assert reliability(5, 100, cfg) == "very_low"
    assert reliability(30, 100, cfg) == "low"
    assert reliability(50, 100, cfg) == "ok"


def test_team_summary():
    summary = team_summary(EVENTS, "home")
    assert (summary.passes, summary.passes_complete, summary.ball_losses) == (3, 2, 3)
    assert summary.pass_completion_share == pytest.approx(0.667)


def test_compute_metrics_uses_zone_network_for_own_clips():
    metrics = compute_metrics(EVENTS, _ctx(), MetricsConfig())
    assert metrics.pass_network["home"].node == "zone"
    assert metrics.ppda["away"].value == 2.0
    assert metrics.zone_labels_hu["final_left"] == "támadóharmad, bal sáv"
    assert compute_metrics(EVENTS, _ctx(None), MetricsConfig()).zone_labels_hu == {}
