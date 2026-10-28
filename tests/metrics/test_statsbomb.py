import pytest

from match_tagger.metrics.statsbomb import context, convert, point, t_sec
from match_tagger.pitch.geometry import PitchDims

MATCH = {
    "match_id": 1,
    "home_team": {"home_team_name": "H"},
    "away_team": {"away_team_name": "A"},
}


def _e(
    index, period, minute, second, ts, type_, team, poss, poss_team, pattern="Regular Play", **kw
):
    return {
        "index": index,
        "period": period,
        "minute": minute,
        "second": second,
        "timestamp": ts,
        "type": {"name": type_},
        "team": {"name": team},
        "player": {"name": f"{team}{index}"},
        "possession": poss,
        "possession_team": {"name": poss_team},
        "play_pattern": {"name": pattern},
        **kw,
    }


RAW = [
    _e(1, 1, 0, 0, "00:00:00.000", "Pass", "H", 1, "H", "From Kick Off",
       location=[60, 40], **{"pass": {"type": {"name": "Kick Off"}}}),
    _e(2, 1, 0, 5, "00:00:05.500", "Pass", "H", 1, "H",
       location=[30, 20], **{"pass": {"end_location": [60, 40], "recipient": {"name": "H9"}}}),
    _e(3, 1, 0, 8, "00:00:08.000", "Pass", "H", 1, "H",
       location=[60, 40], **{"pass": {"outcome": {"name": "Incomplete"}}}),
    _e(4, 1, 0, 9, "00:00:09.250", "Ball Recovery", "A", 2, "A", location=[30, 20]),
    _e(5, 1, 0, 12, "00:00:12.000", "Shot", "A", 2, "A", location=[110, 40],
       shot={"statsbomb_xg": 0.12, "outcome": {"name": "Goal"}, "type": {"name": "Open Play"}}),
    _e(6, 1, 0, 20, "00:00:20.000", "Pass", "H", 3, "H", "From Kick Off",
       location=[60, 40], **{"pass": {"type": {"name": "Kick Off"}}}),
    _e(7, 2, 45, 0, "00:00:00.000", "Pass", "A", 4, "A", "From Kick Off",
       location=[60, 40], **{"pass": {"type": {"name": "Kick Off"}}}),
    _e(8, 2, 50, 0, "00:05:00.000", "Shot", "H", 5, "H", "From Free Kick", location=[108, 40],
       shot={"statsbomb_xg": 0.78, "outcome": {"name": "Saved"}, "type": {"name": "Penalty"}}),
    _e(9, 5, 120, 0, "00:00:00.000", "Shot", "H", 6, "H", location=[108, 40],
       shot={"statsbomb_xg": 0.78, "outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}),
]  # fmt: skip


def test_point_scales_flips_and_turns_away_events():
    assert point([30, 20], home=True, dims=PitchDims()) == (26.25, 51.0)
    assert point([30, 20], home=False, dims=PitchDims()) == (78.75, 17.0)
    assert point(None, home=True, dims=PitchDims()) is None


def test_t_sec_uses_match_clock_and_fraction():
    assert t_sec(RAW[1]) == 5.5
    assert t_sec(RAW[7]) == 3000.0


def test_convert_types_and_fields():
    converted = convert(RAW, MATCH)  # sorted by (t_sec, id)
    events = {e.id: e for e in converted}
    assert [(e.id, e.type, e.team) for e in converted] == [
        ("e1", "set_piece", "home"),
        ("e2", "pass", "home"),
        ("e3", "pass", "home"),
        ("e4-loss", "ball_loss", "home"),
        ("e5", "shot", "away"),
        ("e6", "set_piece", "home"),
        ("e7", "set_piece", "away"),
        ("e8", "shot", "home"),
        ("e8-loss", "ball_loss", "away"),
        ("e8-sp", "set_piece", "home"),
    ]
    p = events["e2"]
    assert (p.x_m, p.y_m, p.end_x_m, p.end_y_m) == (26.25, 51.0, 52.5, 34.0)
    assert (p.outcome, p.player, p.recipient, p.period) == ("complete", "H2", "H9", 1)
    assert events["e3"].outcome == "incomplete"
    loss = events["e4-loss"]  # located where A won the ball, seen from A
    assert (loss.t_sec, loss.x_m, loss.y_m, loss.player) == (9.25, 78.75, 17.0, None)
    assert (events["e5"].xg, events["e5"].outcome) == (0.12, "goal")
    assert events["e8"].outcome == "saved"


def test_kick_off_and_period_change_are_not_ball_losses():
    losses = [e.id for e in convert(RAW, MATCH) if e.type == "ball_loss"]
    assert losses == ["e4-loss", "e8-loss"]


def test_context():
    ctx = context(RAW, MATCH)
    assert (ctx.match_id, ctx.source, ctx.has_video, ctx.home_attacks) == (
        "sb_1",
        "statsbomb",
        False,
        "right",
    )
    assert ctx.duration_sec == pytest.approx(3000.0)  # the shoot-out is left out
    assert ctx.teams == {"home": "H", "away": "A"}
    assert ctx.teams_hu == {"home": "Hazai csapat", "away": "Vendég csapat"}  # not mapped
    assert context(RAW, MATCH, {"H": "Hálóország"}).teams_hu["home"] == "Hálóország"
    assert "StatsBomb" in ctx.attribution
