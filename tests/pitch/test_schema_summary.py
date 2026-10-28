import pandas as pd
import pytest
from pydantic import ValidationError

from match_tagger.pitch.geometry import PitchDims
from match_tagger.pitch.schema import (
    HomographyRecord,
    PositionRecord,
    read_homographies,
    read_positions,
    write_homographies,
    write_positions,
)
from match_tagger.pitch.summary import player_plausibility, quality_shares


def _pos(frame=0, track_id=1, cls="player", x=10.0, y=20.0, q="ok", **kw):
    return PositionRecord(
        frame=frame, t_sec=frame / 25, track_id=track_id, cls=cls, x_m=x, y_m=y, h_quality=q, **kw
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"q": "missing"},  # missing needs null coordinates
        {"x": None, "y": None, "q": "ok"},  # ok needs coordinates
        {"x": None},  # only one coordinate
        {"x": None, "y": None, "q": "missing", "out_of_pitch": True},
        {"cls": "referee", "team": 1},
    ],
)
def test_invalid_positions_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _pos(**overrides)


def test_positions_round_trip(tmp_path):
    rows = [
        _pos(team=0),
        _pos(track_id=None, cls="ball", x=-6.0, out_of_pitch=True),
        _pos(x=None, y=None, q="missing"),
    ]
    path = tmp_path / "positions.parquet"
    assert write_positions(rows, path) == 3
    df = read_positions(path)
    assert df["h_quality"].tolist() == ["ok", "ok", "missing"]
    assert df["out_of_pitch"].tolist() == [False, True, False]
    assert df["track_id"].tolist() == [1, pd.NA, 1]
    assert df["x_m"].isna().tolist() == [False, False, True]


def test_homographies_round_trip(tmp_path):
    rows = [
        HomographyRecord(frame=0, t_sec=0.0, h_quality="ok", matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1]),
        HomographyRecord(frame=2, t_sec=0.08, h_quality="missing"),
    ]
    path = tmp_path / "homographies.parquet"
    assert write_homographies(rows, path) == 2
    assert read_homographies(path) == rows
    with pytest.raises(ValidationError):
        HomographyRecord(frame=0, t_sec=0.0, h_quality="reused")  # reused needs a matrix


def test_quality_shares():
    assert quality_shares(["ok", "ok", "reused", "missing"]) == {
        "ok": 0.5, "reused": 0.25, "missing": 0.25,
    }  # fmt: skip


def test_player_plausibility():
    # frames 0, 5, 10 at 25 fps -> 0.2 s steps. Track 1 moves 1 m, then 4 m (20 m/s = jump).
    rows = [
        _pos(0, 1, x=10.0), _pos(5, 1, x=11.0), _pos(10, 1, x=15.0),
        _pos(0, 2, x=-1.0), _pos(10, 2, x=-1.0),  # outside by 1 m; frames not consecutive
        _pos(0, 3, x=None, y=None, q="missing"),
        _pos(0, None, cls="ball", x=50.0),
    ]  # fmt: skip
    stats = player_plausibility(rows, PitchDims(), {0: 5, 5: 10}, max_speed_mps=12.0)
    assert stats["player_points"] == 5
    assert stats["player_inside_share"] == pytest.approx(3 / 5)
    assert stats["player_out_of_pitch_share"] == 0.0
    assert stats["player_steps"] == 2
    assert stats["player_jump_share"] == pytest.approx(0.5)
