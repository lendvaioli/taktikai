import pandas as pd
import pytest
from pydantic import ValidationError

from match_tagger.detection.schema import DetectionRecord, read_detections, write_detections

BOX = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0}


def _record(**overrides):
    fields = {"frame": 5, "t_sec": 0.2, "track_id": 7, "cls": "player", "team": 1, "conf": 0.9}
    return DetectionRecord(**{**fields, **BOX, **overrides})


def test_valid_player_record():
    record = _record()
    assert (record.cls, record.team, record.track_id) == ("player", 1, 7)


def test_ball_without_track_or_team_is_valid():
    assert _record(cls="ball", track_id=None, team=None).track_id is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"cls": "referee", "team": 0},
        {"cls": "ball", "track_id": None, "team": 1},
        {"cls": "ball", "track_id": 3, "team": None},
        {"cls": "player", "track_id": None},
        {"cls": "coach"},
        {"team": 2},
        {"x2": 10.0},
        {"y2": 5.0},
        {"conf": 1.5},
        {"frame": -1},
    ],
)
def test_invalid_records_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _record(**overrides)


def test_parquet_round_trip_keeps_dtypes_and_nulls(tmp_path):
    records = [
        _record(),
        _record(cls="referee", track_id=9, team=None),
        _record(cls="ball", track_id=None, team=None, conf=0.4),
    ]
    path = tmp_path / "m" / "detections.parquet"
    assert write_detections(records, path) == 3

    df = read_detections(path)
    assert list(df.columns) == [
        "frame", "t_sec", "track_id", "cls", "team", "x1", "y1", "x2", "y2", "conf"
    ]  # fmt: skip
    assert str(df["track_id"].dtype) == "Int32"
    assert str(df["team"].dtype) == "Int8"
    assert df["track_id"].tolist() == [7, 9, pd.NA]
    assert df["team"].tolist() == [1, pd.NA, pd.NA]
    assert df["conf"].tolist() == pytest.approx([0.9, 0.9, 0.4])
