import pytest
from pydantic import ValidationError

from match_tagger.events.schema import Event, EventsFile


def test_minimal_event():
    e = Event(id="e1", t_sec=3.5, type="pass", team="home")
    assert (e.x_m, e.y_m, e.confidence) == (None, None, None)


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "tackle"},
        {"team": 0},
        {"t_sec": -0.1},
        {"x_m": 10.0},  # y_m missing
        {"confidence": 1.2},
        {"id": ""},
    ],
)
def test_invalid_events_are_rejected(fields):
    with pytest.raises(ValidationError):
        Event(**{"id": "e1", "t_sec": 1.0, "type": "shot", "team": "away", **fields})


def test_events_file_round_trip():
    f = EventsFile(clip_id="c1", events=[Event(id="p1", t_sec=1.0, type="shot", team="away",
                                               x_m=90.0, y_m=30.0, confidence=0.7)])  # fmt: skip
    assert EventsFile.model_validate_json(f.model_dump_json()) == f
