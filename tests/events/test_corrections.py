from pathlib import Path

import pytest
from pydantic import ValidationError

from match_tagger.events.corrections import (
    append_entry,
    merge,
    merged_json,
    read_log,
    undo_target,
)
from match_tagger.events.schema import Event

E1 = Event(id="m-e0001", t_sec=3.0, type="pass", team="home", x_m=40.0, y_m=20.0, confidence=0.8)
E2 = Event(id="m-e0002", t_sec=7.5, type="ball_loss", team="away", confidence=0.6)
EVENTS = [E1, E2]


def _log(tmp_path, *entries):
    path = tmp_path / "corrections" / "m.jsonl"
    for fields in entries:
        append_entry(path, **fields)
    return path


def test_replaying_the_same_log_twice_gives_identical_output(tmp_path):
    added = Event(id="m-a1b2c3d4", t_sec=5.0, type="pass", team="away")
    path = _log(
        tmp_path,
        {"action": "accept", "event_id": E1.id, "before": E1},
        {
            "action": "change_type",
            "event_id": E2.id,
            "before": E2,
            "after": E2.model_copy(update={"type": "pass"}),
        },
        {"action": "add", "event_id": added.id, "after": added},
    )
    first = merged_json(merge(EVENTS, read_log(path)))
    second = merged_json(merge(EVENTS, read_log(path)))
    assert first == second
    result = merge(EVENTS, read_log(path))
    assert [e.id for e in result.events] == [E1.id, added.id, E2.id]  # sorted by time
    assert result.status == {E1.id: "accepted", added.id: "added", E2.id: "corrected"}
    assert result.conflicts == []


def test_delete_followed_by_undo_restores_the_event(tmp_path):
    path = _log(tmp_path, {"action": "delete", "event_id": E2.id, "before": E2})
    assert [e.id for e in merge(EVENTS, read_log(path)).events] == [E1.id]
    append_entry(path, action="undo", target_seq=undo_target(read_log(path)))
    result = merge(EVENTS, read_log(path))
    assert result.events == EVENTS
    assert result.status[E2.id] == "unreviewed"


def test_added_events_must_be_valid_events(tmp_path):
    path = tmp_path / "m.jsonl"
    with pytest.raises(ValidationError):
        append_entry(
            path,
            action="add",
            event_id="m-ax",
            after={"id": "m-ax", "t_sec": -1.0, "type": "pass", "team": "home"},
        )
    with pytest.raises(ValidationError):
        append_entry(
            path,
            action="add",
            event_id="m-ay",
            after={"id": "m-ay", "t_sec": 1.0, "type": "tackle", "team": "home"},
        )
    assert not path.exists()  # nothing was written


def test_log_is_append_only(tmp_path):
    path = _log(tmp_path, {"action": "accept", "event_id": E1.id, "before": E1})
    first_line = path.read_text(encoding="utf-8")
    append_entry(path, action="delete", event_id=E2.id, before=E2)
    assert path.read_text(encoding="utf-8").startswith(first_line)
    assert [e.seq for e in read_log(path)] == [1, 2]


def test_change_team_changes_only_the_team():
    with pytest.raises(ValidationError):  # also changes the time
        append_entry(
            Path("unused.jsonl"),
            action="change_team",
            event_id=E1.id,
            before=E1,
            after=E1.model_copy(update={"team": "away", "t_sec": 9.0}),
        )


def test_conflict_when_the_event_is_not_as_expected(tmp_path):
    stale = E1.model_copy(update={"t_sec": 99.0})  # e.g. events.json was regenerated
    path = _log(tmp_path, {"action": "delete", "event_id": E1.id, "before": stale})
    result = merge(EVENTS, read_log(path))
    assert result.events == EVENTS
    assert [c.seq for c in result.conflicts] == [1]


def test_undo_of_an_undo_is_a_conflict(tmp_path):
    path = _log(
        tmp_path,
        {"action": "delete", "event_id": E1.id, "before": E1},
        {"action": "undo", "target_seq": 1},
        {"action": "undo", "target_seq": 2},
    )
    result = merge(EVENTS, read_log(path))
    assert result.events == EVENTS  # the first undo still applies
    assert [c.seq for c in result.conflicts] == [3]


def test_undo_target_skips_undone_entries(tmp_path):
    path = _log(
        tmp_path,
        {"action": "accept", "event_id": E1.id, "before": E1},
        {"action": "delete", "event_id": E2.id, "before": E2},
        {"action": "undo", "target_seq": 2},
    )
    assert undo_target(read_log(path)) == 1
