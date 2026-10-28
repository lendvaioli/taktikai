from match_tagger.evals.golden import GoldenClip, GoldenSet
from match_tagger.events.corrections import MergeResult, append_entry, merge, read_log
from match_tagger.events.review import (
    correction_for,
    filter_rows,
    golden_candidate,
    review_rows,
    write_candidate,
)
from match_tagger.events.schema import Event
from match_tagger.pitch.geometry import PitchDims

DIMS = PitchDims()
P1 = Event(id="p1", t_sec=3.0, type="pass", team="home", x_m=10.0, y_m=5.0, confidence=0.9)
P2 = Event(id="p2", t_sec=9.0, type="pass", team="home", confidence=0.4)  # no golden near
P3 = Event(id="p3", t_sec=12.0, type="ball_loss", team="away", x_m=60.0, y_m=60.0)
MERGED = MergeResult(
    events=[P1, P2, P3],
    status={"p1": "accepted", "p2": "unreviewed", "p3": "corrected"},
    conflicts=[],
)
GOLDEN = GoldenClip(
    clip_id="c",
    video="v.mp4",
    fps=25,
    duration_sec=30,
    complete=True,
    events=[
        Event(id="g1", t_sec=3.5, type="pass", team="home"),
        Event(id="g2", t_sec=20.0, type="pass", team="away"),
        Event(id="g3", t_sec=12.4, type="ball_loss", team="away"),
    ],
)


def test_rows_without_golden_have_no_eval_label():
    rows = review_rows(MERGED, None, DIMS, 2.0)
    assert [(r.id, r.zone, r.status, r.eval) for r in rows] == [
        ("p1", 1, "accepted", None),
        ("p2", None, "unreviewed", None),
        ("p3", 12, "corrected", None),  # (60, 60): band 3, row 2 -> 12,
    ]


def test_eval_view_marks_tp_fp_and_lists_missed_golden_events():
    rows = review_rows(MERGED, GOLDEN, DIMS, 2.0)
    assert [(r.id, r.source, r.eval) for r in rows] == [
        ("p1", "prediction", "TP"),
        ("p2", "prediction", "FP"),
        ("p3", "prediction", "TP"),
        ("g2", "golden", "FN"),
    ]
    assert rows[-1].status is None


def test_filters_combine():
    rows = review_rows(MERGED, GOLDEN, DIMS, 2.0)
    assert [r.id for r in filter_rows(rows, evals={"FN", "FP"})] == ["p2", "g2"]
    assert [r.id for r in filter_rows(rows, types={"pass"}, teams={"home"})] == ["p1", "p2"]
    assert [r.id for r in filter_rows(rows, zones={12})] == ["p3"]
    assert [r.id for r in filter_rows(rows, zones={None})] == ["p2", "g2"]


def test_export_is_a_valid_incomplete_golden_clip_outside_evals_golden(tmp_path):
    candidate = golden_candidate(
        MERGED, "c", "data/raw/dfl/c.mp4", 25.0, 30.0, {"home": "fehér", "away": "piros"}
    )
    assert candidate.complete is False
    assert all(e.confidence is None for e in candidate.events)
    path = write_candidate(candidate, tmp_path / "candidates")
    assert path == tmp_path / "candidates" / "c.json"
    assert GoldenClip.model_validate_json(path.read_text(encoding="utf-8")) == candidate
    # it can be placed into a golden set as is
    golden = GoldenSet(set_id="own", clips=[candidate])
    assert golden.clip("c").events[0].id == "p1"
    assert not (tmp_path / "golden").exists()


def test_every_component_action_becomes_one_valid_log_entry(tmp_path):
    log = tmp_path / "m.jsonl"
    actions = [
        {"action": "accept", "id": "p1"},
        {"action": "change_type", "id": "p2", "type": "ball_loss"},
        {"action": "change_team", "id": "p3", "team": "home"},
        {"action": "add", "type": "pass", "team": "away", "t_sec": 5.4321},
        {"action": "delete", "id": "p1"},
        {"action": "undo"},  # restores p1 (accepted)
        {"action": "delete", "id": "gone"},  # unknown id -> nothing
    ]
    for value in actions:
        current = merge(MERGED.events, read_log(log))
        fields = correction_for(value, current, "m", read_log(log), lambda: "00000001")
        if fields is not None:
            append_entry(log, **fields)
    result = merge(MERGED.events, read_log(log))
    assert [e.action for e in read_log(log)] == [
        "accept",
        "change_type",
        "change_team",
        "add",
        "delete",
        "undo",
    ]
    assert [(e.id, e.type, e.team, e.t_sec) for e in result.events] == [
        ("p1", "pass", "home", 3.0),
        ("m-a00000001", "pass", "away", 5.43),
        ("p2", "ball_loss", "home", 9.0),
        ("p3", "ball_loss", "home", 12.0),
    ]
    assert result.status == {
        "p1": "accepted",
        "m-a00000001": "added",
        "p2": "corrected",
        "p3": "corrected",
    }


def test_undo_with_nothing_to_undo_gives_no_entry():
    assert correction_for({"action": "undo"}, MERGED, "m", [], lambda: "x") is None
