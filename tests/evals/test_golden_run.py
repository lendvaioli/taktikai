import json

import pytest
from pydantic import ValidationError

from match_tagger.evals.cli import main
from match_tagger.evals.golden import (
    GoldenClip,
    GoldenSet,
    list_golden_sets,
    load_golden_set,
    save_golden_set,
)
from match_tagger.evals.report import compare_runs, compare_table, set_table
from match_tagger.evals.run import EvalRun, evaluate_set, file_prediction_loader
from match_tagger.events.schema import Event, EventsFile


def ev(id_, t, type_="pass", team="home"):
    return Event(id=id_, t_sec=t, type=type_, team=team)


def _clip(clip_id, events, complete=True):
    return GoldenClip(clip_id=clip_id, video=f"data/raw/x/{clip_id}.mp4", fps=25,
                      duration_sec=30, complete=complete, events=events)  # fmt: skip


def _set():
    return GoldenSet(set_id="dfl", source="roboflow/sports", clips=[
        _clip("a", [ev("a1", 5.0), ev("a2", 10.0, "shot", "away")]),
        _clip("b", [ev("b1", 3.0)]),
        _clip("c", [ev("c1", 1.0)], complete=False),
    ])  # fmt: skip


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError):
        GoldenSet(set_id="s", clips=[_clip("a", []), _clip("a", [])])
    with pytest.raises(ValidationError):
        GoldenSet(set_id="s", clips=[_clip("a", [ev("e", 1.0)]), _clip("b", [ev("e", 2.0)])])


def test_save_sorts_events_and_round_trips(tmp_path):
    golden = GoldenSet(set_id="s", clips=[_clip("a", [ev("e2", 9.0), ev("e1", 2.0)])])
    save_golden_set(golden, tmp_path)
    loaded = load_golden_set(tmp_path, "s")
    assert [e.id for e in loaded.clips[0].events] == ["e1", "e2"]
    assert list_golden_sets(tmp_path) == ["s"]
    assert not list(tmp_path.glob("*.tmp"))


def test_evaluate_set_per_clip_complete_only_and_missing_predictions():
    preds = {"a": [ev("p1", 5.5), ev("p2", 3.0)], "c": [ev("p3", 1.0)]}
    result = evaluate_set(_set(), preds.get, tolerance_sec=2.0)
    assert result.n_clips_evaluated == 2
    assert result.skipped_incomplete == ["c"]
    assert result.missing_predictions == ["b"]
    # pass: a1 matched by p1; p2 (3.0 in clip a) must not match b1 (3.0 in clip b)
    assert (result.per_type["pass"].tp, result.per_type["pass"].fp) == (1, 1)
    assert result.per_type["pass"].fn == 1
    assert result.per_type["shot"].fn == 1
    assert (result.overall.tp, result.overall.fp, result.overall.fn) == (1, 1, 2)
    assert "no predictions for: b" in set_table(result)


def _write_events(path, clip_id, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EventsFile(clip_id=clip_id, events=events).model_dump_json())


def test_cli_eval_writes_results_and_leaves_golden_unchanged(tmp_path, capsys):
    golden_dir, results_dir = tmp_path / "golden", tmp_path / "results"
    save_golden_set(_set(), golden_dir)
    before = (golden_dir / "dfl.json").read_bytes()
    _write_events(tmp_path / "pred" / "a" / "events.json", "a", [ev("p1", 5.0)])
    config = tmp_path / "configs"
    config.mkdir()
    (config / "evals.yaml").write_text(json.dumps({
        "golden_dir": str(golden_dir), "results_dir": str(results_dir),
        "predictions": str(tmp_path / "pred" / "{clip_id}" / "events.json"),
    }))  # fmt: skip
    assert main(["--config-dir", str(config), "eval", "--set", "dfl"]) == 0
    assert (golden_dir / "dfl.json").read_bytes() == before
    run = EvalRun.model_validate_json(next(results_dir.glob("*.json")).read_text())
    assert run.sets[0].per_type["pass"].tp == 1
    assert "pass" in capsys.readouterr().out


def test_file_loader_returns_none_for_missing_file(tmp_path):
    assert file_prediction_loader(str(tmp_path / "{clip_id}.json"))("nope") is None


def test_compare_runs_differences():
    loader_a = {"a": [ev("p1", 5.0)], "b": []}.get
    loader_b = {"a": [ev("p1", 5.0)], "b": [ev("p2", 3.0)]}.get
    golden = _set()
    run_a = EvalRun(created_at="2026-10-27T10:00:00Z", git_commit=None, tolerance_sec=2,
                    sets=[evaluate_set(golden, loader_a, 2.0)])  # fmt: skip
    run_b = EvalRun(created_at="2026-10-27T11:00:00Z", git_commit=None, tolerance_sec=2,
                    sets=[evaluate_set(golden, loader_b, 2.0)])  # fmt: skip
    rows = {(r.row, r.metric): r for r in compare_runs(run_a, run_b)}
    # pass recall: A 1/2 -> B 2/2
    assert (rows["pass", "recall"].a, rows["pass", "recall"].b) == (0.5, 1.0)
    assert rows["pass", "recall"].diff == pytest.approx(0.5)
    # shot precision undefined in both runs -> no difference
    assert rows["shot", "precision"].diff is None
    assert "+0.500" in compare_table(list(rows.values()))


def test_compare_set_missing_in_one_run():
    run_a = EvalRun(created_at="2026-10-27T10:00:00Z", git_commit=None, tolerance_sec=2, sets=[])
    run_b = EvalRun(created_at="2026-10-27T11:00:00Z", git_commit=None, tolerance_sec=2,
                    sets=[evaluate_set(_set(), {}.get, 2.0)])  # fmt: skip
    rows = compare_runs(run_a, run_b)
    assert all(r.a is None and r.diff is None for r in rows)
