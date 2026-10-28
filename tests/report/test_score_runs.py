import argparse
from datetime import UTC, datetime

import pytest

from match_tagger.metrics.config import ReliabilityConfig
from match_tagger.metrics.context import MatchContext
from match_tagger.metrics.sample import Sample
from match_tagger.report.render import render_markdown
from match_tagger.report.runs import RunRecord, all_records, write_record
from match_tagger.report.schema import Report
from match_tagger.report.score import (
    ReportScore,
    append_score,
    ask_score,
    read_scores,
    score_run,
    summarize,
    unscored,
)

from .conftest import report_args


def _record(run_id, status="ok", version="report_system v3", backend="ollama", minute=0):
    return RunRecord(
        run_id=run_id, created_at=datetime(2026, 10, 28, 12, minute, tzinfo=UTC), match_id="m",
        backend=backend, model="qwen", prompt_version=version, prompt_sha256="abc",
        status=status, steps=3, errors=[], report=report_args() if status == "ok" else None,
    )  # fmt: skip


def _score(run_id, u, c, t, scorer="Oli"):
    return ReportScore(
        scored_at=datetime(2026, 10, 28, 13, tzinfo=UTC), scorer=scorer, run_id=run_id,
        match_id="m", backend="ollama", model="qwen", prompt_version="report_system v3",
        prompt_sha256="abc", usefulness=u, clarity=c, terms=t,
    )  # fmt: skip


def test_records_are_written_and_listed(tmp_path):
    write_record(_record("b", minute=2), tmp_path / "m", "# riport\n")
    write_record(_record("a", status="rejected", minute=1), tmp_path / "m", None)
    records = all_records(tmp_path)
    assert [r.run_id for r in records] == ["a", "b"]  # oldest first
    assert records[1].markdown == "b.md" and records[0].markdown is None
    assert (tmp_path / "m" / "report" / "runs" / "b.md").read_text(encoding="utf-8") == "# riport\n"


def test_scores_append_and_refuse_duplicates(tmp_path):
    path = tmp_path / "report_scores.jsonl"
    append_score(path, _score("r1", 4, 3, 5))
    with pytest.raises(ValueError, match="already scored"):
        append_score(path, _score("r1", 2, 2, 2))
    append_score(path, _score("r1", 2, 2, 2), replace=True)
    append_score(path, _score("r1", 5, 5, 5, scorer="Anna"))
    assert len(read_scores(path)) == 3  # append-only: the replaced line stays in the file


def test_summary_uses_latest_score_per_scorer_and_counts_rejections():
    records = [
        _record("r1"), _record("r2"), _record("r3", status="rejected"),
        _record("v2", version="report_system v2"),
    ]  # fmt: skip
    scores = [_score("r1", 4, 3, 5), _score("r1", 2, 2, 2), _score("r2", 3, 4, 4)]
    rows = {r.prompt_version: r for r in summarize(records, scores)}
    v3 = rows["report_system v3"]
    assert (v3.runs, v3.accepted, v3.scored) == (3, 2, 2)
    assert (v3.usefulness, v3.clarity, v3.terms) == (2.5, 3.0, 3.0)  # r1 replaced by 2/2/2
    assert (rows["report_system v2"].scored, rows["report_system v2"].usefulness) == (0, None)


def test_unscored_lists_accepted_runs_only():
    records = [_record("r1"), _record("r2"), _record("r3", status="rejected")]
    assert [r.run_id for r in unscored(records, [_score("r1", 4, 4, 4)], "Oli")] == ["r2"]
    assert [r.run_id for r in unscored(records, [_score("r1", 4, 4, 4)], "Anna")] == ["r1", "r2"]


def test_ask_score_repeats_until_valid():
    answers = iter(["", "7", "x", "4"])
    assert ask_score(lambda _: next(answers), "Érthetőség") == 4


def test_score_run_mixes_flags_and_questions():
    args = argparse.Namespace(scorer="Oli", usefulness=4, clarity=None, terms=5, comment=None)
    answers = iter(["3", "  rövid, de jó  "])
    score = score_run(_record("r1"), args, lambda _: next(answers))
    assert (score.usefulness, score.clarity, score.terms, score.comment) == (
        4,
        3,
        5,
        "rövid, de jó",
    )
    assert (score.run_id, score.prompt_version, score.scorer) == ("r1", "report_system v3", "Oli")


def test_render_uses_hungarian_team_labels_and_shows_sources():
    ctx = MatchContext(
        match_id="sb_1", source="statsbomb", duration_sec=5400,
        teams={"home": "France", "away": "Croatia"},
        teams_hu={"home": "Franciaország", "away": "Horvátország"}, has_video=False,
        attribution="Adatforrás: StatsBomb Open Data",
    )  # fmt: skip
    sample = Sample(
        duration_sec=5703.8,
        duration_min=95.1,
        n_events=975,
        events_per_type={"pass": 769, "ball_loss": 105},
        reliability="ok",
        thresholds=ReliabilityConfig(),
    )
    md = render_markdown(Report.model_validate(report_args()), ctx, sample, "ollama / qwen")
    note = "**Mintaméret:** 95,1 perc, 975 esemény (769 passz, 105 labdavesztés). "
    assert note + "Megbízhatóság: megfelelő" in md
    assert md.startswith("# Elemzői riport: Franciaország – Horvátország")
    assert "France" not in md
    assert (
        "- labdavesztés: 3 *(forrás: `get_metric(name=summary, team=home) -> summary.ball_losses`)*"
        in md
    )
    assert md.rstrip().endswith("*Adatforrás: StatsBomb Open Data*")
