"""Generated README tables: event results, report-agent acceptance, models and licences.

Results come from the local (private) eval results and report run records, but only aggregates
over a whole golden set or over all runs of a prompt version are written; per-clip results are
never read into the tables. Each table replaces the text between its marker comments.
"""

import re
from collections.abc import Sequence
from pathlib import Path

from match_tagger.evals.run import EvalRun
from match_tagger.report.runs import RunRecord
from match_tagger.report.score import ReportScore, summarize

TYPE_LABELS = {"pass": "pass", "ball_loss": "ball loss", "shot": "shot", "set_piece": "set piece"}


def _f(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def events_table(run: EvalRun) -> str:
    """Per golden set and event type: golden events, P, R, F1 (aggregates only)."""
    lines = [
        "| golden set | event type | golden events | precision | recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for s in run.sets:
        for kind, row in s.per_type.items():
            golden = row.tp + row.fn
            if row.tp + row.fp == 0:
                continue  # not predicted by this version (e.g. shots in events v1)
            label = TYPE_LABELS.get(kind, kind)
            lines.append(
                f"| {s.set_id} ({s.n_clips_evaluated} clips) | {label} | {golden} "
                f"| {_f(row.precision)} | {_f(row.recall)} | {_f(row.f1)} |"
            )
    lines.append(
        f"\nRun of {run.created_at:%Y-%m-%d} at commit `{run.git_commit}`, matching tolerance "
        f"{run.tolerance_sec:g} s. Thresholds were tuned leave-one-clip-out; for passes the "
        "held-out estimate equals this result (docs/EXPERIMENTS.md)."
    )
    return "\n".join(lines)


def report_table(records: Sequence[RunRecord], scores: Sequence[ReportScore]) -> str:
    """Per backend, model and prompt version: runs, accepted, mean rubric scores."""
    lines = [
        "| prompt version | model | runs | accepted | rubric (useful / clear / terms, 1-5) |",
        "|---|---|---|---|---|",
    ]
    for row in summarize(records, scores):
        rubric = (
            f"{row.usefulness:.1f} / {row.clarity:.1f} / {row.terms:.1f} ({row.scored} scored)"
            if row.scored
            else "not scored yet"
        )
        model = row.model.split("/")[-1]
        lines.append(f"| {row.prompt_version} | {model} | {row.runs} | {row.accepted} | {rubric} |")
    return "\n".join(lines)


def models_table(models_md: str) -> str:
    """The first table of docs/MODELS.md, without the source column.

    Licences are cut after their first sentence; the full text stays in docs/MODELS.md.
    """
    lines = models_md.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[first + 2 :]:  # skip header and separator
        if not line.startswith("|"):
            break  # the first table ends at the first non-table line
        rows.append([c.strip() for c in line.strip("|").split("|")])
    out = ["| model / dataset | licence | used in |", "|---|---|---|"]
    for cells in rows:
        end = re.search(r"\.(\*\*)?\s", cells[2])  # first sentence end, also inside bold
        licence = cells[2][: end.end() - 1] if end else cells[2]
        out.append(f"| {cells[0]} | {licence} | {cells[3]} |")
    return "\n".join(out)


def replace_block(text: str, name: str, content: str) -> str:
    """Replace the text between <!-- name:start --> and <!-- name:end -->."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"README has no {start} ... {end} block")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", text)


def latest_run(results_dir: Path) -> EvalRun:
    """The newest evaluation run (file names are UTC timestamps)."""
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no evaluation runs in {results_dir}")
    return EvalRun.model_validate_json(files[-1].read_text(encoding="utf-8"))
