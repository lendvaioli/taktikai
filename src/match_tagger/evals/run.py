"""Evaluation runs: predictions vs golden sets, per golden set, per type and overall."""

import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from match_tagger.evals.golden import GoldenClip, GoldenSet
from match_tagger.evals.matching import match_events
from match_tagger.evals.metrics import Counts, MetricRow
from match_tagger.events.schema import EVENT_TYPES, Event, EventsFile

PredictionLoader = Callable[[str], list[Event] | None]


class EvalConfig(BaseModel):
    """Contents of configs/evals.yaml."""

    tolerance_sec: float = Field(2.0, gt=0)
    golden_dir: Path = Path("evals/golden")
    results_dir: Path = Path("evals/results")
    predictions: str = Field(
        "data/processed/{clip_id}/events.json", description="Path pattern of predictions."
    )


class ClipResult(BaseModel):
    """Counts of one clip, per event type."""

    clip_id: str
    predictions_found: bool
    counts: dict[str, Counts]


class SetResult(BaseModel):
    """Metrics of one golden set over its complete clips."""

    set_id: str
    source: str
    n_clips_evaluated: int
    skipped_incomplete: list[str]
    missing_predictions: list[str]
    per_type: dict[str, MetricRow]
    overall: MetricRow
    clips: list[ClipResult]


class EvalRun(BaseModel):
    """One evaluation run, written to evals/results/<timestamp>.json."""

    created_at: datetime
    git_commit: str | None
    tolerance_sec: float
    sets: list[SetResult]


def file_prediction_loader(pattern: str) -> PredictionLoader:
    """Load events.json for a clip; None if the file does not exist."""

    def load(clip_id: str) -> list[Event] | None:
        path = Path(pattern.format(clip_id=clip_id))
        if not path.exists():
            return None
        return EventsFile.model_validate_json(path.read_text(encoding="utf-8")).events

    return load


def evaluate_clip(
    clip: GoldenClip, predictions: Sequence[Event], tolerance_sec: float
) -> dict[str, Counts]:
    """Match one clip type by type and count TP / FP / FN."""
    counts = {}
    for event_type in EVENT_TYPES:
        preds = [p for p in predictions if p.type == event_type]
        gold = [g for g in clip.events if g.type == event_type]
        result = match_events(preds, gold, tolerance_sec)
        tp = len(result.pairs)
        counts[event_type] = Counts(tp=tp, fp=len(preds) - tp, fn=len(gold) - tp)
    return counts


def evaluate_set(golden: GoldenSet, load: PredictionLoader, tolerance_sec: float) -> SetResult:
    """Evaluate all complete clips of a golden set; incomplete clips are skipped."""
    complete = [c for c in golden.clips if c.complete]
    clips, missing = [], []
    for clip in complete:
        predictions = load(clip.clip_id)
        if predictions is None:
            missing.append(clip.clip_id)
        counts = evaluate_clip(clip, predictions or [], tolerance_sec)
        clips.append(ClipResult(clip_id=clip.clip_id, predictions_found=predictions is not None,
                                counts=counts))  # fmt: skip
    totals = {t: sum((c.counts[t] for c in clips), Counts()) for t in EVENT_TYPES}
    return SetResult(
        set_id=golden.set_id, source=golden.source, n_clips_evaluated=len(complete),
        skipped_incomplete=[c.clip_id for c in golden.clips if not c.complete],
        missing_predictions=missing,
        per_type={t: MetricRow.from_counts(c) for t, c in totals.items()},
        overall=MetricRow.from_counts(sum(totals.values(), Counts())), clips=clips,
    )  # fmt: skip


def git_commit() -> str | None:
    """Short commit hash of HEAD, with ``-dirty`` if the work tree has changes.

    Changes under evals/results/ are ignored: they are outputs of runs, not inputs, and a fresh
    tuning report would otherwise mark the next evaluation as dirty.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":!evals/results"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def new_run(sets: list[SetResult], tolerance_sec: float) -> EvalRun:
    """Create a run stamped with the current UTC time and git commit."""
    now = datetime.now(UTC).replace(microsecond=0)
    return EvalRun(created_at=now, git_commit=git_commit(), tolerance_sec=tolerance_sec, sets=sets)


def write_run(run: EvalRun, results_dir: Path) -> Path:
    """Write the run as evals/results/<YYYYMMDDTHHMMSSZ>.json and return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{run.created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
