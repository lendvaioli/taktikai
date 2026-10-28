"""Review-app logic without UI: rows with status and evaluation labels, filters, export."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from match_tagger.evals.golden import GoldenClip, list_golden_sets, load_golden_set
from match_tagger.evals.matching import match_events
from match_tagger.events.corrections import CorrectionEntry, MergeResult, Status, undo_target
from match_tagger.events.schema import EVENT_TYPES, Event, Team
from match_tagger.events.zones import zone
from match_tagger.pitch.geometry import PitchDims

EvalLabel = Literal["TP", "FP", "FN"]


class ReviewRow(BaseModel):
    """One row of the review list: a (merged) predicted event or a missed golden event."""

    id: str
    t_sec: float
    type: str
    team: Team
    zone: int | None
    confidence: float | None
    source: Literal["prediction", "golden"]
    status: Status | None  # None for golden-only (FN) rows
    eval: EvalLabel | None  # None without a golden clip


def find_golden_clip(golden_dir: Path, clip_id: str) -> GoldenClip | None:
    """The complete golden clip with this id in any golden set, if there is one."""
    for set_id in list_golden_sets(golden_dir):
        clip = load_golden_set(golden_dir, set_id).clip(clip_id)
        if clip is not None and clip.complete:
            return clip
    return None


def eval_labels(
    events: Sequence[Event], golden: Sequence[Event], tolerance_sec: float
) -> tuple[dict[str, EvalLabel], list[Event]]:
    """TP/FP per predicted event id and the missed golden events, matched per type."""
    labels: dict[str, EvalLabel] = {}
    missed: list[Event] = []
    for event_type in EVENT_TYPES:
        preds = [e for e in events if e.type == event_type]
        gold = [g for g in golden if g.type == event_type]
        result = match_events(preds, gold, tolerance_sec)
        labels.update({p: "TP" for p, _, _ in result.pairs})
        labels.update({p: "FP" for p in result.unmatched_predictions})
        missed += [g for g in gold if g.id in set(result.unmatched_golden)]
    return labels, missed


def _row(
    e: Event, dims: PitchDims, source: str, status: Status | None, label: EvalLabel | None
) -> ReviewRow:
    return ReviewRow(
        id=e.id,
        t_sec=e.t_sec,
        type=e.type,
        team=e.team,
        zone=zone(e.x_m, e.y_m, dims),
        confidence=e.confidence,
        source=source,
        status=status,
        eval=label,
    )


def review_rows(
    merged: MergeResult, golden: GoldenClip | None, dims: PitchDims, tolerance_sec: float
) -> list[ReviewRow]:
    """Merged events (and, with a golden clip, TP/FP labels plus FN rows), sorted by time."""
    if golden is None:
        return [_row(e, dims, "prediction", merged.status[e.id], None) for e in merged.events]
    labels, missed = eval_labels(merged.events, golden.events, tolerance_sec)
    rows = [_row(e, dims, "prediction", merged.status[e.id], labels[e.id]) for e in merged.events]
    rows += [_row(g, dims, "golden", None, "FN") for g in missed]
    return sorted(rows, key=lambda r: (r.t_sec, r.source, r.id))


def filter_rows(
    rows: Iterable[ReviewRow],
    types: set[str] | None = None,
    teams: set[str] | None = None,
    zones: set[int | None] | None = None,
    evals: set[str] | None = None,
) -> list[ReviewRow]:
    """Keep rows matching every given filter (None = no filter on that field)."""
    return [
        r
        for r in rows
        if (types is None or r.type in types)
        and (teams is None or r.team in teams)
        and (zones is None or r.zone in zones)
        and (evals is None or r.eval in evals)
    ]


def golden_candidate(
    merged: MergeResult,
    clip_id: str,
    video: str,
    fps: float,
    duration_sec: float,
    kits: dict[Team, str],
) -> GoldenClip:
    """The reviewed events as a golden clip candidate (not complete; you promote it manually)."""
    events = [e.model_copy(update={"confidence": None}) for e in merged.events]
    return GoldenClip(
        clip_id=clip_id,
        video=video,
        fps=fps,
        duration_sec=duration_sec,
        kits=kits,
        complete=False,
        tagged_by="review",
        events=events,
    )


def write_candidate(candidate: GoldenClip, candidates_dir: Path) -> Path:
    """Write evals/candidates/<clip_id>.json (never under evals/golden/)."""
    candidates_dir.mkdir(parents=True, exist_ok=True)
    path = candidates_dir / f"{candidate.clip_id}.json"
    path.write_text(candidate.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def correction_for(
    value: dict,
    merged: MergeResult,
    match_id: str,
    entries: Sequence[CorrectionEntry],
    new_id: Callable[[], str],
) -> dict | None:
    """Fields of the log entry for one component action; None if it no longer applies.

    ``value`` is the component message: ``{"action": ..., "id"/"type"/"team"/"t_sec": ...}``.
    An action on an event that is not in the merged state (e.g. a stale double click) and an
    undo with nothing left to undo give None.
    """
    action = value["action"]
    if action == "undo":
        target = undo_target(entries)
        return None if target is None else {"action": "undo", "target_seq": target}
    if action == "add":
        event = Event(
            id=f"{match_id}-a{new_id()}",
            t_sec=round(float(value["t_sec"]), 2),
            type=value["type"],
            team=value["team"],
        )
        return {"action": "add", "event_id": event.id, "after": event}
    before = {e.id: e for e in merged.events}.get(value.get("id"))
    if before is None:
        return None
    after = None
    if action in ("change_type", "change_team"):
        field = "type" if action == "change_type" else "team"
        after = before.model_copy(update={field: value[field]})
    return {"action": action, "event_id": before.id, "before": before, "after": after}
