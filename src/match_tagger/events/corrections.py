"""Review corrections: an append-only log per match and a deterministic merge over events.json.

events.json and the video are never modified; every change is a line in
evals/corrections/<match>.jsonl. ``merge`` replays the log in ``seq`` order (not by timestamp),
so the same events and log always give the same result.
"""

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from match_tagger.events.schema import Event

Action = Literal["accept", "delete", "change_type", "change_team", "add", "undo"]
Status = Literal["unreviewed", "accepted", "corrected", "added"]
_CHANGED_FIELD = {"change_type": "type", "change_team": "team"}


class CorrectionEntry(BaseModel):
    """One line of the corrections log."""

    seq: int = Field(ge=1)
    timestamp: datetime
    action: Action
    event_id: str | None = None
    before: Event | None = None
    after: Event | None = None
    target_seq: int | None = Field(None, ge=1, description="undo: the entry it reverts")

    @model_validator(mode="after")
    def _check_shape(self) -> "CorrectionEntry":
        if self.action == "undo":
            if self.target_seq is None or self.target_seq >= self.seq or self.event_id:
                raise ValueError("undo needs an earlier target_seq and no event")
            return self
        if self.target_seq is not None:
            raise ValueError("only undo has a target_seq")
        if self.action == "add":
            if self.before is not None or self.after is None or self.after.id != self.event_id:
                raise ValueError("add needs after (with id == event_id) and no before")
            return self
        if self.before is None or self.before.id != self.event_id:
            raise ValueError(f"{self.action} needs before with id == event_id")
        if self.action in ("accept", "delete") and self.after is not None:
            raise ValueError(f"{self.action} has no after")
        if self.action in _CHANGED_FIELD:
            self._check_change()
        return self

    def _check_change(self) -> None:
        field = _CHANGED_FIELD[self.action]
        if self.after is None or self.after.id != self.event_id:
            raise ValueError(f"{self.action} needs after with the same id")
        same = self.before.model_dump(exclude={field}) == self.after.model_dump(exclude={field})
        if not same or getattr(self.before, field) == getattr(self.after, field):
            raise ValueError(f"{self.action} may change only {field}, and must change it")


class Conflict(BaseModel):
    """A log entry that could not be applied."""

    seq: int
    reason: str


class MergeResult(BaseModel):
    """Events after replaying the log, their review status, and entries that did not apply."""

    events: list[Event]
    status: dict[str, Status]
    conflicts: list[Conflict]


def read_log(path: Path) -> list[CorrectionEntry]:
    """All entries of a log (empty if the file does not exist); seq must run 1, 2, 3, ..."""
    if not path.exists():
        return []
    entries = [
        CorrectionEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [e.seq for e in entries] != list(range(1, len(entries) + 1)):
        raise ValueError(f"{path}: seq must be 1..n in file order")
    return entries


def append_entry(path: Path, **fields: object) -> CorrectionEntry:
    """Validate and append one entry (next seq, current UTC time); never rewrites the file."""
    entry = CorrectionEntry(
        seq=len(read_log(path)) + 1, timestamp=datetime.now(UTC).replace(microsecond=0), **fields
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def undo_target(entries: Sequence[CorrectionEntry]) -> int | None:
    """Seq of the most recent entry that is not an undo and not undone yet (None if none)."""
    undone = {e.target_seq for e in entries if e.action == "undo"}
    candidates = [e.seq for e in entries if e.action != "undo" and e.seq not in undone]
    return candidates[-1] if candidates else None


def _undone(entries: Sequence[CorrectionEntry], conflicts: list[Conflict]) -> set[int]:
    by_seq = {e.seq: e for e in entries}
    undone: set[int] = set()
    for e in entries:
        if e.action != "undo":
            continue
        target = by_seq.get(e.target_seq)
        if target is None or target.action == "undo" or e.target_seq in undone:
            conflicts.append(Conflict(seq=e.seq, reason=f"cannot undo seq {e.target_seq}"))
        else:
            undone.add(e.target_seq)
    return undone


def _apply(e: CorrectionEntry, state: dict[str, Event], status: dict[str, Status]) -> str | None:
    """Apply one entry in place; return a conflict reason instead if it does not fit."""
    if e.action == "add":
        if e.event_id in state:
            return f"event {e.event_id} already exists"
        state[e.event_id], status[e.event_id] = e.after, "added"
        return None
    if state.get(e.event_id) != e.before:
        return f"event {e.event_id} is not in the state the entry expects"
    if e.action == "delete":
        del state[e.event_id]
        status.pop(e.event_id)
    elif e.action == "accept":
        status[e.event_id] = (
            "accepted" if status[e.event_id] == "unreviewed" else status[e.event_id]
        )
    else:
        state[e.event_id] = e.after
        status[e.event_id] = "added" if status[e.event_id] == "added" else "corrected"
    return None


def merge(events: Sequence[Event], entries: Sequence[CorrectionEntry]) -> MergeResult:
    """Replay the log over the predicted events; deterministic for the same inputs."""
    conflicts: list[Conflict] = []
    undone = _undone(entries, conflicts)
    state = {e.id: e for e in events}
    status: dict[str, Status] = {e.id: "unreviewed" for e in events}
    for e in sorted(entries, key=lambda x: x.seq):
        if e.action == "undo" or e.seq in undone:
            continue
        reason = _apply(e, state, status)
        if reason:
            conflicts.append(Conflict(seq=e.seq, reason=reason))
    ordered = sorted(state.values(), key=lambda ev: (ev.t_sec, ev.id))
    return MergeResult(
        events=ordered, status={ev.id: status[ev.id] for ev in ordered}, conflicts=conflicts
    )


def merged_json(result: MergeResult) -> str:
    """Canonical JSON of a merge result (for byte-level comparisons)."""
    return json.dumps(result.model_dump(mode="json"), sort_keys=True)
