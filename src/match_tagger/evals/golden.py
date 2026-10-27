"""Golden sets: manually tagged events, one JSON file per source under evals/golden/.

Only the tagger app (apps/tagger.py) may call ``save_golden_set``; evaluation code only reads.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from match_tagger.events.schema import Event, Team


class GoldenClip(BaseModel):
    """One tagged clip. Event times are relative to the clip start."""

    clip_id: str = Field(min_length=1)
    video: str = Field(description="Repo-relative path of the clip, e.g. data/raw/dfl/x.mp4.")
    fps: float = Field(gt=0)
    duration_sec: float = Field(gt=0)
    kits: dict[Team, str] = Field(default_factory=dict, description="Kit description per team.")
    complete: bool = Field(False, description="Only complete clips are evaluated.")
    tagged_by: str | None = None
    updated_at: datetime | None = None
    events: list[Event] = Field(default_factory=list)


class GoldenSet(BaseModel):
    """All golden clips from one source (e.g. the DFL sample clips)."""

    set_id: str = Field(min_length=1)
    source: str = ""
    description: str = ""
    clips: list[GoldenClip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "GoldenSet":
        clip_ids = [c.clip_id for c in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip_id values must be unique within a golden set")
        event_ids = [e.id for c in self.clips for e in c.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event ids must be unique within a golden set")
        return self

    def clip(self, clip_id: str) -> GoldenClip | None:
        """Return the clip with this id, if any."""
        return next((c for c in self.clips if c.clip_id == clip_id), None)


def golden_path(golden_dir: Path, set_id: str) -> Path:
    """Path of a golden set file."""
    return golden_dir / f"{set_id}.json"


def list_golden_sets(golden_dir: Path) -> list[str]:
    """Return the ids of all golden sets in ``golden_dir``, sorted."""
    return sorted(p.stem for p in golden_dir.glob("*.json")) if golden_dir.exists() else []


def load_golden_set(golden_dir: Path, set_id: str) -> GoldenSet:
    """Read and validate one golden set."""
    text = golden_path(golden_dir, set_id).read_text(encoding="utf-8")
    return GoldenSet.model_validate_json(text)


def save_golden_set(golden: GoldenSet, golden_dir: Path) -> Path:
    """Atomically write a golden set (tagger app only). Events are stored sorted by time."""
    for clip in golden.clips:
        clip.events.sort(key=lambda e: (e.t_sec, e.id))
    validated = GoldenSet.model_validate(golden.model_dump())
    path = golden_path(golden_dir, golden.set_id)
    golden_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=golden_dir, prefix=f".{golden.set_id}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(validated.model_dump_json(indent=2) + "\n")
    os.replace(tmp, path)
    return path
