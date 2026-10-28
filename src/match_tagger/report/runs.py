"""Per-run report records: data/processed/<match>/report/runs/<run_id>.json (+ .md if accepted).

A record keeps the report as plain JSON, so records written under an older ``Report`` schema
(e.g. before number sources existed) stay readable for scoring.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunRecord(BaseModel):
    """One report run: who wrote it with which prompt, the outcome, and the report itself."""

    run_id: str
    created_at: datetime
    match_id: str
    backend: str
    model: str
    prompt_version: str | None
    prompt_sha256: str
    status: str
    steps: int
    errors: list[str]
    report: dict[str, Any] | None = None
    markdown: str | None = None  # file name of the rendered report next to the record


def runs_dir(match_dir: Path) -> Path:
    """Directory of the run records of a match."""
    return match_dir / "report" / "runs"


def write_record(record: RunRecord, match_dir: Path, markdown: str | None) -> Path:
    """Write <run_id>.json (and <run_id>.md for an accepted report); return the JSON path."""
    out = runs_dir(match_dir)
    out.mkdir(parents=True, exist_ok=True)
    if markdown is not None:
        (out / f"{record.run_id}.md").write_text(markdown, encoding="utf-8")
        record = record.model_copy(update={"markdown": f"{record.run_id}.md"})
    path = out / f"{record.run_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def all_records(processed_dir: Path) -> list[RunRecord]:
    """Every run record under <processed_dir>/*/report/runs/, oldest first."""
    records = [
        RunRecord.model_validate_json(p.read_text(encoding="utf-8"))
        for p in processed_dir.glob("*/report/runs/*.json")
    ]
    return sorted(records, key=lambda r: (r.created_at, r.run_id))
