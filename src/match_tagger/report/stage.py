"""Report stage: metrics.json + reviewed events + clips -> a run record, report.json and .md.

Every run writes report/runs/<run_id>.json (and .md if accepted) and appends to
report/runs.jsonl; report.json / report.md are copies of the latest accepted report.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from match_tagger.clips.stage import load_index
from match_tagger.config import DefaultConfig, load_config
from match_tagger.evals.run import EvalConfig
from match_tagger.metrics.context import MatchContext, team_labels_hu
from match_tagger.metrics.stage import MatchMetrics, load_events
from match_tagger.report.agent import AgentResult, AgentSettings, run_agent
from match_tagger.report.backends import LLMBackend, make_backend
from match_tagger.report.config import ReportConfig
from match_tagger.report.log import RunLog
from match_tagger.report.render import render_markdown
from match_tagger.report.runs import RunRecord, write_record
from match_tagger.report.tools import ToolBox


@dataclass
class ReportStageConfig:
    """All configs the report stage reads."""

    paths: DefaultConfig
    report: ReportConfig
    evals: EvalConfig

    @classmethod
    def load(cls, config_dir: Path) -> "ReportStageConfig":
        """Load default, report and evals YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            report=load_config("report", ReportConfig, config_dir),
            evals=load_config("evals", EvalConfig, config_dir),
        )


class ReportRun(BaseModel):
    """Summary printed by the CLI."""

    match_id: str
    run_id: str
    backend: str
    model: str
    status: str
    steps: int
    errors: list[str]
    output: str


def prompt_version(prompt: str) -> str | None:
    """The label of the "<!-- version: ... -->" comment of a prompt file, if any."""
    match = re.search(r"<!--\s*version:\s*(.+?)\s*-->", prompt)
    return match.group(1) if match else None


def user_message(ctx: MatchContext) -> str:
    """The task message: match, teams, whether clips exist (numbers only via tools)."""
    video = "yes: cite clip ids from get_clips" if ctx.has_video else "no: leave clip_ids empty"
    teams = team_labels_hu(ctx)
    return (
        f"Match id: {ctx.match_id}\nHome team: {teams['home']}\n"
        f"Away team: {teams['away']}\nVideo clips available: {video}\n"
        "Write the coaching report for the home team's staff. Start with get_metric summary."
    )


def foreign_team_names(ctx: MatchContext) -> list[str]:
    """Team names that must not appear in the report because they have a Hungarian label."""
    labels = team_labels_hu(ctx)
    return [name for t, name in ctx.teams.items() if name != labels[t]]


def _write_outputs(match_dir: Path, result: AgentResult, ctx: MatchContext, header: dict) -> Path:
    """Run record (+ markdown); copies of an accepted report as report.json / report.md."""
    by = f"{header['backend']} / {header['model']}"
    markdown = render_markdown(result.report, ctx, by) if result.report else None
    record = RunRecord(
        **header,
        created_at=datetime.now(UTC).replace(microsecond=0),
        status=result.status,
        steps=result.steps,
        errors=result.errors,
        report=result.report.model_dump(mode="json") if result.report else None,
    )
    path = write_record(record, match_dir, markdown)
    if result.report is not None:
        out = match_dir / "report"
        (out / "report.json").write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
        (out / "report.md").write_text(
            markdown.replace("../../clips/", "../clips/"), encoding="utf-8"
        )
    return path


def run_report(
    video: Path | None, match_id: str, cfg: ReportStageConfig, backend: LLMBackend | None = None
) -> ReportRun:
    """Run the agent once for a match; every run is appended to report/runs.jsonl."""
    match_dir = cfg.paths.paths.processed_dir / match_id
    metrics = MatchMetrics.model_validate_json((match_dir / "metrics.json").read_text("utf-8"))
    backend = backend or make_backend(cfg.report)
    prompt = cfg.report.prompt_path.read_text(encoding="utf-8")
    run_id = uuid.uuid4().hex[:12]
    header = {
        "run_id": run_id,
        "match_id": match_id,
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": prompt_version(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    }
    out_dir = match_dir / "report"
    log = RunLog(out_dir / "runs.jsonl", run_id, {k: v for k, v in header.items() if k != "run_id"})
    box = ToolBox(
        metrics,
        load_events(match_dir, cfg.evals.corrections_dir),
        load_index(match_dir),
        cfg.report.max_events,
    )
    settings = AgentSettings(
        match_id=match_id,
        has_video=metrics.context.has_video,
        max_steps=cfg.report.max_steps,
        retries=cfg.report.retries,
        tolerance=cfg.report.number_tolerance,
        semantics=cfg.report.semantics,
        foreign_names=foreign_team_names(metrics.context),
    )
    result = run_agent(backend, prompt, user_message(metrics.context), box, settings, log)
    output = _write_outputs(match_dir, result, metrics.context, header)
    return ReportRun(
        match_id=match_id, run_id=run_id, backend=backend.name, model=backend.model,
        status=result.status, steps=result.steps, errors=result.errors, output=str(output),
    )  # fmt: skip
