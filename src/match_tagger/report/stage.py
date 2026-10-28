"""Report stage: metrics.json + reviewed events + clips -> report.json and report.md."""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from match_tagger.clips.stage import load_index
from match_tagger.config import DefaultConfig, load_config
from match_tagger.evals.run import EvalConfig
from match_tagger.metrics.context import MatchContext
from match_tagger.metrics.stage import MatchMetrics, load_events
from match_tagger.report.agent import AgentResult, AgentSettings, run_agent
from match_tagger.report.backends import LLMBackend, make_backend
from match_tagger.report.config import ReportConfig
from match_tagger.report.log import RunLog
from match_tagger.report.render import render_markdown
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


def user_message(ctx: MatchContext) -> str:
    """The task message: match, teams, whether clips exist (numbers only via tools)."""
    video = "yes: cite clip ids from get_clips" if ctx.has_video else "no: leave clip_ids empty"
    return (
        f"Match id: {ctx.match_id}\nHome team: {ctx.teams['home']}\n"
        f"Away team: {ctx.teams['away']}\nVideo clips available: {video}\n"
        "Write the coaching report for the home team's staff. Start with get_metric summary."
    )


def _write_outputs(out_dir: Path, result: AgentResult, ctx: MatchContext, by: str) -> Path:
    if result.report is None:
        path = out_dir / "report_failed.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path
    (out_dir / "report.json").write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
    path = out_dir / "report.md"
    path.write_text(render_markdown(result.report, ctx, by), encoding="utf-8")
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
        "match_id": match_id,
        "backend": backend.name,
        "model": backend.model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    }
    out_dir = match_dir / "report"
    log = RunLog(out_dir / "runs.jsonl", run_id, header)
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
    )
    result = run_agent(backend, prompt, user_message(metrics.context), box, settings, log)
    output = _write_outputs(out_dir, result, metrics.context, f"{backend.name} / {backend.model}")
    return ReportRun(
        match_id=match_id, run_id=run_id, backend=backend.name, model=backend.model,
        status=result.status, steps=result.steps, errors=result.errors, output=str(output),
    )  # fmt: skip
