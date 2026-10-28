"""Metrics stage: events (+ review corrections) and match context -> metrics.json."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from match_tagger.config import DefaultConfig, load_config
from match_tagger.evals.run import EvalConfig
from match_tagger.events.config import MatchesConfig
from match_tagger.events.corrections import merge, read_log
from match_tagger.events.schema import Event, EventsFile, Team
from match_tagger.metrics.config import MetricsConfig
from match_tagger.metrics.context import TEAMS, MatchContext, find_video
from match_tagger.metrics.frame import ZONE_LABELS_HU
from match_tagger.metrics.sample import Sample, TeamSummary, sample, team_summary
from match_tagger.metrics.tactical import (
    BallLosses,
    PassNetwork,
    Pressing,
    ShotMap,
    Unavailable,
    ball_losses_by_zone,
    pass_network,
    ppda_proxy,
    shot_map,
)
from match_tagger.pitch.config import PitchConfig
from match_tagger.video.frames import read_meta

METRIC_NAMES = ("summary", "ball_losses_by_zone", "pass_network", "ppda", "shot_map")


@dataclass
class MetricsStageConfig:
    """All configs the metrics stage reads."""

    paths: DefaultConfig
    metrics: MetricsConfig
    matches: MatchesConfig
    pitch: PitchConfig
    evals: EvalConfig

    @classmethod
    def load(cls, config_dir: Path) -> "MetricsStageConfig":
        """Load default, metrics, matches, pitch and evals YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            metrics=load_config("metrics", MetricsConfig, config_dir),
            matches=load_config("matches", MatchesConfig, config_dir),
            pitch=load_config("pitch", PitchConfig, config_dir),
            evals=load_config("evals", EvalConfig, config_dir),
        )


class MatchMetrics(BaseModel):
    """metrics.json: every metric for both teams, plus the sample block and zone labels."""

    match_id: str
    context: MatchContext
    sample: Sample
    summary: dict[Team, TeamSummary]
    ball_losses_by_zone: dict[Team, BallLosses]
    pass_network: dict[Team, PassNetwork | Unavailable]
    ppda: dict[Team, Pressing | Unavailable]
    shot_map: dict[Team, ShotMap | Unavailable]
    zone_labels_hu: dict[str, str]


def compute_metrics(events: Sequence[Event], ctx: MatchContext, cfg: MetricsConfig) -> MatchMetrics:
    """All metrics of a match (pure)."""
    node = "player" if ctx.source == "statsbomb" else "zone"
    return MatchMetrics(
        match_id=ctx.match_id,
        context=ctx,
        sample=sample(ctx, events, cfg.reliability),
        summary={t: team_summary(events, t) for t in TEAMS},
        ball_losses_by_zone={t: ball_losses_by_zone(events, ctx, t) for t in TEAMS},
        pass_network={t: pass_network(events, ctx, t, node, cfg.pass_network) for t in TEAMS},
        ppda={t: ppda_proxy(events, ctx, t, cfg.pressing) for t in TEAMS},
        shot_map={t: shot_map(events, ctx, t) for t in TEAMS},
        zone_labels_hu=ZONE_LABELS_HU if ctx.home_attacks is not None else {},
    )


def load_events(match_dir: Path, corrections_dir: Path) -> list[Event]:
    """events.json with the review corrections of the match replayed over it."""
    events = EventsFile.model_validate_json((match_dir / "events.json").read_text()).events
    return merge(events, read_log(corrections_dir / f"{match_dir.name}.jsonl")).events


def pipeline_context(match_id: str, video: Path, cfg: MetricsStageConfig) -> MatchContext:
    """Context of an own clip from configs/matches.yaml, configs/pitch.yaml and the video."""
    info = cfg.matches.matches.get(match_id)
    if info is None:
        raise ValueError(f"No entry for {match_id!r} in configs/matches.yaml")
    meta = read_meta(video)
    return MatchContext(
        match_id=match_id,
        source="pipeline",
        duration_sec=meta.n_frames / meta.fps,
        teams={
            t: f"{label} ({info.kits[t]} mez)"
            for t, label in (("home", "Hazai"), ("away", "Vendég"))
        },
        has_video=True,
        dims=cfg.pitch.dims_for(match_id),
        home_attacks=info.home_attacks,
    )


def load_context(match_id: str, video: Path | None, cfg: MetricsStageConfig) -> MatchContext:
    """match.json of a reference match, else the context of an own clip (rebuilt every run)."""
    match_dir = cfg.paths.paths.processed_dir / match_id
    path = match_dir / "match.json"
    if path.exists():
        ctx = MatchContext.model_validate_json(path.read_text(encoding="utf-8"))
        if ctx.source != "pipeline":
            return ctx
    video = video or find_video(cfg.paths.paths.raw_dir, match_id)
    if video is None:
        raise FileNotFoundError(f"No video for {match_id!r} under {cfg.paths.paths.raw_dir}")
    ctx = pipeline_context(match_id, video, cfg)
    path.write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    return ctx


def run_metrics(video: Path | None, match_id: str, cfg: MetricsStageConfig) -> MatchMetrics:
    """Run the stage and write metrics.json (and match.json for own clips)."""
    match_dir = cfg.paths.paths.processed_dir / match_id
    ctx = load_context(match_id, video, cfg)
    metrics = compute_metrics(load_events(match_dir, cfg.evals.corrections_dir), ctx, cfg.metrics)
    (match_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics
