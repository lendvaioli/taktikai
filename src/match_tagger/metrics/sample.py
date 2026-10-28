"""Sample size and per-team counts: what the events of a match can and cannot support."""

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from match_tagger.events.schema import Event, Team
from match_tagger.metrics.config import ReliabilityConfig
from match_tagger.metrics.context import MatchContext

Reliability = Literal["very_low", "low", "ok"]


class Sample(BaseModel):
    """Duration, event counts and the reliability level derived from them."""

    duration_sec: float
    duration_min: float
    n_events: int
    events_per_type: dict[str, int]
    reliability: Reliability
    thresholds: ReliabilityConfig


class TeamSummary(BaseModel):
    """Event counts of one team."""

    team: Team
    events_per_type: dict[str, int]
    passes: int
    passes_complete: int
    pass_completion_share: float | None
    ball_losses: int


def reliability(duration_min: float, n_events: int, cfg: ReliabilityConfig) -> Reliability:
    """very_low if short or few events, low if shorter than low_max_min, else ok."""
    if duration_min < cfg.very_low_max_min or n_events < cfg.very_low_max_events:
        return "very_low"
    return "low" if duration_min < cfg.low_max_min else "ok"


def sample(ctx: MatchContext, events: Sequence[Event], cfg: ReliabilityConfig) -> Sample:
    """Sample block of a match."""
    minutes = round(ctx.duration_sec / 60, 1)
    return Sample(
        duration_sec=round(ctx.duration_sec, 1),
        duration_min=minutes,
        n_events=len(events),
        events_per_type=dict(sorted(Counter(e.type for e in events).items())),
        reliability=reliability(ctx.duration_sec / 60, len(events), cfg),
        thresholds=cfg,
    )


def team_summary(events: Sequence[Event], team: Team) -> TeamSummary:
    """Counts of one team; completion share only over passes with a known outcome."""
    own = [e for e in events if e.team == team]
    passes = [e for e in own if e.type == "pass"]
    known = [p for p in passes if p.outcome is not None]
    complete = sum(p.outcome == "complete" for p in known)
    return TeamSummary(
        team=team,
        events_per_type=dict(sorted(Counter(e.type for e in own).items())),
        passes=len(passes),
        passes_complete=complete,
        pass_completion_share=round(complete / len(known), 3) if known else None,
        ball_losses=sum(e.type == "ball_loss" for e in own),
    )
