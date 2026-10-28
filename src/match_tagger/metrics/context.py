"""Match context of the metrics: source, duration, teams, pitch size and direction of play."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from match_tagger.events.config import Side
from match_tagger.events.schema import Team
from match_tagger.pitch.geometry import PitchDims

TEAMS: tuple[Team, Team] = ("home", "away")


class MatchContext(BaseModel):
    """data/processed/<match>/match.json.

    ``home_attacks`` is the goal the home team attacks in the absolute pitch frame (left =
    towards x = 0), for the whole match or per period; None if unknown.
    """

    match_id: str
    source: Literal["pipeline", "statsbomb"]
    duration_sec: float
    teams: dict[Team, str]
    has_video: bool
    dims: PitchDims = PitchDims()
    home_attacks: Side | dict[int, Side] | None = None
    attribution: str | None = None


def other(team: Team) -> Team:
    """The opponent of a team."""
    return "away" if team == "home" else "home"


def attack_sign(ctx: MatchContext, team: Team, period: int | None) -> int | None:
    """+1 if ``team`` attacks towards x = length in ``period``, -1 if towards x = 0, else None."""
    side = ctx.home_attacks
    if isinstance(side, dict):
        side = side.get(period) if period is not None else None
    if side is None:
        return None
    home_sign = 1 if side == "right" else -1
    return home_sign if team == "home" else -home_sign


def find_video(raw_dir: Path, match_id: str) -> Path | None:
    """The source video of a match: <raw_dir>/**/<match_id>.mp4 (first in sorted order)."""
    return next(iter(sorted(raw_dir.rglob(f"{match_id}.mp4"))), None)
