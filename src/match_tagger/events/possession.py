"""Possession: the nearest player to the ground ball within a radius, held for a minimum time."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from match_tagger.events.schema import Team


@dataclass(frozen=True)
class FramePlayers:
    """Players with a known team and pitch position in one output frame."""

    track_ids: np.ndarray
    teams: list[Team]
    xy: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """The player controlling the ball in one frame."""

    track_id: int
    team: Team
    dist_m: float


@dataclass(frozen=True)
class Segment:
    """One continuous possession by one player (frame indices are inclusive)."""

    track_id: int
    team: Team
    start: int
    end: int
    mean_dist_m: float


def frame_candidate(ball: np.ndarray, players: FramePlayers, radius_m: float) -> Candidate | None:
    """Nearest player to the ball if within ``radius_m``; None without ball or player."""
    if np.isnan(ball[0]) or len(players.track_ids) == 0:
        return None
    dist = np.linalg.norm(players.xy - ball, axis=1)
    i = int(np.argmin(dist))
    if dist[i] > radius_m:
        return None
    return Candidate(int(players.track_ids[i]), players.teams[i], float(dist[i]))


def _runs(cands: Sequence[Candidate | None]) -> list[tuple[int, int]]:
    """Maximal runs [start, end] of consecutive frames with the same candidate track."""
    runs, start = [], None
    for i, c in enumerate(cands):
        prev = cands[i - 1] if i > 0 else None
        if c is not None and (prev is None or prev.track_id != c.track_id):
            start = i
        if c is not None and (
            i + 1 == len(cands) or cands[i + 1] is None or cands[i + 1].track_id != c.track_id
        ):
            runs.append((start, i))
    return runs


def segments(
    times: np.ndarray, cands: Sequence[Candidate | None], min_hold_sec: float
) -> list[Segment]:
    """Debounced possession segments.

    Runs shorter than ``min_hold_sec`` (run duration = frames x frame spacing) are dropped;
    neighbouring kept runs of the same track are merged, so a short flicker to another player
    does not split a possession.
    """
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    kept = [(a, b) for a, b in _runs(cands) if (b - a + 1) * dt >= min_hold_sec - 1e-9]
    merged: list[list[int]] = []
    for a, b in kept:
        if merged and cands[merged[-1][0]].track_id == cands[a].track_id:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        dists = [c.dist_m for c in cands[a : b + 1] if c is not None]
        first = cands[a]
        out.append(Segment(first.track_id, first.team, a, b, float(np.mean(dists))))
    return out
