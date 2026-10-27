"""Run summary of the pitch stage (pitch_summary.json)."""

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from match_tagger.pitch.geometry import PitchDims, outside_distance
from match_tagger.pitch.schema import PositionRecord


class PitchSummary(BaseModel):
    """Homography coverage, plausibility of player positions and timings of one run."""

    match_id: str
    device: str
    pitch_length_m: float
    pitch_width_m: float
    n_frames: int = Field(ge=0)
    h_quality_frame_share: dict[str, float]
    mean_confident_keypoints: float | None = None
    mean_inlier_error_m: float | None = Field(None, description="Over frames with an ok fit.")
    player_points: int = Field(0, description="Player rows with a position.")
    player_inside_share: float | None = Field(None, description="Inside 0..length x 0..width.")
    player_out_of_pitch_share: float | None = Field(None, description="Flagged out_of_pitch.")
    player_steps: int = Field(0, description="Consecutive-frame steps of one player track.")
    player_jump_share: float | None = Field(None, description="Steps faster than max_speed.")
    timings_sec: dict[str, float] = {}


def quality_shares(qualities: Sequence[str]) -> dict[str, float]:
    """Share of frames per h_quality value (all three keys always present)."""
    counts = Counter(qualities)
    n = len(qualities)
    return {q: (counts[q] / n if n else 0.0) for q in ("ok", "reused", "missing")}


def player_plausibility(
    records: Sequence[PositionRecord],
    dims: PitchDims,
    next_frame: dict[int, int],
    max_speed_mps: float,
) -> dict[str, float | int | None]:
    """Inside-pitch, out-of-pitch and jump shares of player positions.

    A step joins two rows of the same track on consecutive output frames (``next_frame``).
    """
    players = [r for r in records if r.cls == "player" and r.x_m is not None]
    xy = np.array([(r.x_m, r.y_m) for r in players]).reshape(-1, 2)
    by_track: dict[int, dict[int, PositionRecord]] = defaultdict(dict)
    for r in players:
        by_track[r.track_id][r.frame] = r
    speeds = [
        np.hypot(b.x_m - a.x_m, b.y_m - a.y_m) / (b.t_sec - a.t_sec)
        for rows in by_track.values()
        for frame, a in rows.items()
        if (b := rows.get(next_frame.get(frame, -1))) is not None
    ]
    n = len(players)
    return {
        "player_points": n,
        "player_inside_share": float(np.mean(outside_distance(xy, dims) == 0)) if n else None,
        "player_out_of_pitch_share": sum(r.out_of_pitch for r in players) / n if n else None,
        "player_steps": len(speeds),
        "player_jump_share": float(np.mean(np.array(speeds) > max_speed_mps)) if speeds else None,
    }
