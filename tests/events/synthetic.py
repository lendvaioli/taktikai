"""Synthetic clips at 12.5 fps (dt = 0.08 s) for the event-rule tests."""

import numpy as np
import pandas as pd

from match_tagger.events.possession import FramePlayers
from match_tagger.events.stage import ClipData
from match_tagger.events.teams import TeamMapping

DT = 0.08


def make_clip(n_frames, players, ball, size_ratio=None, h_quality=None):
    """players: {track_id: (team, frames iterable, (x, y))}; ball: {frame: (x, y)}."""
    times = np.round(np.arange(n_frames) * DT, 4)
    per_frame = []
    for f in range(n_frames):
        rows = [(tid, team, xy) for tid, (team, frames, xy) in players.items() if f in frames]
        per_frame.append(
            FramePlayers(
                np.array([r[0] for r in rows], dtype=int),
                [r[1] for r in rows],
                np.array([r[2] for r in rows], dtype=float).reshape(-1, 2),
            )
        )
    ball_xy = np.full((n_frames, 2), np.nan)
    for f, xy in ball.items():
        ball_xy[f] = xy
    ratio = np.full(n_frames, np.nan) if size_ratio is None else np.asarray(size_ratio, float)
    columns = ["track_id", "team", "first_t", "last_t", "first_x", "first_y", "last_x", "last_y"]
    tracks = pd.DataFrame(
        [
            (tid, team, min(fr) * DT, max(fr) * DT, xy[0], xy[1], xy[0], xy[1])
            for tid, (team, fr, xy) in players.items()
        ],
        columns=columns,
    ).set_index("track_id")
    mapping = TeamMapping({0: "home", 1: "away"}, {}, 0.0)
    return ClipData(
        "c",
        times,
        np.arange(n_frames) * 2,
        h_quality or ["ok"] * n_frames,
        ball_xy,
        ratio,
        per_frame,
        mapping,
        tracks,
    )


def hold(frames, xy):
    """Ball held at one position over a range of frames."""
    return {f: xy for f in frames}
