"""Error analysis of event detection: ball observability, miss causes, false-pass causes."""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from match_tagger.events.config import DiagnosticsConfig
from match_tagger.events.possession import Segment
from match_tagger.events.schema import Event
from match_tagger.events.stage import ClipData, Detection

MISS_CAUSES = ("ball_gap", "no_player_near", "homography", "team_mapping", "timing", "other")


def near(times: np.ndarray, t: float, window: float) -> np.ndarray:
    """Mask of frames within +-window of t."""
    return np.abs(times - t) <= window + 1e-9


def observable(times: np.ndarray, known: np.ndarray, t: float, window: float) -> bool:
    """True if a ball position is known in any frame within +-window of t."""
    return bool(known[near(times, t, window)].any())


def miss_cause(
    golden: Event,
    clip: ClipData,
    det: Detection,
    predictions: Sequence[Event],
    mapping_correct: bool | None,
    tolerance_sec: float,
    cfg: DiagnosticsConfig,
) -> str:
    """First applicable cause, in the order of ``MISS_CAUSES``, for one missed golden event."""
    w = near(clip.times, golden.t_sec, cfg.observable_window_sec)
    if not (~np.isnan(det.ball_xy[w, 0])).any():
        return "ball_gap"
    if not det.possessed[w].any():
        return "no_player_near"
    if any(q != "ok" for q, inside in zip(clip.h_quality, w, strict=True) if inside):
        return "homography"
    if mapping_correct is False:
        return "team_mapping"
    if any(
        p.type == golden.type
        and p.team == golden.team
        and tolerance_sec < abs(p.t_sec - golden.t_sec) <= cfg.timing_near_sec
        for p in predictions
    ):
        return "timing"
    return "other"


def pass_pair(event: Event, segs: Sequence[Segment], times: np.ndarray) -> tuple[Segment, Segment]:
    """The possession segments A -> B that produced a pass event (A ends at the pass time)."""
    for a, b in zip(segs, segs[1:], strict=False):
        if abs(round(float(times[a.end]), 2) - event.t_sec) < 1e-6:
            return a, b
    raise ValueError(f"No segment pair for event {event.id}")


def is_track_break(a: Segment, b: Segment, clip: ClipData, cfg: DiagnosticsConfig) -> bool:
    """True if B's track starts where and shortly after A's track ended (same player re-ided)."""
    if (
        a.team != b.team
        or a.track_id not in clip.tracks.index
        or b.track_id not in clip.tracks.index
    ):
        return False
    ta, tb = clip.tracks.loc[a.track_id], clip.tracks.loc[b.track_id]
    gap = tb.first_t - ta.last_t
    dist = float(np.hypot(tb.first_x - ta.last_x, tb.first_y - ta.last_y))
    return 0 <= gap <= cfg.track_break_max_gap_sec and dist <= cfg.track_break_max_dist_m


def false_pass_causes(
    false_passes: Sequence[Event], clip: ClipData, det: Detection, cfg: DiagnosticsConfig
) -> Counter[str]:
    """Count false-positive passes caused by track breaks vs other causes."""
    causes: Counter[str] = Counter()
    for event in false_passes:
        a, b = pass_pair(event, det.segments, clip.times)
        causes["track_break" if is_track_break(a, b, clip, cfg) else "other"] += 1
    return causes
