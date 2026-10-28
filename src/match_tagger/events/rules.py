"""Possession transitions -> pass and ball_loss events, each with a rule-based confidence."""

from collections.abc import Sequence

import numpy as np

from match_tagger.events.config import PossessionConfig, RulesConfig
from match_tagger.events.possession import Segment
from match_tagger.events.schema import Event


def confidence(
    a: Segment,
    b: Segment,
    times: np.ndarray,
    gap_sec: float,
    poss: PossessionConfig,
    rules: RulesConfig,
) -> float:
    """Mean of three scores in [0, 1]: hold length, transition shortness, ball proximity."""
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    ref = max(2 * poss.min_hold_sec, dt)
    held = min((a.end - a.start + 1) * dt, (b.end - b.start + 1) * dt) / ref
    hold_score = min(1.0, held)
    gap_score = max(0.0, 1.0 - gap_sec / rules.max_transition_sec)
    near_score = max(0.0, 1.0 - (a.mean_dist_m + b.mean_dist_m) / 2 / poss.control_radius_m)
    return round((hold_score + gap_score + near_score) / 3, 3)


def events_from_segments(
    segs: Sequence[Segment],
    times: np.ndarray,
    ball_xy: np.ndarray,
    clip_id: str,
    poss: PossessionConfig,
    rules: RulesConfig,
) -> list[Event]:
    """Turn consecutive possession segments A -> B into events.

    Same team: a pass at A's last frame if the ball travelled >= pass_min_ball_travel_m (a
    shorter "pass" is most likely a broken track of one player). Other team: a ball_loss at B's
    first frame; if the ball travelled >= intercept_min_dist_m it was an intercepted pass, so a
    pass at A's last frame is added too. Transitions longer than max_transition_sec are ignored.
    """
    events: list[Event] = []
    for a, b in zip(segs, segs[1:], strict=False):
        gap = float(times[b.start] - times[a.end])
        if gap > rules.max_transition_sec + 1e-9:
            continue
        travel = float(np.linalg.norm(ball_xy[b.start] - ball_xy[a.end]))
        conf = confidence(a, b, times, gap, poss, rules)
        is_pass = (a.team == b.team and travel >= rules.pass_min_ball_travel_m) or (
            a.team != b.team and travel >= rules.intercept_min_dist_m
        )
        if is_pass:
            events.append(_event(clip_id, len(events), "pass", a, a.end, times, ball_xy, conf))
        if a.team != b.team:
            events.append(
                _event(clip_id, len(events), "ball_loss", a, b.start, times, ball_xy, conf)
            )
    return events


def _event(
    clip_id: str,
    n: int,
    type_: str,
    actor: Segment,
    frame: int,
    times: np.ndarray,
    ball_xy: np.ndarray,
    conf: float,
) -> Event:
    x, y = ball_xy[frame]
    return Event(
        id=f"{clip_id}-e{n:04d}",
        t_sec=round(float(times[frame]), 2),
        type=type_,
        team=actor.team,
        x_m=round(float(x), 2),
        y_m=round(float(y), 2),
        confidence=conf,
    )
