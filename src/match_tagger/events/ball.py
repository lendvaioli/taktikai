"""Ball track on the pitch: airborne flag and short-gap interpolation.

The pitch stage projects the ball as if it were on the ground. A ball in the air projects to a
point further from the camera, where a real ground ball would look smaller; so a ball whose
pixel size is too large for its projected position is likely airborne. A projected position
that jumps implausibly fast relative to both neighbours is flagged as well.
"""

import numpy as np

from match_tagger.pitch.homography import project


def expected_ball_px(matrix: np.ndarray, x_m: float, y_m: float, diameter_m: float) -> float:
    """Pixel width a ground ball of ``diameter_m`` would have at pitch point (x_m, y_m)."""
    half = diameter_m / 2
    pts = project(np.linalg.inv(matrix), np.array([[x_m - half, y_m], [x_m + half, y_m]]))
    return float(np.linalg.norm(pts[1] - pts[0]))


def speed_spikes(t: np.ndarray, xy: np.ndarray, max_speed_mps: float) -> np.ndarray:
    """Flag observations whose speed to both neighbours exceeds the limit.

    The first and last observation are never flagged by speed: with one neighbour it is
    unknown which of the two positions is wrong.
    """
    flags = np.zeros(len(t), dtype=bool)
    if len(t) < 3:
        return flags
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
    fast = step > max_speed_mps
    flags[1:-1] = fast[:-1] & fast[1:]
    return flags


def airborne_flags(
    t: np.ndarray, xy: np.ndarray, size_ratio: np.ndarray, max_ratio: float, max_speed_mps: float
) -> np.ndarray:
    """Airborne if too large for the projected position or a speed spike (NaN ratio: no check)."""
    too_big = np.nan_to_num(size_ratio, nan=0.0) > max_ratio
    return too_big | speed_spikes(t, xy, max_speed_mps)


def interpolate_gaps(
    times: np.ndarray, xy: np.ndarray, max_gap_sec: float, blocked: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN runs linearly when both ends are known and they are <= max_gap_sec apart.

    ``times`` covers every output frame; ``xy`` holds NaN where no ground ball is known.
    Gaps containing a ``blocked`` frame (an airborne ball) are never filled: a straight ground
    line under a ball in flight would invent possession for players below it.
    Returns the filled positions and a mask of the interpolated frames.
    """
    out, filled = xy.copy(), np.zeros(len(times), dtype=bool)
    blocked = np.zeros(len(times), dtype=bool) if blocked is None else blocked
    known = np.flatnonzero(~np.isnan(xy[:, 0]))
    for a, b in zip(known, known[1:], strict=False):
        if b - a < 2 or times[b] - times[a] > max_gap_sec + 1e-9 or blocked[a + 1 : b].any():
            continue
        w = (times[a + 1 : b] - times[a]) / (times[b] - times[a])
        out[a + 1 : b] = xy[a] + w[:, None] * (xy[b] - xy[a])
        filled[a + 1 : b] = True
    return out, filled
