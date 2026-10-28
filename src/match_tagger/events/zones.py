"""18 absolute pitch zones: 6 bands along the length x 3 across the width.

Zones are numbered 1-18 in the pitch frame of the pitch stage (origin bottom-left as seen from
the main camera): zones 1-3 lie at the left goal line (1 near touchline, 3 far touchline),
zones 16-18 at the right goal line. They do not depend on which team attacks which way.
Points slightly outside the pitch count to the nearest zone.
"""

from match_tagger.pitch.geometry import PitchDims

N_LENGTH, N_WIDTH = 6, 3


def zone(x_m: float | None, y_m: float | None, dims: PitchDims) -> int | None:
    """Zone number 1-18 of a pitch position; None without a position."""
    if x_m is None or y_m is None:
        return None
    col = min(max(int(x_m / (dims.length_m / N_LENGTH)), 0), N_LENGTH - 1)
    row = min(max(int(y_m / (dims.width_m / N_WIDTH)), 0), N_WIDTH - 1)
    return col * N_WIDTH + row + 1


def zone_bounds(number: int, dims: PitchDims) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of a zone in metres."""
    if not 1 <= number <= N_LENGTH * N_WIDTH:
        raise ValueError(f"zone must be 1-{N_LENGTH * N_WIDTH}")
    col, row = divmod(number - 1, N_WIDTH)
    dx, dy = dims.length_m / N_LENGTH, dims.width_m / N_WIDTH
    return col * dx, (col + 1) * dx, row * dy, (row + 1) * dy
