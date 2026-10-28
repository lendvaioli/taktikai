"""Attack frame and zones of the metrics.

In the attack frame the acting team attacks towards x = length; y = 0 is on its right (the
absolute frame has its origin bottom-left, so turning it by 180 degrees keeps that true).
Attack-relative zones are thirds (own / middle / final) x lanes (right / centre / left, as
seen by the attacking team). Without a known direction the 18 absolute zones of
``events.zones`` are used and keyed "1".."18".
"""

from match_tagger.events.schema import Event
from match_tagger.events.zones import zone as absolute_zone
from match_tagger.metrics.context import MatchContext, attack_sign
from match_tagger.pitch.geometry import PitchDims

THIRDS = ("own", "middle", "final")
LANES = ("right", "centre", "left")
_THIRDS_HU = {"own": "saját harmad", "middle": "középső harmad", "final": "támadóharmad"}
_LANES_HU = {"right": "jobb sáv", "centre": "középső sáv", "left": "bal sáv"}
ZONE_LABELS_HU = {
    f"{t}_{lane}": f"{_THIRDS_HU[t]}, {_LANES_HU[lane]}" for t in THIRDS for lane in LANES
}


def _turn(x: float | None, y: float | None, dims: PitchDims) -> tuple[float | None, float | None]:
    if x is None or y is None:
        return x, y
    return round(dims.length_m - x, 2), round(dims.width_m - y, 2)


def to_attack_frame(event: Event, ctx: MatchContext) -> Event | None:
    """The event with coordinates in its team's attack frame; None if the direction is unknown."""
    sign = attack_sign(ctx, event.team, event.period)
    if sign is None:
        return None
    if sign == 1:
        return event
    x, y = _turn(event.x_m, event.y_m, ctx.dims)
    end_x, end_y = _turn(event.end_x_m, event.end_y_m, ctx.dims)
    return event.model_copy(update={"x_m": x, "y_m": y, "end_x_m": end_x, "end_y_m": end_y})


def relative_zone(x: float | None, y: float | None, dims: PitchDims) -> str | None:
    """Attack-relative zone key such as "final_left" of a point in the attack frame."""
    if x is None or y is None:
        return None
    third = min(max(int(x / (dims.length_m / 3)), 0), 2)
    lane = min(max(int(y / (dims.width_m / 3)), 0), 2)
    return f"{THIRDS[third]}_{LANES[lane]}"


def event_zone(event: Event, ctx: MatchContext, end: bool = False) -> str | None:
    """Zone key of an event's start (or end) point: attack-relative if possible, else absolute."""
    framed = to_attack_frame(event, ctx)
    source = framed or event
    x, y = (source.end_x_m, source.end_y_m) if end else (source.x_m, source.y_m)
    if framed is not None:
        return relative_zone(x, y, ctx.dims)
    number = absolute_zone(x, y, ctx.dims)
    return None if number is None else str(number)


def zone_frame(ctx: MatchContext) -> str:
    """The zone frame of a match: "attack" (attack-relative) or "absolute"."""
    return "absolute" if ctx.home_attacks is None else "attack"
