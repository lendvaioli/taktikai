"""Tactical metrics: pure functions on events of one match and its context.

Every function returns a pydantic model; a metric that cannot be computed returns
``Unavailable`` with a reason instead of a guess.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from match_tagger.events.schema import Event, Outcome, Team
from match_tagger.metrics.config import PassNetworkConfig, PressingConfig
from match_tagger.metrics.context import MatchContext, other
from match_tagger.metrics.frame import event_zone, to_attack_frame, zone_frame


class Unavailable(BaseModel):
    """A metric that cannot be computed for this match."""

    available: Literal[False] = False
    reason: str


class ZoneCount(BaseModel):
    """Count and share of located events in one zone."""

    count: int
    share: float


class BallLosses(BaseModel):
    """Ball losses of a team per zone (shares of the located ones)."""

    available: Literal[True] = True
    team: Team
    frame: str
    total: int
    located: int
    zones: dict[str, ZoneCount]


class Node(BaseModel):
    """A pass network node: a player or a zone, with the passes it played."""

    id: str
    passes: int
    mean_x_m: float | None = None
    mean_y_m: float | None = None


class Edge(BaseModel):
    """Passes from one node to another."""

    source: str
    target: str
    count: int


class PassNetwork(BaseModel):
    """Pass network of a team: players (completed passes) or zones (all passes with an end)."""

    available: Literal[True] = True
    team: Team
    node: Literal["player", "zone"]
    frame: str
    passes: int
    complete: int
    completion_share: float | None
    nodes: list[Node]
    edges: list[Edge]


class Pressing(BaseModel):
    """PPDA-style proxy: opponent passes per opponent ball loss in the opponent's own area."""

    available: Literal[True] = True
    team: Team
    value: float | None
    opponent_passes: int
    opponent_ball_losses: int
    zone_max_x_share: float


class Shot(BaseModel):
    """One shot in the shooter's attack frame (absolute frame if the direction is unknown)."""

    id: str
    t_sec: float
    x_m: float | None
    y_m: float | None
    xg: float | None
    outcome: Outcome | None


class ShotMap(BaseModel):
    """Shots of a team with totals."""

    available: Literal[True] = True
    team: Team
    shots: int
    goals: int
    xg_total: float | None
    items: list[Shot]


def _of(events: Sequence[Event], team: Team, type_: str) -> list[Event]:
    return [e for e in events if e.team == team and e.type == type_]


def ball_losses_by_zone(events: Sequence[Event], ctx: MatchContext, team: Team) -> BallLosses:
    """Ball losses of ``team`` per zone (attack-relative if the direction is known)."""
    losses = _of(events, team, "ball_loss")
    zones = Counter(z for z in (event_zone(e, ctx) for e in losses) if z is not None)
    located = sum(zones.values())
    return BallLosses(
        team=team,
        frame=zone_frame(ctx),
        total=len(losses),
        located=located,
        zones={
            k: ZoneCount(count=c, share=round(c / located, 3))
            for k, c in sorted(zones.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    )


def _completion(passes: Sequence[Event]) -> tuple[int, float | None]:
    known = [p for p in passes if p.outcome is not None]
    complete = sum(p.outcome == "complete" for p in known)
    return complete, round(complete / len(known), 3) if known else None


def _top_edges(counts: Counter, cfg: PassNetworkConfig) -> list[Edge]:
    kept = [(k, c) for k, c in counts.items() if c >= cfg.min_edge_count]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return [Edge(source=s, target=t, count=c) for (s, t), c in kept[: cfg.max_edges]]


def _player_nodes(passes: Sequence[Event], ctx: MatchContext) -> list[Node]:
    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    made = Counter(p.player for p in passes if p.player)
    for p in passes:
        framed = to_attack_frame(p, ctx) or p
        if p.player and framed.x_m is not None:
            points[p.player].append((framed.x_m, framed.y_m))
    return [
        Node(
            id=name,
            passes=n,
            mean_x_m=round(sum(x for x, _ in points[name]) / len(points[name]), 1)
            if points[name]
            else None,
            mean_y_m=round(sum(y for _, y in points[name]) / len(points[name]), 1)
            if points[name]
            else None,
        )
        for name, n in sorted(made.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def pass_network(
    events: Sequence[Event],
    ctx: MatchContext,
    team: Team,
    node: Literal["player", "zone"],
    cfg: PassNetworkConfig,
) -> PassNetwork | Unavailable:
    """Player network (completed passes passer -> recipient) or zone network (start -> end)."""
    passes = _of(events, team, "pass")
    complete, share = _completion(passes)
    if node == "player":
        if not any(p.player and p.recipient for p in passes):
            return Unavailable(reason="no passer and recipient identities in these events")
        counts = Counter(
            (p.player, p.recipient)
            for p in passes
            if p.player and p.recipient and p.outcome == "complete"
        )
        nodes = _player_nodes(passes, ctx)
    else:
        ends = [(event_zone(p, ctx), event_zone(p, ctx, end=True)) for p in passes]
        counts = Counter((s, t) for s, t in ends if s is not None and t is not None)
        starts = Counter(s for s, _ in ends if s is not None)
        nodes = [Node(id=z, passes=n) for z, n in sorted(starts.items(), key=lambda kv: -kv[1])]
    return PassNetwork(
        team=team,
        node=node,
        frame=zone_frame(ctx),
        passes=len(passes),
        complete=complete,
        completion_share=share,
        nodes=nodes,
        edges=_top_edges(counts, cfg),
    )


def ppda_proxy(
    events: Sequence[Event], ctx: MatchContext, team: Team, cfg: PressingConfig
) -> Pressing | Unavailable:
    """Pressing of ``team``: opponent passes / opponent ball losses in the opponent's own area.

    The area is x <= zone_max_x_share * length in the opponent's attack frame. Real PPDA
    divides by the pressing team's defensive actions (tackles, interceptions, fouls); the
    only defensive outcome these events record is the opponent's ball loss.
    """
    if ctx.home_attacks is None:
        return Unavailable(reason="direction of play unknown (home_attacks not set)")
    limit = cfg.zone_max_x_share * ctx.dims.length_m

    def in_area(e: Event) -> bool:
        framed = to_attack_frame(e, ctx)
        return framed is not None and framed.x_m is not None and framed.x_m <= limit

    opp = other(team)
    passes = sum(in_area(e) for e in _of(events, opp, "pass"))
    losses = sum(in_area(e) for e in _of(events, opp, "ball_loss"))
    return Pressing(
        team=team,
        value=round(passes / losses, 2) if losses else None,
        opponent_passes=passes,
        opponent_ball_losses=losses,
        zone_max_x_share=cfg.zone_max_x_share,
    )


def shot_map(events: Sequence[Event], ctx: MatchContext, team: Team) -> ShotMap | Unavailable:
    """Shots of ``team``; unavailable for pipeline events (shots are not detected yet)."""
    if ctx.source == "pipeline":
        return Unavailable(reason="shots are not detected by the events stage yet")
    shots = _of(events, team, "shot")
    items = []
    for s in shots:
        framed = to_attack_frame(s, ctx) or s
        items.append(
            Shot(id=s.id, t_sec=s.t_sec, x_m=framed.x_m, y_m=framed.y_m, xg=s.xg, outcome=s.outcome)
        )
    xgs = [s.xg for s in shots if s.xg is not None]
    return ShotMap(
        team=team,
        shots=len(shots),
        goals=sum(s.outcome == "goal" for s in shots),
        xg_total=round(sum(xgs), 2) if xgs else None,
        items=items,
    )
