"""StatsBomb Open Data events -> our ``Event`` schema, so the metrics run on reference data.

Works on the raw JSON dicts of github.com/statsbomb/open-data (events/<match_id>.json and the
entry of the match in matches/<competition>/<season>.json); no network access here.

- Coordinates: StatsBomb uses 120 x 80 with the origin top-left, every event seen from the
  acting team attacking towards x = 120. They are scaled to the pitch in metres and turned
  into our absolute frame (origin bottom-left) in which the home team attacks right in every
  period, so ``home_attacks`` is "right".
- Types: open-play passes -> pass; passes from a corner, free kick, throw-in, goal kick or
  kick-off -> set_piece; shots -> shot (a free-kick or penalty shot also gets a set_piece, as
  in docs/GOLDEN.md); ball_loss -> every change of possession team within a period that does
  not start with a kick-off, at the first event of the new possession (as in our rules).
- Time: match clock (minute, second plus the fraction of the timestamp); the penalty
  shoot-out (period 5) is left out.
"""

from collections.abc import Iterator
from typing import Any

from match_tagger.events.schema import Event, Outcome, Team
from match_tagger.metrics.context import MatchContext
from match_tagger.pitch.geometry import PitchDims

SB_LENGTH, SB_WIDTH = 120.0, 80.0
SET_PIECE_PASSES = {"corner", "free kick", "throw-in", "goal kick", "kick off"}
SET_PIECE_SHOTS = {"free kick", "penalty"}
SHOT_OUTCOMES: dict[str, Outcome] = {
    "goal": "goal",
    "saved": "saved",
    "saved to post": "saved",
    "blocked": "blocked",
    "off t": "off_target",
    "wayward": "off_target",
    "post": "off_target",
    "saved off target": "off_target",
}
ATTRIBUTION = "Adatforrás: StatsBomb Open Data (github.com/statsbomb/open-data)"
Raw = dict[str, Any]
DEFAULT_DIMS = PitchDims()


def _name(value: Raw | None) -> str:
    return (value or {}).get("name", "").lower()


def point(loc: list[float] | None, home: bool, dims: PitchDims) -> tuple[float, float] | None:
    """StatsBomb location of an event of the home (or away) team -> absolute metres."""
    if not loc:
        return None
    x = loc[0] * dims.length_m / SB_LENGTH
    y = (SB_WIDTH - loc[1]) * dims.width_m / SB_WIDTH
    if not home:
        x, y = dims.length_m - x, dims.width_m - y
    return round(x, 2), round(y, 2)


def t_sec(e: Raw) -> float:
    """Match clock of an event in seconds."""
    fraction = float("0." + e["timestamp"].split(".")[1]) if "." in e["timestamp"] else 0.0
    return round(e["minute"] * 60 + e["second"] + fraction, 2)


def _base(e: Raw, team: Team, type_: str, dims: PitchDims, suffix: str = "") -> dict[str, Any]:
    xy = point(e.get("location"), team == "home", dims)
    return {
        "id": f"e{e['index']}{suffix}",  # short and unique within the match (not the UUID)
        "t_sec": t_sec(e),
        "type": type_,
        "team": team,
        "x_m": xy[0] if xy else None,
        "y_m": xy[1] if xy else None,
        "player": (e.get("player") or {}).get("name"),
        "period": e["period"],
    }


def _pass(e: Raw, team: Team, dims: PitchDims) -> Event:
    p = e["pass"]
    if _name(p.get("type")) in SET_PIECE_PASSES:
        return Event(**_base(e, team, "set_piece", dims))
    end = point(p.get("end_location"), team == "home", dims)
    return Event(
        **_base(e, team, "pass", dims),
        end_x_m=end[0] if end else None,
        end_y_m=end[1] if end else None,
        outcome="incomplete" if p.get("outcome") else "complete",
        recipient=(p.get("recipient") or {}).get("name"),
    )


def _shot(e: Raw, team: Team, dims: PitchDims) -> list[Event]:
    s = e["shot"]
    shot = Event(
        **_base(e, team, "shot", dims),
        xg=s.get("statsbomb_xg"),
        outcome=SHOT_OUTCOMES.get(_name(s.get("outcome"))),
    )
    if _name(s.get("type")) in SET_PIECE_SHOTS:
        return [shot, Event(**_base(e, team, "set_piece", dims, suffix="-sp"))]
    return [shot]


def _ball_losses(events: list[Raw], teams: dict[str, Team], dims: PitchDims) -> Iterator[Event]:
    """A ball loss for the old team at the first event of every new possession of the other."""
    prev: Raw | None = None
    for e in events:
        if prev is not None and e["possession"] != prev["possession"]:
            changed = e["possession_team"]["name"] != prev["possession_team"]["name"]
            kick_off = _name(e.get("play_pattern")) == "from kick off"
            if changed and e["period"] == prev["period"] and not kick_off:
                actor = teams[e["team"]["name"]]
                loser = teams[prev["possession_team"]["name"]]
                xy = point(e.get("location"), actor == "home", dims)  # seen from e's team
                fields = _base(e, loser, "ball_loss", dims, suffix="-loss")
                fields.update(x_m=xy[0] if xy else None, y_m=xy[1] if xy else None, player=None)
                yield Event(**fields)
        prev = e


def convert(events: list[Raw], match: Raw, dims: PitchDims = DEFAULT_DIMS) -> list[Event]:
    """Our events of one StatsBomb match, sorted by time."""
    teams: dict[str, Team] = {
        match["home_team"]["home_team_name"]: "home",
        match["away_team"]["away_team_name"]: "away",
    }
    play = sorted((e for e in events if e["period"] <= 4), key=lambda e: e["index"])
    out: list[Event] = []
    for e in play:
        team = teams[e["team"]["name"]]
        kind = _name(e.get("type"))
        if kind == "pass":
            out.append(_pass(e, team, dims))
        elif kind == "shot":
            out.extend(_shot(e, team, dims))
    out.extend(_ball_losses(play, teams, dims))
    return sorted(out, key=lambda ev: (ev.t_sec, ev.id))


def context(events: list[Raw], match: Raw, dims: PitchDims = DEFAULT_DIMS) -> MatchContext:
    """Match context of a StatsBomb match (no video; home attacks right in our frame)."""
    play = [e for e in events if e["period"] <= 4]
    return MatchContext(
        match_id=f"sb_{match['match_id']}",
        source="statsbomb",
        duration_sec=max(t_sec(e) for e in play),
        teams={
            "home": match["home_team"]["home_team_name"],
            "away": match["away_team"]["away_team_name"],
        },
        has_video=False,
        dims=dims,
        home_attacks="right",
        attribution=ATTRIBUTION,
    )
