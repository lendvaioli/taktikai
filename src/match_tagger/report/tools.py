"""Tools of the report agent: read-only views of metrics.json, the events and the clip index.

Every result is recorded in ``ToolBox.outputs``; the numeric validator only accepts numbers
that appear there, and only clip ids that ``get_clips`` returned count as citable.
"""

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from match_tagger.clips.stage import ClipsIndex
from match_tagger.events.schema import EVENT_TYPES, Event, EventType, Team
from match_tagger.metrics.context import team_labels_hu
from match_tagger.metrics.frame import event_zone
from match_tagger.metrics.stage import METRIC_NAMES, MatchMetrics
from match_tagger.report.sample_note import RELIABILITY_HU
from match_tagger.report.schema import Report, ToolCall, ToolSpec

TEAM_SCHEMA = {"type": "string", "enum": ["home", "away"]}
THIRD_LANE_LABELS_HU = {
    "own": "saját harmad",
    "middle": "középső harmad",
    "final": "támadóharmad",
    "right": "jobb sáv",
    "centre": "középső sáv",
    "left": "bal sáv",
}
FIELD_LABELS_HU = {
    "passes": "passzok",
    "passes_complete": "sikeres passzok",
    "pass_completion_share": "passzpontosság (arány)",
    "complete": "sikeres passzok",
    "completion_share": "passzpontosság (arány)",
    "ball_losses": "labdavesztések",
    "total": "összesen",
    "located": "helyhez kötött",
    "count": "darab",
    "share": "arány",
    "thirds": "harmadonként",
    "lanes": "sávonként",
    "value": "letámadási mutató (PPDA-proxy)",
    "opponent_passes": "ellenfél passzai a saját területén",
    "opponent_ball_losses": "ellenfél labdavesztései a saját területén",
    "shots": "lövések",
    "goals": "gólok",
    "xg_total": "várható gólok (xG) összesen",
    "duration_sec": "időtartam (másodperc)",
    "duration_min": "időtartam (perc)",
    "n_events": "események száma",
    "events_per_type": "események típusonként",
}


def normalize_arguments(tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Arguments in canonical form (defaults filled), or None if they are invalid."""
    models: dict[str, type[BaseModel]] = {
        "get_metric": MetricArgs,
        "get_events": EventArgs,
        "get_clips": ClipArgs,
    }
    try:
        return models[tool].model_validate(args).model_dump()
    except (KeyError, ValidationError):
        return None


class MetricArgs(BaseModel):
    """get_metric arguments."""

    name: Literal["summary", "ball_losses_by_zone", "pass_network", "ppda", "shot_map"]
    team: Team


class EventArgs(BaseModel):
    """get_events arguments."""

    type: EventType
    team: Team
    zone: str | None = None


class ClipArgs(BaseModel):
    """get_clips arguments."""

    event_ids: list[str]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """A pydantic JSON schema with every $ref replaced by its definition (small models)."""
    defs = schema.get("$defs", {})

    def walk(node: object) -> object:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


SUBMIT_SPEC = ToolSpec(
    name="submit_report",
    description="Submit the final report (2-3 findings). "
    "Every number must come from a tool result.",
    parameters=inline_refs(Report.model_json_schema()),
)


def tool_specs() -> list[ToolSpec]:
    """The three data tools with JSON schemas."""
    return [
        ToolSpec(
            name="get_metric",
            description="One metric for one team. 'summary' also returns the sample block "
            "(duration, event counts, reliability) and the team names.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(METRIC_NAMES)},
                    "team": TEAM_SCHEMA,
                },
                "required": ["name", "team"],
            },
        ),
        ToolSpec(
            name="get_events",
            description="Events of one type and team, optionally in one zone (keys as in the "
            "metrics). Returns the total and at most the first N events with ids.",
            parameters={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(EVENT_TYPES)},
                    "team": TEAM_SCHEMA,
                    "zone": {"type": "string"},
                },
                "required": ["type", "team"],
            },
        ),
        ToolSpec(
            name="get_clips",
            description="Video clip ids and time windows for event ids "
            "(only if the match has video).",
            parameters={
                "type": "object",
                "properties": {"event_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["event_ids"],
            },
        ),
    ]


class ToolBox:
    """Executes tool calls for one match and remembers every result."""

    def __init__(
        self,
        metrics: MatchMetrics,
        events: Sequence[Event],
        clips: ClipsIndex | None,
        max_events: int,
    ) -> None:
        """Bind the match data; nothing is read from disk afterwards."""
        self.metrics, self.events, self.max_events = metrics, list(events), max_events
        self.clips = {c.event_id: c for c in clips.clips} if clips else {}
        self.calls: list[tuple[ToolCall, dict[str, Any]]] = []
        self.returned_clips: set[str] = set()

    @property
    def outputs(self) -> list[dict[str, Any]]:
        """Every tool result of the run, in call order."""
        return [result for _, result in self.calls]

    def run(self, call: ToolCall) -> dict[str, Any]:
        """Run one call; invalid arguments or unknown tools give an error result."""
        handlers = {
            "get_metric": self.get_metric,
            "get_events": self.get_events,
            "get_clips": self.get_clips,
        }
        try:
            if call.name not in handlers:
                raise ValueError(f"unknown tool {call.name!r}")
            result = handlers[call.name](call.arguments)
        except (ValidationError, ValueError) as exc:
            result = {"error": str(exc).splitlines()[0]}
        self.calls.append((call, result))
        return result

    def get_metric(self, args: dict[str, Any]) -> dict[str, Any]:
        """A slice of metrics.json with Hungarian labels for its fields."""
        a = MetricArgs.model_validate(args)
        m = self.metrics
        if a.name == "summary":
            return {
                "team_name_hu": team_labels_hu(m.context)[a.team],
                "summary": m.summary[a.team].model_dump(),
                "sample": m.sample.model_dump(),
                "reliability_hu": RELIABILITY_HU[m.sample.reliability],
                "has_video": m.context.has_video,
                "labels_hu": FIELD_LABELS_HU,
            }
        out = getattr(m, a.name)[a.team].model_dump()
        if a.name in ("ball_losses_by_zone", "pass_network") and m.zone_labels_hu:
            out["zone_labels_hu"] = m.zone_labels_hu | THIRD_LANE_LABELS_HU
        return out | {"labels_hu": FIELD_LABELS_HU}

    def _row(self, e: Event) -> dict[str, Any]:
        ctx = self.metrics.context
        row = {"id": e.id, "t_sec": e.t_sec, "zone": event_zone(e, ctx)}
        if e.type == "pass":
            row["end_zone"] = event_zone(e, ctx, end=True)
        extra = {"outcome": e.outcome, "player": e.player, "recipient": e.recipient, "xg": e.xg}
        return row | {k: v for k, v in extra.items() if v is not None}

    def get_events(self, args: dict[str, Any]) -> dict[str, Any]:
        """Filtered events, capped at max_events."""
        a = EventArgs.model_validate(args)
        ctx = self.metrics.context
        hits = [
            e
            for e in self.events
            if e.type == a.type
            and e.team == a.team
            and (a.zone is None or event_zone(e, ctx) == a.zone)
        ]
        rows = [self._row(e) for e in hits[: self.max_events]]
        return {"total": len(hits), "returned": len(rows), "events": rows}

    def get_clips(self, args: dict[str, Any]) -> dict[str, Any]:
        """Clip ids and windows of the given events (None for events without a clip)."""
        a = ClipArgs.model_validate(args)
        if not self.metrics.context.has_video:
            return {"has_video": False, "clips": {}}
        found: dict[str, Any] = {}
        for event_id in a.event_ids:
            rec = self.clips.get(event_id)
            found[event_id] = (
                {"clip_id": rec.event_id, "start_sec": rec.plan.start, "end_sec": rec.plan.end}
                if rec
                else None
            )
            if rec:
                self.returned_clips.add(rec.event_id)
        return {"has_video": True, "clips": found}
