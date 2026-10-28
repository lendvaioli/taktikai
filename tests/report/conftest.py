from collections.abc import Sequence

import pytest

from match_tagger.clips.plan import ClipPlan
from match_tagger.clips.stage import ClipRecord, ClipsIndex
from match_tagger.events.schema import Event
from match_tagger.metrics.config import MetricsConfig
from match_tagger.metrics.context import MatchContext
from match_tagger.metrics.stage import compute_metrics
from match_tagger.report.schema import LLMReply, Message, ToolCall, ToolSpec
from match_tagger.report.tools import ToolBox


# 10 minutes, home attacks right: 2 home passes (1 complete), 3 home ball losses in
# x 70-80 (final third), 1 away pass.
def _ev(n, type_, team, x, y, end=None, outcome=None):
    end_x, end_y = end or (None, None)
    return Event(id=f"m-e{n}", t_sec=10.0 * n, type=type_, team=team, x_m=x, y_m=y,
                 end_x_m=end_x, end_y_m=end_y, outcome=outcome)  # fmt: skip


EVENTS = [
    _ev(1, "pass", "home", 20, 10, end=(40, 10), outcome="complete"),
    _ev(2, "pass", "home", 50, 30, end=(60, 30), outcome="incomplete"),
    _ev(3, "ball_loss", "home", 75, 10),
    _ev(4, "ball_loss", "home", 80, 12),
    _ev(5, "ball_loss", "home", 72, 50),
    _ev(6, "pass", "away", 50, 30, end=(40, 30), outcome="complete"),
]


def _ctx(has_video=True):
    return MatchContext(
        match_id="m",
        source="pipeline",
        duration_sec=600,
        teams={"home": "Hazai (fehér mez)", "away": "Vendég (piros mez)"},
        has_video=has_video,
        home_attacks="right",
    )


def _clips():
    plan = ClipPlan(requested_start=25, requested_end=33, start=25, end=33, method="reencode")
    return ClipsIndex(
        match_id="m",
        clips=[
            ClipRecord(event_id=e.id, t_sec=e.t_sec, type=e.type, team=e.team, path=f"{e.id}.mp4",
                       plan=plan, method="reencode", params_hash="x")
            for e in EVENTS
        ],
        methods={"reencode": len(EVENTS)},
        skipped_existing=0,
        seconds=0.0,
    )  # fmt: skip


@pytest.fixture
def box():
    return ToolBox(
        compute_metrics(EVENTS, _ctx(), MetricsConfig()), EVENTS, _clips(), max_events=30
    )


@pytest.fixture
def box_no_video():
    return ToolBox(
        compute_metrics(EVENTS, _ctx(False), MetricsConfig()), EVENTS, None, max_events=30
    )


class ScriptedBackend:
    """Returns prepared replies in order (the last one repeats); records what it was offered."""

    name = "scripted"
    model = "fake-1"

    def __init__(self, replies: Sequence[LLMReply]):
        self.replies = list(replies)
        self.calls: list[tuple[list[Message], list[str]]] = []

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        self.calls.append((list(messages), [t.name for t in tools]))
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


def reply(*calls: tuple[str, dict], text: str = "") -> LLMReply:
    return LLMReply(
        text=text,
        tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)],
        input_tokens=100,
        output_tokens=20,
        latency_ms=5.0,
        model="fake-1",
    )


SUMMARY_HOME = {"name": "summary", "team": "home"}


def report_args(loss_count: float = 3, clip: str = "m-e3", note_minutes: str = "10") -> dict:
    """A report whose numbers all come from get_metric summary (home): 3 ball losses."""
    source = {"tool": "get_metric", "arguments": SUMMARY_HOME, "path": "summary.ball_losses"}
    finding = {
        "title": "Labdavesztések a támadóharmadban",
        "claim": f"A hazai csapat {loss_count:g} labdát veszített, főleg a jobb sávban.",
        "numbers": [{"label": "labdavesztés", "value": loss_count, "source": source}],
        "clip_ids": [clip],
        "training": "Kis létszámú játék szűk területen, gyors döntésekkel.",
    }
    return {
        "match_id": "m",
        "sample_note": f"{note_minutes} perc, 6 esemény: nagyon kevés adat.",
        "findings": [finding, {**finding, "title": "Második megfigyelés"}],
    }
