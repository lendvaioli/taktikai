import json

from match_tagger.report.agent import AgentSettings, feedback, run_agent
from match_tagger.report.log import RunLog
from match_tagger.report.stage import prompt_version

from .conftest import ScriptedBackend, reply, report_args

LOOKUPS = [
    ("get_metric", {"name": "summary", "team": "home"}),
    ("get_events", {"type": "ball_loss", "team": "home"}),
    ("get_clips", {"event_ids": ["m-e3"]}),
]


def _run(backend, box, tmp_path, **kw):
    log = RunLog(tmp_path / "runs.jsonl", "r1", {"backend": backend.name})
    settings = AgentSettings(match_id="m", has_video=box.metrics.context.has_video, **kw)
    return run_agent(backend, "system", "user", box, settings, log)


def _lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "runs.jsonl").read_text("utf-8").splitlines()]


def test_loop_stops_after_max_steps(box, tmp_path):
    backend = ScriptedBackend([reply(LOOKUPS[0])])  # never submits
    result = _run(backend, box, tmp_path)
    assert (result.status, result.steps, result.report) == ("max_steps", 8, None)
    assert len(backend.calls) == 8
    assert backend.calls[-1][1] == ["submit_report"]  # the last step only allows submitting
    assert "get_metric" in backend.calls[0][1]
    lines = _lines(tmp_path)
    assert [x["kind"] for x in lines] == ["step"] * 8 + ["end"]
    assert lines[-1]["input_tokens"] == 800 and lines[-1]["status"] == "max_steps"


def test_valid_report_is_accepted(box, tmp_path):
    backend = ScriptedBackend([reply(*LOOKUPS), reply(("submit_report", report_args()))])
    result = _run(backend, box, tmp_path)
    assert (result.status, result.steps) == ("ok", 2)
    assert result.report.findings[0].clip_ids == ["m-e3"]
    step1 = _lines(tmp_path)[0]
    assert [t["name"] for t in step1["tool_calls"]] == ["get_metric", "get_events", "get_clips"]
    assert step1["latency_ms"] == 5.0 and step1["output_tokens"] == 20


def test_fabricated_number_is_rejected_then_fixed(box, tmp_path):
    backend = ScriptedBackend(
        [
            reply(*LOOKUPS),
            reply(("submit_report", report_args(loss_count=7))),  # 7 appears in no tool result
            reply(("submit_report", report_args())),
        ]
    )
    result = _run(backend, box, tmp_path)
    assert (result.status, result.steps) == ("ok", 3)
    feedback = backend.calls[2][0][-1]
    assert feedback.role == "tool" and "7 does not appear" in feedback.content
    assert _lines(tmp_path)[1]["validation_errors"]


def test_second_fabrication_is_final(box, tmp_path):
    bad = reply(("submit_report", report_args(loss_count=7)))
    backend = ScriptedBackend([reply(*LOOKUPS), bad, bad])
    result = _run(backend, box, tmp_path)
    assert (result.status, result.report, result.steps) == ("rejected", None, 3)
    assert any("7" in e for e in result.errors)
    assert len(backend.calls) == 3


def test_clip_not_returned_by_get_clips_is_rejected(box, tmp_path):
    backend = ScriptedBackend(
        [reply(*LOOKUPS[:2]), reply(("submit_report", report_args(clip="m-e3")))]
    )
    result = _run(backend, box, tmp_path, retries=0)
    assert result.status == "rejected"
    assert any("not returned by get_clips" in e for e in result.errors)


def test_text_only_reply_gets_a_nudge(box, tmp_path):
    backend = ScriptedBackend(
        [reply(text="Gondolkodom..."), reply(*LOOKUPS), reply(("submit_report", report_args()))]
    )
    result = _run(backend, box, tmp_path)
    assert result.status == "ok"
    assert backend.calls[1][0][-1].content.startswith("Use a tool")


def test_schema_error_counts_as_rejection(box, tmp_path):
    args = report_args()
    args["findings"] = args["findings"][:1]  # only one finding
    backend = ScriptedBackend([reply(*LOOKUPS), reply(("submit_report", args))])
    result = _run(backend, box, tmp_path, retries=0)
    assert result.status == "rejected"
    assert any(e.startswith("findings") for e in result.errors)


def test_feedback_restates_the_rules():
    with_video = feedback(["finding 2: cite at least one clip id from get_clips"], has_video=True)
    assert with_video.startswith("REJECTED.")
    assert "2 or 3 findings" in with_video
    assert "at least one clip id that get_clips returned" in with_video
    assert "empty clip_ids" in feedback(["x"], has_video=False)


def test_prompt_version_label():
    assert prompt_version("<!-- version: report_system v2 -->\n<!-- v1 (old) -->\ntext") == (
        "report_system v2"
    )
    assert prompt_version("no header") is None
