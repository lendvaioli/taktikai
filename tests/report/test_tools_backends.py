from types import SimpleNamespace

import pytest

from match_tagger.report.backends import AnthropicBackend, OllamaBackend, make_backend
from match_tagger.report.config import AnthropicConfig, OllamaConfig, ReportConfig
from match_tagger.report.schema import Message, ToolCall
from match_tagger.report.tools import SUBMIT_SPEC, tool_specs


def test_get_metric_summary_and_zone_labels(box):
    out = box.run(
        ToolCall(id="1", name="get_metric", arguments={"name": "summary", "team": "home"})
    )
    assert out["summary"]["ball_losses"] == 3 and out["sample"]["reliability"] == "very_low"
    zones = box.run(
        ToolCall(
            id="2", name="get_metric", arguments={"name": "ball_losses_by_zone", "team": "home"}
        )
    )
    # (75, 10) and (80, 12): final third, right lane; (72, 50): final third, left lane
    assert {k: v["count"] for k, v in zones["zones"].items()} == {"final_right": 2, "final_left": 1}
    assert zones["zone_labels_hu"]["final_right"] == "támadóharmad, jobb sáv"
    assert len(box.outputs) == 2


def test_get_events_filters_by_zone_and_caps(box):
    out = box.run(
        ToolCall(
            id="1",
            name="get_events",
            arguments={"type": "ball_loss", "team": "home", "zone": "final_right"},
        )
    )
    assert (out["total"], [e["id"] for e in out["events"]]) == (2, ["m-e3", "m-e4"])
    box.max_events = 1
    capped = box.run(
        ToolCall(id="2", name="get_events", arguments={"type": "pass", "team": "home"})
    )
    assert (capped["total"], capped["returned"]) == (2, 1)
    assert capped["events"][0]["end_zone"] == "middle_right"


def test_get_clips_records_returned_ids(box, box_no_video):
    out = box.run(ToolCall(id="1", name="get_clips", arguments={"event_ids": ["m-e3", "nope"]}))
    assert out["clips"]["m-e3"]["clip_id"] == "m-e3" and out["clips"]["nope"] is None
    assert box.returned_clips == {"m-e3"}
    none = box_no_video.run(ToolCall(id="1", name="get_clips", arguments={"event_ids": ["m-e3"]}))
    assert none == {"has_video": False, "clips": {}}


def test_bad_arguments_and_unknown_tools_return_errors(box):
    assert "error" in box.run(
        ToolCall(id="1", name="get_metric", arguments={"name": "xg", "team": "home"})
    )
    assert "error" in box.run(ToolCall(id="2", name="drop_table", arguments={}))


def test_submit_schema_has_no_refs():
    assert "$ref" not in str(SUBMIT_SPEC.parameters)
    assert SUBMIT_SPEC.parameters["properties"]["findings"]["maxItems"] == 3


def test_ollama_wire_format_and_usage():
    sent = {}

    def post(url, body, timeout):
        sent.update(url=url, body=body)
        call = {
            "function": {"name": "get_metric", "arguments": {"name": "summary", "team": "home"}}
        }
        return {
            "model": "qwen",
            "message": {"role": "assistant", "content": "", "tool_calls": [call]},
            "prompt_eval_count": 321,
            "eval_count": 12,
        }

    backend = OllamaBackend(OllamaConfig(options={"seed": 0}), post=post)
    history = [
        Message(role="user", content="u"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c0", name="get_metric", arguments={"team": "home"})],
        ),
        Message(role="tool", content="{}", tool_call_id="c0", tool_name="get_metric"),
    ]
    out = backend.chat("sys", history, tool_specs())
    body = sent["body"]
    assert sent["url"].endswith("/api/chat") and body["stream"] is False and body["think"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "get_metric"
    assert body["messages"][3] == {"role": "tool", "content": "{}", "tool_name": "get_metric"}
    assert body["tools"][0]["type"] == "function" and body["options"] == {"seed": 0}
    assert (out.tool_calls[0].name, out.input_tokens, out.output_tokens) == ("get_metric", 321, 12)


class _FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _anthropic(stop_reason="tool_use", cfg=None):
    content = [
        SimpleNamespace(type="text", text="Megnézem."),
        SimpleNamespace(
            type="tool_use", id="tu_1", name="get_metric", input={"name": "summary", "team": "home"}
        ),
    ]
    response = SimpleNamespace(
        content=content, stop_reason=stop_reason, model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=500, output_tokens=40),
    )  # fmt: skip
    messages = _FakeMessages(response)
    return AnthropicBackend(
        cfg or AnthropicConfig(), client=SimpleNamespace(beta=SimpleNamespace(messages=messages))
    ), messages


def test_anthropic_wire_format_with_mocked_client():
    backend, fake = _anthropic()
    raw = ["<original content blocks>"]
    history = [
        Message(role="user", content="u"),
        Message(role="assistant", raw=raw, tool_calls=[ToolCall(id="tu_0", name="a", arguments={}),
                                                      ToolCall(id="tu_9", name="b", arguments={})]),
        Message(role="tool", content="r0", tool_call_id="tu_0", tool_name="a"),
        Message(role="tool", content="r9", tool_call_id="tu_9", tool_name="b"),
    ]  # fmt: skip
    out = backend.chat("sys", history, tool_specs())
    kw = fake.kwargs
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "sys"
    assert "fallbacks" not in kw and "betas" not in kw  # Haiku 5.5 has no server fallback
    assert kw["messages"][1] == {"role": "assistant", "content": raw}  # sent back unchanged
    results = kw["messages"][2]["content"]  # both results in one user message
    assert [r["tool_use_id"] for r in results] == ["tu_0", "tu_9"]
    assert kw["tools"][0]["input_schema"]["required"] == ["name", "team"]
    assert (out.text, out.tool_calls[0].id, out.input_tokens) == ("Megnézem.", "tu_1", 500)


def test_anthropic_server_fallback_is_opt_in():
    backend, fake = _anthropic(cfg=AnthropicConfig(model="claude-opus-5-5", server_fallback=True))
    backend.chat("sys", [Message(role="user", content="u")], [])
    assert fake.kwargs["fallbacks"] == "default"
    assert fake.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_refusal_raises():
    backend, _ = _anthropic(stop_reason="refusal")
    with pytest.raises(RuntimeError, match="refused"):
        backend.chat("sys", [Message(role="user", content="u")], [])


def test_backend_choice_from_env():
    cfg = ReportConfig()
    assert make_backend(cfg, env={}).name == "ollama"
    with pytest.raises(ValueError, match="unknown"):
        make_backend(cfg, env={"TAKTIKAI_LLM_BACKEND": "gpt"})
