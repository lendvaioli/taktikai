"""LLM backends behind one interface: a local Ollama model (default) and the Anthropic API.

Both turn the backend-neutral ``Message`` list into their wire format and return an
``LLMReply`` with tool calls, token counts and latency. Neither is called in tests: the
Ollama backend takes an injectable ``post`` function, the Anthropic backend a client.
"""

import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from match_tagger.report.config import AnthropicConfig, OllamaConfig, ReportConfig
from match_tagger.report.schema import LLMReply, Message, ToolCall, ToolSpec

if TYPE_CHECKING:
    import anthropic

Post = Callable[[str, dict[str, Any], float], dict[str, Any]]
Get = Callable[[str, float], dict[str, Any]]
OLLAMA_TIMINGS = ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration")


class LLMBackend(Protocol):
    """A chat model that can call tools."""

    name: str
    model: str

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        """One model call over the whole conversation."""
        ...


def http_post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST JSON and parse the JSON response (stdlib only)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local server)
        return json.loads(resp.read())


def http_get(url: str, timeout: float) -> dict[str, Any]:
    """GET and parse a JSON response (stdlib only)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local server)
        return json.loads(resp.read())


def ollama_timings_ms(resp: dict[str, Any]) -> dict[str, float]:
    """Ollama's own durations of a call (nanoseconds in the API) in milliseconds."""
    return {
        k.removesuffix("_duration") + "_ms": round(resp[k] / 1e6, 1)
        for k in OLLAMA_TIMINGS
        if k in resp
    }


class OllamaBackend:
    """Ollama /api/chat with native tool calling (docs: ollama/docs/api.md)."""

    name = "ollama"

    def __init__(self, cfg: OllamaConfig, post: Post = http_post, get: Get = http_get) -> None:
        """Keep the config and the HTTP functions."""
        self.cfg, self.post, self.get, self.model = cfg, post, get, cfg.model

    def gpu_share(self) -> float | None:
        """Share of the loaded model in GPU memory (/api/ps: size_vram / size); None if unknown."""
        try:
            loaded = self.get(f"{self.cfg.host}/api/ps", 10.0).get("models", [])
        except OSError:
            return None
        model = next((m for m in loaded if m.get("name") == self.cfg.model), None)
        if not model or not model.get("size"):
            return None
        return round(model.get("size_vram", 0) / model["size"], 3)

    @staticmethod
    def _message(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "content": m.content, "tool_name": m.tool_name}
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
            ]
        return out

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        """Call the model once (non-streaming)."""
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}]
            + [self._message(m) for m in messages],
            "tools": [{"type": "function", "function": t.model_dump()} for t in tools],
            "stream": False,
            "think": self.cfg.think,
            "options": self.cfg.options,
        }
        started = time.perf_counter()
        resp = self.post(f"{self.cfg.host}/api/chat", body, self.cfg.timeout_sec)
        msg = resp.get("message", {})
        calls = [
            ToolCall(id=f"call_{i}", name=c["function"]["name"], arguments=_args(c["function"]))
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        return LLMReply(
            text=msg.get("content") or "",
            tool_calls=calls,
            input_tokens=resp.get("prompt_eval_count"),
            output_tokens=resp.get("eval_count"),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=resp.get("model", self.cfg.model),
            backend_timings_ms=ollama_timings_ms(resp),
            gpu_share=self.gpu_share(),
        )


def _args(function: dict[str, Any]) -> dict[str, Any]:
    """Tool call arguments; documented as an object, tolerated as a JSON string."""
    args = function.get("arguments") or {}
    return json.loads(args) if isinstance(args, str) else dict(args)


class AnthropicBackend:
    """Anthropic Messages API with client tools (optional; the SDK is the ``anthropic`` extra)."""

    name = "anthropic"

    def __init__(self, cfg: AnthropicConfig, client: "anthropic.Anthropic | None" = None) -> None:
        """Use the given client or build one from the environment's credentials."""
        if client is None:
            import anthropic  # optional dependency, only needed for this backend

            client = anthropic.Anthropic()
        self.cfg, self.client, self.model = cfg, client, cfg.model

    @staticmethod
    def _messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Wire messages; consecutive tool results go into one user message."""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                out.append(
                    {"role": "assistant", "content": m.raw if m.raw is not None else m.content}
                )
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def chat(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMReply:
        """Call the model once; a refusal raises instead of returning an empty reply."""
        fallback = (
            {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            if self.cfg.server_fallback
            else {}
        )
        started = time.perf_counter()
        resp = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=self._messages(messages),
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
            output_config={"effort": self.cfg.effort},
            **fallback,
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        return LLMReply(
            text="".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
                for b in resp.content
                if b.type == "tool_use"
            ],
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=resp.model,
            raw=resp.content,
        )


def make_backend(cfg: ReportConfig, env: dict[str, str] | None = None) -> LLMBackend:
    """The configured backend; the environment variable ``cfg.backend_env`` overrides it."""
    env = os.environ if env is None else env
    name = env.get(cfg.backend_env) or cfg.backend
    if name == "ollama":
        return OllamaBackend(cfg.ollama)
    if name == "anthropic":
        return AnthropicBackend(cfg.anthropic)
    raise ValueError(f"unknown LLM backend {name!r} (ollama | anthropic)")
