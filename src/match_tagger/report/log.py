"""JSONL run log of the report agent: one line per step and one closing line per run."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from match_tagger.report.schema import LLMReply


class ToolRecord(BaseModel):
    """One executed tool call."""

    name: str
    arguments: dict[str, Any]
    ms: float
    result_chars: int
    error: str | None = None


class RunLog:
    """Appends lines to <match>/report/runs.jsonl; every line carries the run id."""

    def __init__(self, path: Path, run_id: str, header: dict[str, Any]) -> None:
        """Remember where and what to log; ``header`` is repeated on every line."""
        self.path, self.run_id, self.header = path, run_id, header
        self.input_tokens = self.output_tokens = 0
        self.latency_ms = 0.0
        path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"run_id": self.run_id, **self.header, **record}, ensure_ascii=False)
            )
            f.write("\n")

    def step(
        self,
        step: int,
        reply: LLMReply,
        tools: list[ToolRecord],
        validation: list[str] | None,
    ) -> None:
        """Log one LLM call with its tool calls and, for a submission, the validator's verdict."""
        self.input_tokens += reply.input_tokens or 0
        self.output_tokens += reply.output_tokens or 0
        self.latency_ms += reply.latency_ms
        self._write(
            {
                "kind": "step",
                "step": step,
                "model": reply.model,
                "latency_ms": reply.latency_ms,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "text_chars": len(reply.text),
                "tool_calls": [t.model_dump() for t in tools],
                "submitted": validation is not None,
                "validation_errors": validation,
            }
        )

    def end(self, status: str, steps: int, errors: list[str]) -> None:
        """Closing line with the outcome and the totals of the run."""
        self._write(
            {
                "kind": "end",
                "status": status,
                "steps": steps,
                "errors": errors,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "llm_latency_ms": round(self.latency_ms, 1),
            }
        )
