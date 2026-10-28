"""Data contracts of the report agent: the report itself and the backend-neutral chat types."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SupportingNumber(BaseModel):
    """A value behind a finding, copied from a tool result."""

    label: str = Field(min_length=1)
    value: float
    unit: str | None = None


class Finding(BaseModel):
    """One coaching finding: claim, evidence, clips and a training focus."""

    title: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    numbers: list[SupportingNumber] = Field(min_length=1)
    clip_ids: list[str] = Field(default_factory=list)
    training: str = Field(min_length=1)


class Report(BaseModel):
    """Structured output of the report agent (Hungarian text)."""

    match_id: str
    sample_note: str = Field(min_length=1)
    findings: list[Finding] = Field(min_length=2, max_length=3)


class ToolSpec(BaseModel):
    """A tool offered to the model: name, description and JSON schema of its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """One conversation turn in a backend-neutral form.

    ``raw`` keeps the backend's own assistant content (e.g. Anthropic content blocks, which
    must be sent back unchanged); it is never logged.
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    raw: Any = Field(None, exclude=True)


class LLMReply(BaseModel):
    """One model response with usage and latency."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float
    model: str
    raw: Any = Field(None, exclude=True)
