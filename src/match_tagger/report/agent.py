"""The report agent: a plain loop of at most ``max_steps`` LLM calls, no agent framework.

Each step is one model call. Data tool calls are executed and their results appended; a
``submit_report`` call is parsed and validated (schema, numbers and their sources, comparisons,
language, clip ids; see ``validate``). A rejected
report gets one retry with the validator's errors; the last step offers only
``submit_report``. The loop never returns an unvalidated report.
"""

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from match_tagger.report.backends import LLMBackend
from match_tagger.report.config import SemanticsConfig
from match_tagger.report.log import RunLog, ToolRecord
from match_tagger.report.schema import Message, Report, ToolCall
from match_tagger.report.tools import SUBMIT_SPEC, ToolBox, normalize_arguments, tool_specs
from match_tagger.report.validate import Validation, ValidationInput, validate_report

Status = Literal["ok", "rejected", "max_steps", "error"]
NUDGE = "Use a tool, or call submit_report with the final report."


class AgentSettings(BaseModel):
    """Loop limits and validator settings."""

    match_id: str
    has_video: bool
    max_steps: int = 8
    retries: int = 1
    tolerance: float = 0.5
    semantics: SemanticsConfig = Field(default_factory=SemanticsConfig)
    foreign_names: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Outcome of one run."""

    status: Status
    report: Report | None
    errors: list[str]
    steps: int


def parse_submission(call: ToolCall) -> tuple[Report | None, list[str]]:
    """The submitted report; accepts the fields directly or wrapped in a "report" key."""
    args: Any = call.arguments.get("report", call.arguments)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            return None, [f"report is not valid JSON: {exc}"]
    try:
        return Report.model_validate(args), []
    except ValidationError as exc:
        return None, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def _run_tools(calls: list[ToolCall], box: ToolBox, messages: list[Message]) -> list[ToolRecord]:
    records = []
    for call in calls:
        started = time.perf_counter()
        result = box.run(call)
        content = json.dumps(result, ensure_ascii=False)
        records.append(
            ToolRecord(
                name=call.name,
                arguments=call.arguments,
                ms=round((time.perf_counter() - started) * 1000, 2),
                result_chars=len(content),
                error=result.get("error"),
            )
        )
        messages.append(
            Message(role="tool", content=content, tool_call_id=call.id, tool_name=call.name)
        )
    return records


def _check(call: ToolCall, box: ToolBox, s: AgentSettings) -> tuple[Report | None, Validation]:
    report, errors = parse_submission(call)
    if report is None:
        return None, Validation(errors=errors)
    check = ValidationInput(
        calls=box.calls,
        returned_clips=box.returned_clips,
        has_video=s.has_video,
        match_id=s.match_id,
        normalize=normalize_arguments,
        tolerance=s.tolerance,
        semantics=s.semantics,
        foreign_names=s.foreign_names,
    )
    return report, validate_report(report, check)


def feedback(errors: list[str], has_video: bool) -> str:
    """Rejection message: the problems, then the rules every report must follow."""
    clips = (
        "every finding cites at least one clip id that get_clips returned in this run"
        if has_video
        else "this match has no video: every finding has an empty clip_ids list"
    )
    return (
        "REJECTED. Fix these problems and call submit_report again:\n- "
        + "\n- ".join(errors)
        + "\nRules for the whole report:\n"
        "- 2 or 3 findings, each with title, claim, numbers, clip_ids and training\n"
        f"- {clips}\n"
        "- every number is copied from a tool result; leave out numbers no tool returned\n"
        "- every entry of numbers has a source: the tool, its exact arguments and the path of the "
        "field; value equals that field exactly\n"
        "- Hungarian only: Hungarian labels instead of metric keys, team_name_hu for teams; "
        "comparisons (több, kevesebb, magasabb, alacsonyabb) must match the numbers"
    )


def run_agent(
    backend: LLMBackend, system: str, user: str, box: ToolBox, s: AgentSettings, log: RunLog
) -> AgentResult:
    """Run the loop until a report is accepted, rejected twice, or the steps run out."""
    messages = [Message(role="user", content=user)]
    retries_left = s.retries
    for step in range(1, s.max_steps + 1):
        tools = [SUBMIT_SPEC] if step == s.max_steps else [*tool_specs(), SUBMIT_SPEC]
        try:
            reply = backend.chat(system, messages, tools)
        except (OSError, RuntimeError) as exc:  # timeout, connection, refusal
            reason = f"backend error at step {step}: {type(exc).__name__}: {exc}"
            log.end("error", step, [reason])
            return AgentResult(status="error", report=None, errors=[reason], steps=step)
        messages.append(
            Message(
                role="assistant", content=reply.text, tool_calls=reply.tool_calls, raw=reply.raw
            )
        )
        submit = next((c for c in reply.tool_calls if c.name == "submit_report"), None)
        records = _run_tools([c for c in reply.tool_calls if c is not submit], box, messages)
        if submit is None:
            log.step(step, reply, records, None)
            if not reply.tool_calls:
                messages.append(Message(role="user", content=NUDGE))
            continue
        report, validation = _check(submit, box, s)
        errors = validation.errors
        log.step(step, reply, records, errors, validation.unchecked_comparisons)
        if not errors:
            log.end("ok", step, [])
            return AgentResult(status="ok", report=report, errors=[], steps=step)
        if retries_left == 0:
            log.end("rejected", step, errors)
            return AgentResult(status="rejected", report=None, errors=errors, steps=step)
        retries_left -= 1
        messages.append(
            Message(
                role="tool",
                content=feedback(errors, s.has_video),
                tool_call_id=submit.id,
                tool_name=submit.name,
            )
        )
    log.end("max_steps", s.max_steps, ["no accepted report within max_steps"])
    return AgentResult(
        status="max_steps",
        report=None,
        errors=["no accepted report within max_steps"],
        steps=s.max_steps,
    )
