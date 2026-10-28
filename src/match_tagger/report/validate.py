"""Report validator: schema-valid reports are checked against the tool results of their run.

1. Every number in the free text (sample note, titles, claims, training, labels) must appear in
   some tool result (``numbers.matches``).
2. Every supporting number must equal the field its ``source`` names (``semantics``).
3. Comparisons must have the right direction; no English words, metric keys or untranslated
   team names (``semantics``).
4. Clip ids must have been returned by get_clips; with video every finding cites one.
"""

from dataclasses import dataclass, field

from match_tagger.report.config import SemanticsConfig
from match_tagger.report.numbers import matches, numbers_in_text, tool_values
from match_tagger.report.schema import Report, ToolCall
from match_tagger.report.semantics import (
    Normalize,
    check_comparisons,
    english_errors,
    source_error,
)


@dataclass
class ValidationInput:
    """What a report is checked against: the run's tool calls and the match."""

    calls: list[tuple[ToolCall, dict]]
    returned_clips: set[str]
    has_video: bool
    match_id: str
    normalize: Normalize
    tolerance: float = 0.5
    semantics: SemanticsConfig = field(default_factory=SemanticsConfig)
    foreign_names: list[str] = field(default_factory=list)


@dataclass
class Validation:
    """Problems of a report (accepted if empty) and comparisons that could not be checked."""

    errors: list[str]
    unchecked_comparisons: int = 0


def report_texts(report: Report) -> list[tuple[str, str]]:
    """(where, text) of every free-text field of a report."""
    texts = [("sample_note", report.sample_note)]
    for i, f in enumerate(report.findings, 1):
        texts += [(f"finding {i} title", f.title), (f"finding {i} claim", f.claim)]
        texts += [(f"finding {i} training", f.training)]
        texts += [(f"finding {i} numbers.label", n.label) for n in f.numbers]
        texts += [(f"finding {i} numbers.unit", n.unit) for n in f.numbers if n.unit]
    return texts


def _clip_errors(report: Report, v: ValidationInput) -> list[str]:
    errors = []
    for i, f in enumerate(report.findings, 1):
        unknown = [c for c in f.clip_ids if c not in v.returned_clips]
        if unknown:
            errors.append(f"finding {i}: clip ids not returned by get_clips: {unknown}")
        if v.has_video and not f.clip_ids:
            errors.append(f"finding {i}: cite at least one clip id from get_clips")
        if not v.has_video and f.clip_ids:
            errors.append(f"finding {i}: this match has no video, leave clip_ids empty")
    return errors


def validate_report(report: Report, v: ValidationInput) -> Validation:
    """All checks; the report is accepted if no error is returned."""
    known = list(tool_values([result for _, result in v.calls]))
    errors, unchecked = [], 0
    for where, text in report_texts(report):
        errors += [
            f"{where}: {n} does not appear in any tool result"
            for n in numbers_in_text(text)
            if not matches(n, known, v.tolerance)
        ]
        wrong, skipped = check_comparisons(text, v.semantics)
        errors += [f"{where}: comparison {e}" for e in wrong]
        unchecked += skipped
        found = english_errors(text, v.semantics.english_words, v.foreign_names)
        errors += [f"{where}: {e}" for e in found]
    for i, f in enumerate(report.findings, 1):
        errors += [
            f"finding {i} numbers: {e}"
            for n in f.numbers
            if (e := source_error(n, v.calls, v.normalize))
        ]
    if report.match_id != v.match_id:
        errors.append(f"match_id must be {v.match_id!r}")
    return Validation(errors=errors + _clip_errors(report, v), unchecked_comparisons=unchecked)
