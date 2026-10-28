"""Numeric validator: every number in a report must appear in a tool result of the same run.

A number written with d decimals matches a tool value v if it differs from v (or from 100 * v,
for shares written as percentages) by at most ``tolerance * 10**-d``: with the default 0.5
that is exactly "v rounded to the written precision". Hungarian decimal commas are accepted.
Numbers inside words or ids (U19, e0012) are not numbers of the text.
"""

import math
import re
from collections.abc import Iterable, Iterator
from typing import Any

from match_tagger.report.schema import Report

NUMBER = re.compile(r"(?<![\w.,])[-+]?\d+(?:[.,]\d+)?(?![\w])")


def numbers_in_text(text: str) -> list[str]:
    """Numbers as written in a text (sign and decimal comma or point kept)."""
    return NUMBER.findall(text)


def _as_number(text: str) -> float | None:
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def tool_values(node: object) -> Iterator[float]:
    """Every number in a tool result: numeric values, numeric strings and numeric dict keys."""
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, int | float):
        yield float(node)
    elif isinstance(node, str):
        value = _as_number(node.strip())
        if value is not None:
            yield value
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from tool_values(k)
            yield from tool_values(v)
    elif isinstance(node, list | tuple):
        for v in node:
            yield from tool_values(v)


def matches(written: str, known: Iterable[float], tolerance: float) -> bool:
    """True if the written number equals a known value (or its percentage) within tolerance."""
    value = _as_number(written)
    if value is None:
        return False
    decimals = len(re.split(r"[.,]", written)[1]) if re.search(r"[.,]", written) else 0
    limit = tolerance * 10**-decimals + 1e-9
    return any(abs(value - v) <= limit or abs(value - 100 * v) <= limit for v in known)


def report_numbers(report: Report) -> list[tuple[str, str]]:
    """(where, number as written) for every number in the report's text and values."""
    found = [("sample_note", n) for n in numbers_in_text(report.sample_note)]
    for i, f in enumerate(report.findings, 1):
        texts = [("title", f.title), ("claim", f.claim), ("training", f.training)]
        texts += [("numbers.label", n.label) for n in f.numbers]
        found += [(f"finding {i} {where}", n) for where, t in texts for n in numbers_in_text(t)]
        found += [(f"finding {i} numbers.value", f"{n.value:g}") for n in f.numbers]
    return found


def validate_report(
    report: Report,
    outputs: list[Any],
    returned_clips: set[str],
    has_video: bool,
    match_id: str,
    tolerance: float,
) -> list[str]:
    """Problems of a report (empty if it is accepted)."""
    known = list(tool_values(outputs))
    errors = [
        f"{where}: {number} does not appear in any tool result"
        for where, number in report_numbers(report)
        if not matches(number, known, tolerance)
    ]
    if report.match_id != match_id:
        errors.append(f"match_id must be {match_id!r}")
    for i, f in enumerate(report.findings, 1):
        unknown = [c for c in f.clip_ids if c not in returned_clips]
        if unknown:
            errors.append(f"finding {i}: clip ids not returned by get_clips: {unknown}")
        if has_video and not f.clip_ids:
            errors.append(f"finding {i}: cite at least one clip id from get_clips")
        if not has_video and f.clip_ids:
            errors.append(f"finding {i}: this match has no video, leave clip_ids empty")
    return errors
