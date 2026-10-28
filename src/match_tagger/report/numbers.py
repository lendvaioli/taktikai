"""Numbers in report text and in tool results, and the rounding-tolerant match between them.

A number written with d decimals matches a tool value v if it differs from v (or from 100 * v,
for shares written as percentages) by at most ``tolerance * 10**-d``: with the default 0.5
that is exactly "v rounded to the written precision". Hungarian decimal commas are accepted.
Numbers inside words or ids (U19, e0012) are not numbers of the text.
"""

import math
import re
from collections.abc import Iterable, Iterator

NUMBER = re.compile(r"(?<![\w.,])[-+]?\d+(?:[.,]\d+)?(?![\w])")


def numbers_in_text(text: str) -> list[str]:
    """Numbers as written in a text (sign and decimal comma or point kept)."""
    return NUMBER.findall(text)


def as_number(text: str) -> float | None:
    """A written number as float (decimal comma or point); None if it is not one."""
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
        value = as_number(node.strip())
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
    value = as_number(written)
    if value is None:
        return False
    decimals = len(re.split(r"[.,]", written)[1]) if re.search(r"[.,]", written) else 0
    limit = tolerance * 10**-decimals + 1e-9
    return any(abs(value - v) <= limit or abs(value - 100 * v) <= limit for v in known)
