"""Rule-based semantic checks of a report: comparison direction, English words, number sources.

- Comparisons: a sentence (or a clause after ", de" / ", bár" / ...) with exactly one
  comparative word ("több", "kevesebb", ...) and exactly two numbers is checked: the first
  number is the subject ("51, ami kevesebb, mint 54").
  "nem" before the word flips the direction, "képest" swaps the numbers ("a 54-hez képest 51
  kevesebb"). Other sentences with a comparative word cannot be checked and are only counted.
- English: whole words from a configured list, metric keys (words with an underscore) and team
  names that have a different Hungarian label.
- Sources: every supporting number names a tool call of the run and a field path; the value
  must equal that field exactly. A rejection lists the paths where the value does occur.
"""

import re
from collections.abc import Callable, Iterable
from typing import Any

from match_tagger.report.config import SemanticsConfig
from match_tagger.report.numbers import as_number, numbers_in_text
from match_tagger.report.schema import NumberSource, SupportingNumber, ToolCall

SENTENCE_END = re.compile(r"(?<!\d)[.!?;](?!\d)|\n")
WORD = re.compile(r"[^\W\d_]+")
KEY = re.compile(r"\b\w+_\w+\b")
Normalize = Callable[[str, dict[str, Any]], dict[str, Any] | None]


def sentences(text: str, clause_breaks: Iterable[str] = ()) -> list[str]:
    """Sentences of a text, also split before ", <word>" for each clause-break word.

    A decimal point or comma does not end a sentence; a comma alone does not either
    ("51, ami több, mint 54" stays one clause).
    """
    parts = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    words = "|".join(re.escape(w) for w in clause_breaks)
    if not words:
        return parts
    split = re.compile(rf",\s+(?=(?:{words})\b)", re.IGNORECASE)
    return [c.strip() for s in parts for c in split.split(s) if c.strip()]


def _comparatives(words: list[str], cfg: SemanticsConfig) -> list[tuple[str, int]]:
    """(word, +1 greater / -1 less) for every comparative word, negation applied."""
    found = []
    for i, w in enumerate(words):
        if any(w.startswith(x) for x in cfg.not_comparative):
            continue
        sign = 1 if any(w.startswith(s) for s in cfg.greater_stems) else 0
        sign = -1 if any(w.startswith(s) for s in cfg.less_stems) else sign
        if sign:
            negated = i > 0 and words[i - 1] == cfg.negation
            found.append((w, -sign if negated else sign))
    return found


def check_comparisons(text: str, cfg: SemanticsConfig) -> tuple[list[str], int]:
    """Errors for comparisons with the wrong direction, and the number of unchecked ones."""
    errors, unchecked = [], 0
    for sentence in sentences(text, cfg.clause_breaks):
        words = [w.lower() for w in WORD.findall(sentence)]
        found = _comparatives(words, cfg)
        if not found:
            continue
        numbers = numbers_in_text(sentence)
        if len(found) != 1 or len(numbers) != 2:
            unchecked += 1
            continue
        first, second = (as_number(n) for n in numbers)
        if cfg.swap_word in words:
            first, second = second, first
        word, sign = found[0]
        if not (first > second if sign > 0 else first < second):
            errors.append(f'"{sentence}": "{word}" does not hold for {numbers[0]} and {numbers[1]}')
    return errors, unchecked


def english_errors(text: str, words: Iterable[str], foreign_names: Iterable[str]) -> list[str]:
    """English words, metric keys and untranslated team names in a text."""
    vocabulary = {w.lower() for w in words}
    found = [f'metric key "{k}"' for k in KEY.findall(text)]
    found += [f'English word "{w}"' for w in WORD.findall(text) if w.lower() in vocabulary]
    found += [
        f'team name "{name}" (use the Hungarian label)'
        for name in foreign_names
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
    ]
    return list(dict.fromkeys(found))


def resolve(node: object, path: str) -> object:
    """The value at a dot-separated path (list items by index); KeyError if there is none."""
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif (
            isinstance(node, list)
            and part.lstrip("-").isdigit()
            and -len(node) <= int(part) < len(node)
        ):
            node = node[int(part)]
        else:
            raise KeyError(part)
    return node


def _same(a: object, value: float) -> bool:
    is_number = isinstance(a, int | float) and not isinstance(a, bool)
    return is_number and abs(a - value) <= 1e-9 * max(1.0, abs(a))


def paths_with_value(node: object, value: float, prefix: str = "") -> list[str]:
    """Every dot path in a tool result whose field equals ``value`` exactly."""
    if _same(node, value):
        return [prefix]
    items = (
        node.items()
        if isinstance(node, dict)
        else enumerate(node)
        if isinstance(node, list)
        else []
    )
    found = []
    for key, child in items:
        found += paths_with_value(child, value, f"{prefix}.{key}" if prefix else str(key))
    return found


def value_hint(value: float, calls: list[tuple[ToolCall, dict]], limit: int = 5) -> str:
    """Where in this run's tool results the value occurs (at most ``limit`` places)."""
    places = [
        NumberSource(tool=c.name, arguments=c.arguments, path=p).describe()
        for c, result in calls
        if c.name in ("get_metric", "get_events", "get_clips")
        for p in paths_with_value(result, value)
    ]
    if not places:
        return f"; {value:g} occurs in no tool result of this run"
    more = f" (+{len(places) - limit} more)" if len(places) > limit else ""
    return f"; {value:g} is found at: " + "; ".join(places[:limit]) + more


def source_error(
    number: SupportingNumber, calls: list[tuple[ToolCall, dict]], normalize: Normalize
) -> str | None:
    """Why a supporting number does not match its source field (None if it does), with a hint."""
    problem = _source_problem(number, calls, normalize)
    return None if problem is None else problem + value_hint(number.value, calls)


def _source_problem(
    number: SupportingNumber, calls: list[tuple[ToolCall, dict]], normalize: Normalize
) -> str | None:
    src = number.source
    wanted = normalize(src.tool, src.arguments)
    results = [
        r for c, r in calls if c.name == src.tool and normalize(c.name, c.arguments) == wanted
    ]
    if wanted is None or not results:
        return f'"{number.label}": source call {src.describe()} was not made in this run'
    actual: object = None
    for result in results:
        try:
            actual = resolve(result, src.path)
        except KeyError:
            continue
        if _same(actual, number.value):
            return None
    if actual is None:
        return f'"{number.label}": {src.describe()} has no field "{src.path}"'
    return f'"{number.label}": value {number.value:g} but {src.describe()} is {actual}'
