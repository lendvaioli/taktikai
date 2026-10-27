"""Plain-text tables for evaluation runs and run comparisons."""

from pydantic import BaseModel

from match_tagger.evals.metrics import MetricRow
from match_tagger.evals.run import EvalRun, SetResult
from match_tagger.events.schema import EVENT_TYPES

METRICS = ("precision", "recall", "f1")
ROWS = (*EVENT_TYPES, "overall")


def fmt(value: float | None, signed: bool = False) -> str:
    """Format a metric; undefined values are shown as '-'."""
    if value is None:
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _rows(result: SetResult) -> dict[str, MetricRow]:
    return {**result.per_type, "overall": result.overall}


def set_table(result: SetResult) -> str:
    """Table of TP / FP / FN / P / R / F1 per type and overall for one golden set."""
    lines = [
        f"Golden set: {result.set_id} ({result.n_clips_evaluated} complete clips)",
        f"{'type':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}",
    ]
    for name, row in _rows(result).items():
        lines.append(
            f"{name:<10} {row.tp:>4} {row.fp:>4} {row.fn:>4} "
            f"{fmt(row.precision):>6} {fmt(row.recall):>6} {fmt(row.f1):>6}"
        )
    if result.skipped_incomplete:
        lines.append(f"skipped (incomplete): {', '.join(result.skipped_incomplete)}")
    if result.missing_predictions:
        lines.append(f"no predictions for: {', '.join(result.missing_predictions)}")
    return "\n".join(lines)


class CompareRow(BaseModel):
    """One metric of one row (type or overall) of one golden set in runs A and B."""

    set_id: str
    row: str
    metric: str
    a: float | None
    b: float | None
    diff: float | None


def compare_runs(a: EvalRun, b: EvalRun) -> list[CompareRow]:
    """Per golden set, row and metric: A, B and B - A (None if either side is missing)."""
    sets_a = {s.set_id: _rows(s) for s in a.sets}
    sets_b = {s.set_id: _rows(s) for s in b.sets}
    out = []
    for set_id in sorted(sets_a.keys() | sets_b.keys()):
        for row in ROWS:
            for metric in METRICS:
                va = getattr(sets_a[set_id][row], metric) if set_id in sets_a else None
                vb = getattr(sets_b[set_id][row], metric) if set_id in sets_b else None
                diff = vb - va if va is not None and vb is not None else None
                out.append(CompareRow(set_id=set_id, row=row, metric=metric, a=va, b=vb, diff=diff))
    return out


def compare_table(rows: list[CompareRow]) -> str:
    """Render ``compare_runs`` output as a table."""
    lines = [f"{'set':<10} {'type':<10} {'metric':<10} {'A':>7} {'B':>7} {'B-A':>7}"]
    for r in rows:
        lines.append(
            f"{r.set_id:<10} {r.row:<10} {r.metric:<10} "
            f"{fmt(r.a):>7} {fmt(r.b):>7} {fmt(r.diff, signed=True):>7}"
        )
    return "\n".join(lines)
