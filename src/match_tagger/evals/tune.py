"""Leave-one-clip-out tuning of event thresholds over a parameter grid.

All grid combinations are evaluated once per clip; folds then only select from that table:
the combination with the best training F1 for the target type (ties: closest to the start
values, then grid order) is evaluated on the held-out clip.
"""

import itertools
import statistics
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from match_tagger.evals.metrics import Counts, MetricRow

# table[combo index][clip id] -> counts of the target type
CountsTable = list[dict[str, Counts]]


def grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """All combinations of a grid of dotted parameter names, in a deterministic order."""
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def distance_to_start(
    combo: dict[str, Any], start: dict[str, Any], grid: dict[str, list[Any]]
) -> int:
    """Sum over parameters of how many grid steps a combination is from the start values."""
    total = 0
    for key, value in combo.items():
        values = grid[key]
        base = start[key]
        base_idx = (
            values.index(base)
            if base in values
            else min(range(len(values)), key=lambda i: abs(float(values[i]) - float(base)))
        )
        total += abs(values.index(value) - base_idx)
    return total


def summed(table_row: dict[str, Counts], clips: Sequence[str]) -> Counts:
    """Micro-sum of counts over clips."""
    return sum((table_row[c] for c in clips), Counts())


def select_best(table: CountsTable, clips: Sequence[str], tiebreak: Sequence[int]) -> int:
    """Index of the combination with the highest F1 on ``clips`` (undefined F1 ranks last)."""

    def key(i: int) -> tuple[float, int, int]:
        f1 = MetricRow.from_counts(summed(table[i], clips)).f1
        return (-(f1 if f1 is not None else -1.0), tiebreak[i], i)

    return min(range(len(table)), key=key)


class Fold(BaseModel):
    """One leave-one-clip-out fold."""

    held_out: str
    combo_index: int
    params: dict[str, Any]
    train: MetricRow
    held: MetricRow


def leave_one_out(
    table: CountsTable, clips: Sequence[str], combos: list[dict[str, Any]], tiebreak: Sequence[int]
) -> list[Fold]:
    """Tune on all clips but one, evaluate on that one, for every clip."""
    folds = []
    for held in clips:
        train = [c for c in clips if c != held]
        best = select_best(table, train, tiebreak)
        folds.append(
            Fold(
                held_out=held,
                combo_index=best,
                params=combos[best],
                train=MetricRow.from_counts(summed(table[best], train)),
                held=MetricRow.from_counts(table[best][held]),
            )
        )
    return folds


class Spread(BaseModel):
    """Spread of per-fold held-out F1 (folds with undefined F1 are left out)."""

    n: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None


def fold_spread(folds: Sequence[Fold]) -> Spread:
    """Mean, population std, min and max of the folds' held-out F1."""
    values = [f.held.f1 for f in folds if f.held.f1 is not None]
    if not values:
        return Spread(n=0, mean=None, std=None, min=None, max=None)
    return Spread(
        n=len(values),
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        min=min(values),
        max=max(values),
    )


def held_out_total(folds: Sequence[Fold]) -> MetricRow:
    """Micro-average over the held-out folds (each clip is held out exactly once)."""
    return MetricRow.from_counts(
        sum((Counts(**f.held.model_dump(include={"tp", "fp", "fn"})) for f in folds), Counts())
    )
