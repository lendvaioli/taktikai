import pytest

from match_tagger.evals.metrics import Counts
from match_tagger.evals.tune import (
    distance_to_start,
    fold_spread,
    grid_combos,
    held_out_total,
    leave_one_out,
    select_best,
)

CLIPS = ["a", "b", "c"]
COMBOS = [{"p": 1}, {"p": 2}, {"p": 3}]
# Pass counts per combination and clip (TP, FP, FN), chosen so the folds disagree:
# combo 0 is good on a and b, combo 1 is good on c, combo 2 is mediocre everywhere.
TABLE = [
    {"a": Counts(tp=4, fp=0, fn=0), "b": Counts(tp=4, fp=0, fn=0), "c": Counts(tp=0, fp=4, fn=4)},
    {"a": Counts(tp=0, fp=4, fn=4), "b": Counts(tp=1, fp=3, fn=3), "c": Counts(tp=4, fp=0, fn=0)},
    {"a": Counts(tp=2, fp=2, fn=2), "b": Counts(tp=2, fp=2, fn=2), "c": Counts(tp=2, fp=2, fn=2)},
]
NO_TIE = [0, 0, 0]


def test_grid_combos_are_sorted_and_complete():
    combos = grid_combos({"b": [1, 2], "a": [True, False]})
    assert combos == [
        {"a": True, "b": 1},
        {"a": True, "b": 2},
        {"a": False, "b": 1},
        {"a": False, "b": 2},
    ]


def test_distance_to_start_counts_grid_steps():
    grid = {"x": [1.0, 1.5, 2.0, 3.0], "y": [0, 3, 5]}
    assert distance_to_start({"x": 3.0, "y": 0}, {"x": 1.5, "y": 3}, grid) == 2 + 1
    assert distance_to_start({"x": 1.0, "y": 5}, {"x": 1.4, "y": 3}, grid) == 1 + 1  # 1.4 ~ 1.5


def test_folds_never_train_on_the_held_out_clip():
    folds = leave_one_out(TABLE, CLIPS, COMBOS, NO_TIE)
    # train {b, c}: combo 0 F1 = 4/8 -> 0.5, combo 1 F1 = 5/8 -> 0.625, combo 2 0.5 -> combo 1
    # train {a, c}: combo 0 0.5, combo 1 0.5, combo 2 0.5 -> tie -> lowest index (combo 0)
    # train {a, b}: combo 0 1.0 -> combo 0
    assert [(f.held_out, f.combo_index) for f in folds] == [("a", 1), ("b", 0), ("c", 0)]
    assert folds[0].held.recall == 0.0 and folds[1].held.f1 == 1.0 and folds[2].held.f1 == 0.0


def test_tiebreak_prefers_closest_to_start():
    assert select_best(TABLE, ["a", "c"], [5, 5, 0]) == 2


def test_held_out_micro_total_and_spread():
    folds = leave_one_out(TABLE, CLIPS, COMBOS, NO_TIE)
    total = held_out_total(folds)  # a: 0/4/4, b: 4/0/0, c: 0/4/4
    assert (total.tp, total.fp, total.fn) == (4, 8, 8)
    assert total.precision == pytest.approx(1 / 3) and total.recall == pytest.approx(1 / 3)
    spread = fold_spread(folds)  # F1 per fold: 0, 1, 0
    assert (spread.n, spread.min, spread.max) == (3, 0.0, 1.0)
    assert spread.mean == pytest.approx(1 / 3)
    assert spread.std == pytest.approx((2 / 9) ** 0.5)
