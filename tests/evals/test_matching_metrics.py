import pytest

from match_tagger.evals.golden import GoldenClip
from match_tagger.evals.matching import match_events
from match_tagger.evals.metrics import Counts, MetricRow
from match_tagger.evals.run import evaluate_clip
from match_tagger.events.schema import Event

TOL = 2.0


def ev(id_, t, type_="pass", team="home"):
    return Event(id=id_, t_sec=t, type=type_, team=team)


GOLDEN = [ev("g1", 5.0), ev("g2", 12.0, "shot", "away"), ev("g3", 20.0, "ball_loss")]


def _clip(events):
    return GoldenClip(clip_id="c", video="v.mp4", fps=25, duration_sec=30, events=events)


def _overall(counts):
    return MetricRow.from_counts(sum(counts.values(), Counts()))


def test_perfect_prediction_gives_one():
    preds = [ev(f"p{i}", g.t_sec, g.type, g.team) for i, g in enumerate(GOLDEN)]
    row = _overall(evaluate_clip(_clip(GOLDEN), preds, TOL))
    assert (row.precision, row.recall, row.f1) == (1.0, 1.0, 1.0)


def test_empty_prediction_gives_zero_recall():
    row = _overall(evaluate_clip(_clip(GOLDEN), [], TOL))
    assert row.recall == 0.0
    assert row.precision is None  # undefined without predictions
    assert row.f1 == 0.0


def test_prediction_three_seconds_off_is_not_matched():
    result = match_events([ev("p1", 8.0)], [ev("g1", 5.0)], TOL)
    assert result.pairs == []
    assert (result.unmatched_predictions, result.unmatched_golden) == (["p1"], ["g1"])


def test_duplicate_predictions_yield_one_true_positive():
    counts = evaluate_clip(_clip([ev("g1", 5.0)]), [ev("p1", 5.2), ev("p2", 5.4)], TOL)
    assert counts["pass"] == Counts(tp=1, fp=1, fn=0)


def test_exactly_at_tolerance_is_matched():
    assert len(match_events([ev("p1", 7.0)], [ev("g1", 5.0)], TOL).pairs) == 1


def test_type_and_team_must_agree():
    golden = [ev("g1", 5.0)]
    assert match_events([ev("p1", 5.0, "shot")], golden, TOL).pairs == []
    assert match_events([ev("p1", 5.0, team="away")], golden, TOL).pairs == []


def test_greedy_closest_pair_wins():
    # p1 is 0.5 s from g2 and 1.5 s from g1; p2 is 1.0 s from g1. Greedy: (p1, g2), (p2, g1).
    golden = [ev("g1", 10.0), ev("g2", 12.0)]
    preds = [ev("p1", 11.5), ev("p2", 9.0)]
    pairs = {(p, g) for p, g, _ in match_events(preds, golden, TOL).pairs}
    assert pairs == {("p1", "g2"), ("p2", "g1")}


def test_metric_values_by_hand():
    row = MetricRow.from_counts(Counts(tp=3, fp=1, fn=2))
    assert row.precision == pytest.approx(0.75)
    assert row.recall == pytest.approx(0.6)
    assert row.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_undefined_metrics():
    assert MetricRow.from_counts(Counts()).f1 is None  # nothing at all
    no_golden = MetricRow.from_counts(Counts(fp=2))
    assert (no_golden.precision, no_golden.recall, no_golden.f1) == (0.0, None, None)
