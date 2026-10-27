"""One-to-one matching of predicted to golden events."""

from collections.abc import Sequence
from dataclasses import dataclass

from match_tagger.events.schema import Event


@dataclass(frozen=True)
class MatchResult:
    """Matched (prediction id, golden id, |dt|) pairs and the ids left unmatched."""

    pairs: list[tuple[str, str, float]]
    unmatched_predictions: list[str]
    unmatched_golden: list[str]


def match_events(
    predictions: Sequence[Event], golden: Sequence[Event], tolerance_sec: float
) -> MatchResult:
    """Greedily match events of the same type and team within ``tolerance_sec``.

    Candidate pairs are sorted by time difference (ties: golden time, golden id, prediction
    id) and accepted when neither event is matched yet, so each event matches at most once.
    Call per clip: events from different clips must never be compared.
    """
    candidates = sorted(
        (abs(p.t_sec - g.t_sec), g.t_sec, g.id, p.id)
        for p in predictions
        for g in golden
        if p.type == g.type and p.team == g.team and abs(p.t_sec - g.t_sec) <= tolerance_sec + 1e-9
    )
    used_pred: set[str] = set()
    used_gold: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    for dt, _, gold_id, pred_id in candidates:
        if pred_id in used_pred or gold_id in used_gold:
            continue
        used_pred.add(pred_id)
        used_gold.add(gold_id)
        pairs.append((pred_id, gold_id, dt))
    return MatchResult(
        pairs=pairs,
        unmatched_predictions=[p.id for p in predictions if p.id not in used_pred],
        unmatched_golden=[g.id for g in golden if g.id not in used_gold],
    )
