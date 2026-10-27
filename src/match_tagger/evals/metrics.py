"""Counts and precision / recall / F1 with explicit handling of undefined values."""

from pydantic import BaseModel, Field


class Counts(BaseModel):
    """True positives, false positives and false negatives."""

    tp: int = Field(0, ge=0)
    fp: int = Field(0, ge=0)
    fn: int = Field(0, ge=0)

    def __add__(self, other: "Counts") -> "Counts":
        """Sum counts (used for micro-averaging)."""
        return Counts(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)


class MetricRow(Counts):
    """Counts plus metrics; a metric is None when it is undefined.

    Precision is undefined without predictions, recall without golden events. F1 is undefined
    if either is undefined, except that it is 0 when golden events exist and none was found.
    """

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    @classmethod
    def from_counts(cls, c: Counts) -> "MetricRow":
        """Compute precision, recall and F1 from counts."""
        n_pred, n_gold = c.tp + c.fp, c.tp + c.fn
        precision = c.tp / n_pred if n_pred else None
        recall = c.tp / n_gold if n_gold else None
        if precision is not None and recall is not None:
            total = precision + recall
            f1 = 2 * precision * recall / total if total else 0.0
        else:
            f1 = 0.0 if n_gold and c.tp == 0 else None
        return cls(**c.model_dump(), precision=precision, recall=recall, f1=f1)
