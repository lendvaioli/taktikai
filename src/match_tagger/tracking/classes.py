"""One class per track: the detector's per-frame class can flicker (e.g. goalkeeper/player)."""

from collections import defaultdict
from collections.abc import Sequence

from match_tagger.detection.schema import DetectionRecord, ObjectClass


class ClassVotes:
    """Accumulates confidence-weighted class votes per track over every processed frame."""

    def __init__(self) -> None:
        """Start with no votes."""
        self.weights: dict[int, dict[ObjectClass, float]] = defaultdict(dict)

    def add(self, track_id: int, cls: ObjectClass, conf: float) -> None:
        """Add one sighting of ``track_id`` detected as ``cls`` with confidence ``conf``."""
        votes = self.weights[track_id]
        votes[cls] = votes.get(cls, 0.0) + conf

    def resolve(self) -> dict[int, ObjectClass]:
        """Return the class with the largest summed confidence per track.

        Ties go to the class seen first for that track.
        """
        return {tid: max(votes, key=votes.__getitem__) for tid, votes in self.weights.items()}


def apply_track_classes(
    records: Sequence[DetectionRecord], track_cls: dict[int, ObjectClass]
) -> list[DetectionRecord]:
    """Rewrite ``cls`` of tracked records to their track's class; re-validate changed rows."""
    out = []
    for r in records:
        cls = track_cls.get(r.track_id) if r.track_id is not None else None
        if cls is None or cls == r.cls:
            out.append(r)
        else:
            out.append(DetectionRecord.model_validate({**r.model_dump(), "cls": cls}))
    return out
