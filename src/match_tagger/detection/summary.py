"""Run summary of the detection stage (detection_summary.json)."""

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field

from match_tagger.detection.schema import DetectionRecord


class DetectionSummary(BaseModel):
    """Counts, coverage shares and timings of one detection run."""

    match_id: str
    video: str
    device: str
    n_sampled_frames: int = Field(ge=0)
    n_records: int = Field(ge=0)
    n_tracks: int = Field(ge=0)
    records_per_class: dict[str, int]
    player_team_share: float | None = Field(
        None, description="Share of player rows that received a team label."
    )
    ball_frame_share: float | None = Field(
        None, description="Share of sampled frames with a ball detection."
    )
    timings_sec: dict[str, float] = {}


def coverage(records: Sequence[DetectionRecord], n_sampled_frames: int) -> dict[str, object]:
    """Compute counts and shares from the final records."""
    players = [r for r in records if r.cls == "player"]
    ball_frames = {r.frame for r in records if r.cls == "ball"}
    return {
        "n_sampled_frames": n_sampled_frames,
        "n_records": len(records),
        "n_tracks": len({r.track_id for r in records if r.track_id is not None}),
        "records_per_class": dict(Counter(r.cls for r in records)),
        "player_team_share": (
            sum(r.team is not None for r in players) / len(players) if players else None
        ),
        "ball_frame_share": len(ball_frames) / n_sampled_frames if n_sampled_frames else None,
    }
