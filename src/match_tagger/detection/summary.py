"""Run summary of the detection stage (detection_summary.json)."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import median

from pydantic import BaseModel, Field

from match_tagger.detection.schema import DetectionRecord


class DetectionSummary(BaseModel):
    """Counts, coverage shares, ID-stability metrics and timings of one detection run."""

    match_id: str
    video: str
    device: str
    n_processed_frames: int = Field(ge=0, description="Frames run through detector and tracker.")
    n_sampled_frames: int = Field(ge=0, description="Frames written to outputs.")
    n_records: int = Field(ge=0)
    n_tracks: int = Field(ge=0)
    records_per_class: dict[str, int]
    player_team_share: float | None = Field(
        None, description="Share of player rows that received a team label."
    )
    ball_frame_share: float | None = Field(
        None, description="Share of sampled frames with a ball detection."
    )
    n_player_tracks: int = Field(0, ge=0, description="Distinct track ids with a player row.")
    median_player_track_sec: float | None = Field(
        None, description="Median over player tracks of last minus first t_sec of the track."
    )
    mean_labelled_players_per_frame: float | None = Field(
        None, description="Player rows with a team label, averaged over sampled frames."
    )
    timings_sec: dict[str, float] = {}


def track_stability(
    records: Sequence[DetectionRecord], n_sampled_frames: int
) -> dict[str, float | int | None]:
    """ID-stability metrics computed from player rows."""
    spans: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if r.cls == "player" and r.track_id is not None:
            spans[r.track_id].append(r.t_sec)
    labelled = sum(r.cls == "player" and r.team is not None for r in records)
    return {
        "n_player_tracks": len(spans),
        "median_player_track_sec": (
            median(max(t) - min(t) for t in spans.values()) if spans else None
        ),
        "mean_labelled_players_per_frame": (
            labelled / n_sampled_frames if n_sampled_frames else None
        ),
    }


def coverage(records: Sequence[DetectionRecord], n_sampled_frames: int) -> dict[str, object]:
    """Compute counts, shares and stability metrics from the final records."""
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
        **track_stability(records, n_sampled_frames),
    }
