"""ByteTrack multi-object tracking via supervision."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import supervision as sv


class ByteTrackConfig(BaseModel):
    """ByteTrack parameters from configs/tracking.yaml."""

    track_activation_threshold: float = Field(0.25, ge=0, le=1)
    lost_track_seconds: float = Field(2.0, gt=0)
    minimum_matching_threshold: float = Field(0.8, ge=0, le=1)
    minimum_consecutive_frames: int = Field(3, ge=1)


def build_tracker(cfg: ByteTrackConfig, frame_rate: float) -> "sv.ByteTrack":
    """Create a ByteTrack instance for frames sampled at ``frame_rate``.

    supervision computes ``max_time_lost = frame_rate / 30 * lost_track_buffer`` frames, so
    the buffer is set to ``30 * lost_track_seconds`` to keep tracks for that many seconds.
    """
    import supervision as sv

    return sv.ByteTrack(
        track_activation_threshold=cfg.track_activation_threshold,
        lost_track_buffer=round(30 * cfg.lost_track_seconds),
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        frame_rate=frame_rate,
        minimum_consecutive_frames=cfg.minimum_consecutive_frames,
    )


def track(tracker: "sv.ByteTrack", detections: "sv.Detections") -> "sv.Detections":
    """Update the tracker and drop detections whose track is not yet confirmed."""
    tracked = tracker.update_with_detections(detections)
    return tracked[tracked.tracker_id >= 0]
