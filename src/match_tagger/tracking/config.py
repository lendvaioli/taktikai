"""Configuration model for configs/tracking.yaml."""

from pydantic import BaseModel, Field

from match_tagger.tracking.teams import TeamConfig
from match_tagger.tracking.tracker import ByteTrackConfig


class ClassVoteConfig(BaseModel):
    """How one class per track is chosen from per-frame detector classes."""

    goalkeeper_min_share: float = Field(0.3, gt=0, le=1)


class TrackingConfig(BaseModel):
    """ByteTrack, class-vote and team-classifier parameters."""

    bytetrack: ByteTrackConfig = ByteTrackConfig()
    class_vote: ClassVoteConfig = ClassVoteConfig()
    team: TeamConfig = TeamConfig()
