"""Configuration model for configs/tracking.yaml."""

from pydantic import BaseModel

from match_tagger.tracking.teams import TeamConfig
from match_tagger.tracking.tracker import ByteTrackConfig


class TrackingConfig(BaseModel):
    """ByteTrack and team-classifier parameters."""

    bytetrack: ByteTrackConfig = ByteTrackConfig()
    team: TeamConfig = TeamConfig()
