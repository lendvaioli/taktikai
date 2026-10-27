"""Configuration model for the pitch stage (configs/pitch.yaml)."""

from pydantic import BaseModel, Field

from match_tagger.detection.config import WeightsConfig
from match_tagger.pitch.geometry import PitchDims
from match_tagger.pitch.homography import HomographyConfig


class PitchSize(BaseModel):
    """Per-match pitch size override."""

    length_m: float = Field(gt=0)
    width_m: float = Field(gt=0)


class MinimapConfig(BaseModel):
    """Time window and layout of the side-by-side minimap video."""

    enabled: bool = True
    start_sec: float = Field(0.0, ge=0)
    duration_sec: float = Field(30.0, gt=0)
    panel_height: int = Field(540, gt=0)
    overlay: bool = Field(True, description="Draw keypoints and projected pitch lines.")


class PitchConfig(BaseModel):
    """Keypoint model, homography handling, pitch size and minimap settings."""

    backend: str = "ultralytics_pose"
    weights: WeightsConfig = WeightsConfig(filename="football-pitch-detection.pt")
    imgsz: int = Field(640, gt=0)
    device: str = "auto"
    homography: HomographyConfig = HomographyConfig()
    smooth_window_sec: float = Field(0.4, ge=0)
    max_reuse_sec: float = Field(2.0, ge=0)
    out_margin_m: float = Field(5.0, ge=0)
    max_speed_mps: float = Field(12.0, gt=0, description="Steps faster than this count as jumps.")
    dims: PitchDims = PitchDims()
    match_sizes: dict[str, PitchSize] = Field(default_factory=dict)
    minimap: MinimapConfig = MinimapConfig()

    def dims_for(self, match_id: str) -> PitchDims:
        """Return the pitch dimensions for a match, applying its size override if any."""
        size = self.match_sizes.get(match_id)
        return self.dims if size is None else self.dims.model_copy(update=size.model_dump())
