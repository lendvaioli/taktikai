"""Configuration models for the detection stage (configs/detection.yaml)."""

from pydantic import BaseModel, Field


class WeightsConfig(BaseModel):
    """Where detector weights live and how to fetch them when missing."""

    filename: str = "football-player-detection.pt"
    gdrive_id: str | None = None


class BallModelConfig(BaseModel):
    """Separate ball detector with sliced inference. Not implemented yet."""

    enabled: bool = False


class DebugRenderConfig(BaseModel):
    """Time window of the annotated debug video."""

    enabled: bool = True
    start_sec: float = Field(0.0, ge=0)
    duration_sec: float = Field(30.0, gt=0)


class DetectionConfig(BaseModel):
    """Detector backend, thresholds and class mapping."""

    backend: str = "ultralytics_yolo"
    weights: WeightsConfig = WeightsConfig()
    conf: float = Field(0.3, ge=0, le=1)
    imgsz: int = Field(1280, gt=0)
    nms_iou: float = Field(0.5, ge=0, le=1)
    device: str = "auto"
    class_map: dict[str, str] = Field(
        default_factory=lambda: {
            "player": "player",
            "goalkeeper": "goalkeeper",
            "referee": "referee",
            "ball": "ball",
        },
        description="Model class name -> pipeline class name.",
    )
    ball_model: BallModelConfig = BallModelConfig()
    debug: DebugRenderConfig = DebugRenderConfig()
