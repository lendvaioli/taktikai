"""Configuration models of the metrics stage (configs/metrics.yaml)."""

from pydantic import BaseModel, Field


class ReliabilityConfig(BaseModel):
    """Sample-size levels: very_low below either very_low bound, low below low_max_min."""

    very_low_max_min: float = Field(10.0, gt=0)
    very_low_max_events: int = Field(30, ge=0)
    low_max_min: float = Field(45.0, gt=0)


class PassNetworkConfig(BaseModel):
    """Which edges of a pass network are reported."""

    min_edge_count: int = Field(1, ge=1)
    max_edges: int = Field(10, ge=1)


class PressingConfig(BaseModel):
    """PPDA-style proxy: area where the opponent's passes and ball losses are counted."""

    zone_max_x_share: float = Field(0.6, gt=0, le=1)


class MetricsConfig(BaseModel):
    """Contents of configs/metrics.yaml."""

    reliability: ReliabilityConfig = ReliabilityConfig()
    pass_network: PassNetworkConfig = PassNetworkConfig()
    pressing: PressingConfig = PressingConfig()
