"""Configuration models for the events stage (configs/events.yaml, configs/matches.yaml)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from match_tagger.events.schema import Team


class TeamMappingConfig(BaseModel):
    """Cluster label -> home/away by kit colour."""

    crops_per_team: int = Field(40, ge=1)
    seed: int = 0
    palette: dict[str, tuple[int, int, int]] = Field(default_factory=dict)


class BallConfig(BaseModel):
    """Airborne flag and gap interpolation of the projected ball."""

    airborne_mode: Literal["off", "fixed", "calibrated"] = "fixed"
    airborne_size_ratio: float = Field(1.6, gt=1)
    calibration_near_m: float = Field(1.5, gt=0)
    calibration_min_samples: int = Field(10, ge=1)
    max_ground_speed_mps: float = Field(35.0, gt=0)
    ball_diameter_m: float = Field(0.22, gt=0)
    max_interp_gap_sec: float = Field(0.4, ge=0)


class PossessionConfig(BaseModel):
    """Who controls the ball."""

    control_radius_m: float = Field(1.5, gt=0)
    min_hold_sec: float = Field(0.24, ge=0)


class RulesConfig(BaseModel):
    """Possession transitions -> pass / ball_loss."""

    max_transition_sec: float = Field(3.0, gt=0)
    pass_min_ball_travel_m: float = Field(3.0, ge=0)
    intercept_min_dist_m: float = Field(5.0, ge=0)
    pass_time_fraction: float = Field(0.0, ge=0, le=1)


class DiagnosticsConfig(BaseModel):
    """Heuristics of the error analysis (not used by the rules themselves)."""

    observable_window_sec: float = Field(0.5, gt=0)
    track_break_max_gap_sec: float = Field(1.0, ge=0)
    track_break_max_dist_m: float = Field(2.0, ge=0)
    timing_near_sec: float = Field(5.0, gt=0)


class SplitConfig(BaseModel):
    """Evaluation protocol for tuning (documentation and input of make tune)."""

    method: str = "leave_one_clip_out"
    set_id: str = "dfl"
    clips: list[str] = Field(default_factory=list)


class EventsConfig(BaseModel):
    """Contents of configs/events.yaml."""

    team_mapping: TeamMappingConfig = TeamMappingConfig()
    ball: BallConfig = BallConfig()
    possession: PossessionConfig = PossessionConfig()
    rules: RulesConfig = RulesConfig()
    split: SplitConfig = SplitConfig()
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()

    def with_params(self, params: dict[str, Any]) -> "EventsConfig":
        """Return a copy with dotted overrides applied, e.g. {"possession.control_radius_m": 2}."""
        data = self.model_dump()
        for dotted, value in params.items():
            section, key = dotted.split(".")
            data[section][key] = value
        return EventsConfig.model_validate(data)


Side = Literal["left", "right"]


class MatchInfo(BaseModel):
    """Per-match metadata provided by the analyst.

    ``home_attacks``: the goal the home team attacks as seen in the video, for the whole clip
    or per period ({1: "left", 2: "right"}); None if unknown.
    """

    kits: dict[Team, str]
    home_attacks: Side | dict[int, Side] | None = None


class MatchesConfig(BaseModel):
    """Contents of configs/matches.yaml."""

    matches: dict[str, MatchInfo] = Field(default_factory=dict)
