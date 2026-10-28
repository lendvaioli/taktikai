"""Shared event contract for golden annotations and pipeline predictions."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

EventType = Literal["pass", "ball_loss", "shot", "set_piece"]
EVENT_TYPES: tuple[EventType, ...] = ("pass", "ball_loss", "shot", "set_piece")
Team = Literal["home", "away"]
Outcome = Literal["complete", "incomplete", "goal", "saved", "blocked", "off_target"]
"""Pass: complete / incomplete. Shot: goal / saved / blocked / off_target."""


class Event(BaseModel):
    """One match event at a single instant, relative to the start of its clip (or match).

    ``team`` is the acting team: the passer's, the shooter's, the team taking the set piece, or
    the team that lost the ball. Golden events leave ``confidence`` empty. The optional fields
    after ``confidence`` are filled where the source knows them: the pipeline sets a pass's end
    point and outcome; the StatsBomb adapter also sets players, xG and period.
    """

    id: str = Field(min_length=1)
    t_sec: float = Field(ge=0)
    type: EventType
    team: Team
    x_m: float | None = None
    y_m: float | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    end_x_m: float | None = None
    end_y_m: float | None = None
    outcome: Outcome | None = None
    player: str | None = None
    recipient: str | None = None
    xg: float | None = Field(None, ge=0, le=1)
    period: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _check_location(self) -> "Event":
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError("x_m and y_m must both be set or both be None")
        if (self.end_x_m is None) != (self.end_y_m is None):
            raise ValueError("end_x_m and end_y_m must both be set or both be None")
        return self


class EventsFile(BaseModel):
    """Pipeline output for one clip: data/processed/<clip_id>/events.json."""

    clip_id: str
    events: list[Event]
