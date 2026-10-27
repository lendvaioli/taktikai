"""Data contract for positions.parquet: detections projected onto the pitch in metres."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

from match_tagger.detection.schema import TEAMLESS, ObjectClass


class PositionRecord(BaseModel):
    """One detection projected onto the pitch (metres, origin bottom-left; see geometry.py).

    Players, goalkeepers and referees are projected at the bottom-centre of their box (the
    foot point); the ball at the centre of its box. The projection assumes every point lies on
    the ground plane, so a ball in the air lands in the wrong place, further from the camera.
    ``x_m``/``y_m`` are None exactly when ``h_quality`` is ``missing``. Points more than the
    configured margin outside the pitch are flagged with ``out_of_pitch``, never clipped.
    """

    frame: int = Field(ge=0)
    t_sec: float = Field(ge=0)
    track_id: int | None = Field(None, ge=0)
    cls: ObjectClass
    team: Literal[0, 1] | None = None
    x_m: float | None = None
    y_m: float | None = None
    h_quality: Literal["ok", "reused", "missing"]
    out_of_pitch: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> "PositionRecord":
        has_xy = self.x_m is not None and self.y_m is not None
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError("x_m and y_m must both be set or both be None")
        if has_xy == (self.h_quality == "missing"):
            raise ValueError("x_m/y_m must be None exactly when h_quality is missing")
        if self.out_of_pitch and not has_xy:
            raise ValueError("out_of_pitch requires a position")
        if self.cls in TEAMLESS and self.team is not None:
            raise ValueError(f"{self.cls} must not have a team")
        return self


ARROW_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("t_sec", pa.float64()),
        ("track_id", pa.int32()),
        ("cls", pa.string()),
        ("team", pa.int8()),
        ("x_m", pa.float32()),
        ("y_m", pa.float32()),
        ("h_quality", pa.string()),
        ("out_of_pitch", pa.bool_()),
    ]
)


def write_positions(records: Iterable[PositionRecord], path: Path) -> int:
    """Write validated records to parquet with ``ARROW_SCHEMA``; return the row count."""
    rows = [r.model_dump() for r in records]
    columns = {f.name: [row[f.name] for row in rows] for f in ARROW_SCHEMA}
    table = pa.Table.from_pydict(columns, schema=ARROW_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table.num_rows


def read_positions(path: Path) -> pd.DataFrame:
    """Read positions.parquet with pandas nullable integer dtypes."""
    return pq.read_table(path).to_pandas(
        types_mapper={pa.int32(): pd.Int32Dtype(), pa.int8(): pd.Int8Dtype()}.get
    )
