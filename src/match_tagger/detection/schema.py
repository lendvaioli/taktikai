"""Data contract for detections.parquet: one row per tracked object per sampled frame."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

ObjectClass = Literal["player", "goalkeeper", "referee", "ball"]
CLASSES: tuple[ObjectClass, ...] = ("player", "goalkeeper", "referee", "ball")
TEAMLESS: frozenset[str] = frozenset({"referee", "ball"})


class DetectionRecord(BaseModel):
    """One object in one sampled frame, in source-resolution pixel coordinates.

    The ball has no track id and at most one ball row exists per frame; select it by
    ``cls == "ball"``, never by ``track_id``. Referees and the ball never have a team.
    """

    frame: int = Field(ge=0)
    t_sec: float = Field(ge=0)
    track_id: int | None = Field(None, ge=0)
    cls: ObjectClass
    team: Literal[0, 1] | None = None
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "DetectionRecord":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        if self.cls in TEAMLESS and self.team is not None:
            raise ValueError(f"{self.cls} must not have a team")
        if self.cls == "ball" and self.track_id is not None:
            raise ValueError("ball must not have a track_id")
        if self.cls != "ball" and self.track_id is None:
            raise ValueError(f"{self.cls} must have a track_id")
        return self


ARROW_SCHEMA = pa.schema(
    [
        ("frame", pa.int32()),
        ("t_sec", pa.float64()),
        ("track_id", pa.int32()),
        ("cls", pa.string()),
        ("team", pa.int8()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("x2", pa.float32()),
        ("y2", pa.float32()),
        ("conf", pa.float32()),
    ]
)


def write_detections(records: Iterable[DetectionRecord], path: Path) -> int:
    """Write validated records to parquet with ``ARROW_SCHEMA``; return the row count."""
    rows = [r.model_dump() for r in records]
    columns = {f.name: [row[f.name] for row in rows] for f in ARROW_SCHEMA}
    table = pa.Table.from_pydict(columns, schema=ARROW_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table.num_rows


def read_detections(path: Path) -> pd.DataFrame:
    """Read detections.parquet; nullable columns use pandas nullable integer dtypes."""
    return pq.read_table(path).to_pandas(
        types_mapper={pa.int32(): pd.Int32Dtype(), pa.int8(): pd.Int8Dtype()}.get
    )
