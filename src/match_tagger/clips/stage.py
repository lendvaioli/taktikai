"""Clips stage: events.json + video -> one clip per event and clips/index.json."""

import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from match_tagger.clips.plan import (
    ClipPlan,
    ClipsConfig,
    Method,
    ffmpeg_args,
    params_hash,
    plan_clip,
)
from match_tagger.config import DefaultConfig, load_config
from match_tagger.events.schema import Event, EventsFile
from match_tagger.video.frames import read_meta

Runner = Callable[[list[str]], int]


@dataclass
class ClipsStageConfig:
    """All configs the clips stage reads."""

    paths: DefaultConfig
    clips: ClipsConfig

    @classmethod
    def load(cls, config_dir: Path) -> "ClipsStageConfig":
        """Load default and clips YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            clips=load_config("clips", ClipsConfig, config_dir),
        )


class ClipRecord(BaseModel):
    """One produced clip (clips/index.json)."""

    event_id: str
    t_sec: float
    type: str
    team: str
    path: str
    plan: ClipPlan
    method: Method
    params_hash: str


class ClipsIndex(BaseModel):
    """clips/index.json plus run statistics."""

    match_id: str
    clips: list[ClipRecord]
    methods: dict[str, int]
    skipped_existing: int
    seconds: float


def run_ffmpeg(args: list[str]) -> int:
    """Run ffmpeg; return its exit code."""
    return subprocess.run(args, capture_output=True, check=False).returncode


def keyframes(src: Path, ffprobe: str) -> list[float]:
    """Sorted keyframe times of the first video stream."""
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "csv=p=0",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(float(v.strip(", ")) for v in out.split() if v.strip(", "))


def cut(plan: ClipPlan, src: Path, dst: Path, cfg: ClipsConfig, run: Runner) -> Method:
    """Cut one clip; a failed stream copy falls back to re-encoding. Returns the method used."""
    if plan.method == "copy":
        if (
            run(ffmpeg_args(plan, src, dst, cfg, "copy")) == 0
            and dst.exists()
            and dst.stat().st_size
        ):
            return "copy"
        plan = plan.model_copy(update={"start": plan.requested_start})
    if run(ffmpeg_args(plan, src, dst, cfg, "reencode")) != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {dst.name}")
    return "reencode"


def make_clips(
    events: list[Event],
    src: Path,
    out_dir: Path,
    cfg: ClipsConfig,
    keys: list[float],
    duration: float,
    run: Runner = run_ffmpeg,
    previous: dict[str, ClipRecord] | None = None,
) -> tuple[list[ClipRecord], int]:
    """Cut every event's clip unless an identical one exists; return records and skip count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records, skipped = [], 0
    for event in events:
        plan = plan_clip(event.t_sec, duration, keys, cfg)
        digest, dst = params_hash(src, plan, cfg), out_dir / f"{event.id}.mp4"
        old = (previous or {}).get(event.id)
        if old is not None and old.params_hash == digest and dst.exists():
            records.append(old)
            skipped += 1
            continue
        method = cut(plan, src, dst, cfg, run)
        records.append(
            ClipRecord(
                event_id=event.id,
                t_sec=event.t_sec,
                type=event.type,
                team=event.team,
                path=dst.name,
                plan=plan,
                method=method,
                params_hash=digest,
            )
        )
    return records, skipped


def run_clips(video: Path, match_id: str, cfg: ClipsStageConfig) -> ClipsIndex:
    """Run the stage for the predicted events of a match; never modifies its inputs."""
    started = time.perf_counter()
    match_dir = cfg.paths.paths.processed_dir / match_id
    events = EventsFile.model_validate_json((match_dir / "events.json").read_text()).events
    meta = read_meta(video)
    out_dir, index_path = match_dir / "clips", match_dir / "clips" / "index.json"
    previous = {}
    if index_path.exists():
        previous = {
            r.event_id: r for r in ClipsIndex.model_validate_json(index_path.read_text()).clips
        }
    keys = keyframes(video, cfg.clips.ffprobe)
    records, skipped = make_clips(
        events, video, out_dir, cfg.clips, keys, meta.n_frames / meta.fps, previous=previous
    )
    index = ClipsIndex(
        match_id=match_id,
        clips=records,
        methods=dict(Counter(r.method for r in records)),
        skipped_existing=skipped,
        seconds=round(time.perf_counter() - started, 2),
    )
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index


def load_index(match_dir: Path) -> ClipsIndex | None:
    """clips/index.json of a match, if the clips stage has run."""
    path = match_dir / "clips" / "index.json"
    return ClipsIndex.model_validate(json.loads(path.read_text())) if path.exists() else None
