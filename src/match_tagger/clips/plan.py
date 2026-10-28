"""Clip planning: event time + keyframes -> clip window, method and the exact ffmpeg command."""

import bisect
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["copy", "reencode"]


class ReencodeConfig(BaseModel):
    """Encoder settings for the re-encode fallback."""

    video_codec: str = "libx264"
    preset: str = "veryfast"
    crf: int = Field(23, ge=0, le=51)
    audio_codec: str = "aac"


class ClipsConfig(BaseModel):
    """Contents of configs/clips.yaml."""

    pre_sec: float = Field(5.0, ge=0)
    post_sec: float = Field(3.0, ge=0)
    copy_max_keyframe_offset_sec: float = Field(0.5, ge=0)
    reencode: ReencodeConfig = ReencodeConfig()
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


class ClipPlan(BaseModel):
    """What to cut for one event and how."""

    requested_start: float
    requested_end: float
    start: float = Field(description="Actual start: a keyframe for stream copy.")
    end: float
    method: Method


def window(t_sec: float, duration: float, cfg: ClipsConfig) -> tuple[float, float]:
    """Clip window around an event, clipped to the video."""
    return max(0.0, t_sec - cfg.pre_sec), min(duration, t_sec + cfg.post_sec)


def plan_clip(
    t_sec: float, duration: float, keyframes: Sequence[float], cfg: ClipsConfig
) -> ClipPlan:
    """Stream copy from the last keyframe before the window if it is close enough, else re-encode.

    ``keyframes`` must be sorted.
    """
    start, end = window(t_sec, duration, cfg)
    i = bisect.bisect_right(keyframes, start + 1e-6) - 1
    if i >= 0 and start - keyframes[i] <= cfg.copy_max_keyframe_offset_sec + 1e-9:
        return ClipPlan(
            requested_start=start, requested_end=end, start=keyframes[i], end=end, method="copy"
        )
    return ClipPlan(
        requested_start=start, requested_end=end, start=start, end=end, method="reencode"
    )


def ffmpeg_args(
    plan: ClipPlan, src: Path, dst: Path, cfg: ClipsConfig, method: Method
) -> list[str]:
    """Ffmpeg command for a plan; ``-ss`` before ``-i`` gives a fast (input) seek."""
    head = [
        cfg.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{plan.start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{plan.end - plan.start:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if method == "copy":
        codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        r = cfg.reencode
        codec = [
            "-c:v",
            r.video_codec,
            "-preset",
            r.preset,
            "-crf",
            str(r.crf),
            "-c:a",
            r.audio_codec,
        ]
    return [*head, *codec, "-movflags", "+faststart", str(dst)]


def params_hash(src: Path, plan: ClipPlan, cfg: ClipsConfig) -> str:
    """Stable hash of everything that determines a clip's content (for idempotent re-runs)."""
    stat = src.stat()
    payload = {
        "src": str(src.resolve()),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "plan": plan.model_dump(),
        "reencode": cfg.reencode.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
