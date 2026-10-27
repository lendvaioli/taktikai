"""Frame selection at a fixed sampling rate and iteration over the selected frames."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    """Sampling parameters from configs/video.yaml."""

    stride_fps: float = Field(5.0, gt=0, description="Rate of the frames written to outputs.")
    detect_fps: float | None = Field(
        None, gt=0, description="Rate of the frames processed; None = stride_fps."
    )
    start_sec: float = Field(0.0, ge=0)
    duration_sec: float | None = Field(None, gt=0)


class VideoMeta(BaseModel):
    """Basic properties of a video file."""

    fps: float = Field(gt=0)
    n_frames: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FrameRef(BaseModel):
    """A sampled frame: its index in the source video and its timestamp."""

    index: int = Field(ge=0)
    t_sec: float = Field(ge=0)


def select_frame_indices(meta: VideoMeta, cfg: VideoConfig) -> list[FrameRef]:
    """Return the frames to sample, spaced ``1 / stride_fps`` seconds apart.

    Indices are rounded to the nearest source frame. The window ``[start, start + duration)``
    is clipped to the video length. If ``stride_fps`` >= source fps every frame is used.
    """
    first = round(cfg.start_sec * meta.fps)
    end = meta.n_frames
    if cfg.duration_sec is not None:
        end = min(end, round((cfg.start_sec + cfg.duration_sec) * meta.fps))
    step = max(meta.fps / cfg.stride_fps, 1.0)
    indices: list[int] = []
    k = 0
    while (index := first + round(k * step)) < end:
        if not indices or index != indices[-1]:
            indices.append(index)
        k += 1
    return [FrameRef(index=i, t_sec=i / meta.fps) for i in indices]


def processing_frames(meta: VideoMeta, cfg: VideoConfig) -> tuple[list[FrameRef], set[int]]:
    """Return the frames to process and the indices of those to write to outputs.

    Processed frames are the union of the ``detect_fps`` grid and the ``stride_fps`` output
    grid, so output timestamps stay on the output grid even when the two rates do not align.
    """
    output = select_frame_indices(meta, cfg)
    output_idx = {ref.index for ref in output}
    if cfg.detect_fps is None:
        return output, output_idx
    dense = select_frame_indices(meta, cfg.model_copy(update={"stride_fps": cfg.detect_fps}))
    merged = {ref.index: ref for ref in [*dense, *output]}
    return [merged[i] for i in sorted(merged)], output_idx


def read_meta(path: Path) -> VideoMeta:
    """Read fps, frame count and resolution of a video with OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path, refs: list[FrameRef]) -> Iterator[tuple[FrameRef, np.ndarray]]:
    """Yield ``(ref, BGR image)`` for each requested frame, decoding sequentially."""
    import cv2

    if not refs:
        return
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, refs[0].index)
        position = refs[0].index
        for ref in refs:
            while position < ref.index:
                cap.grab()
                position += 1
            ok, image = cap.read()
            position += 1
            if not ok:
                return
            yield ref, image
    finally:
        cap.release()
