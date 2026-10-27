"""Side-by-side video: original footage left, pitch minimap (mplsoccer) right.

Left panel overlay: detected keypoints (yellow) and pitch lines projected back with the frame's
homography (cyan). A bar above the minimap shows h_quality: green ok, orange reused, red
missing. Minimap markers: filled dot ok, hollow dot reused, cross out_of_pitch.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from match_tagger.detection.render import BALL, PALETTE, color_slot
from match_tagger.pitch.geometry import EDGES, PitchDims, keypoint_vertices
from match_tagger.pitch.homography import project
from match_tagger.pitch.schema import PositionRecord
from match_tagger.video.frames import FrameRef, iter_frames

if TYPE_CHECKING:
    from match_tagger.pitch.stage import PitchStageConfig, State

QUALITY_BGR = {"ok": (80, 200, 80), "reused": (0, 165, 255), "missing": (60, 60, 230)}
PitchToPixel = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FrameKeypoints:
    """Keypoint pixels of one frame and the mask of those used for the homography."""

    xy: np.ndarray
    mask: np.ndarray


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def pitch_canvas(dims: PitchDims, height: int) -> tuple[np.ndarray, PitchToPixel]:
    """Draw an mplsoccer pitch once; return it (BGR) and a metres -> pixel mapping."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    pitch = Pitch(
        pitch_type="custom", pitch_length=dims.length_m, pitch_width=dims.width_m,
        pitch_color="#1e5631", line_color="white", pad_left=6, pad_right=6, pad_top=6,
        pad_bottom=6,
    )  # fmt: skip
    aspect = (dims.length_m + 12) / (dims.width_m + 12)
    fig, ax = pitch.draw(figsize=(height * aspect / 100, height / 100))
    fig.set_dpi(100)
    fig.patch.set_facecolor("#1e5631")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    transform, canvas_h = ax.transData, image.shape[0]
    plt.close(fig)

    def to_pixel(xy: np.ndarray) -> np.ndarray:
        display = transform.transform(np.asarray(xy, dtype=float).reshape(-1, 2))
        return np.column_stack([display[:, 0], canvas_h - display[:, 1]])

    return image, to_pixel


def _overlay(image: np.ndarray, kp: FrameKeypoints | None, matrix: np.ndarray | None,
             dims: PitchDims) -> None:  # fmt: skip
    """Draw projected pitch lines and the keypoints used for the fit onto ``image``."""
    if matrix is not None:
        pts = project(np.linalg.inv(matrix), keypoint_vertices(dims))
        rect = (0, 0, image.shape[1], image.shape[0])
        for a, b in EDGES:
            p, q = np.clip(pts[a - 1], -1e5, 1e5), np.clip(pts[b - 1], -1e5, 1e5)
            ok, p1, p2 = cv2.clipLine(rect, tuple(map(int, p)), tuple(map(int, q)))
            if ok:
                cv2.line(image, p1, p2, (255, 255, 0), 2)
    if kp is not None:
        for x, y in kp.xy[kp.mask]:
            cv2.circle(image, (int(x), int(y)), 7, (0, 255, 255), -1)


def _draw_minimap(canvas: np.ndarray, rows: Sequence[PositionRecord], to_px: PitchToPixel) -> None:
    """Draw one frame's positions onto a copy-ready minimap canvas."""
    placed = [r for r in rows if r.x_m is not None]
    if not placed:
        return
    pixels = to_px(np.array([(r.x_m, r.y_m) for r in placed]))
    for r, (x, y) in zip(placed, pixels.astype(int), strict=True):
        slot = color_slot(r.cls, r.team)
        color, radius = _bgr(PALETTE[slot]), 4 if slot == BALL else 7
        if r.out_of_pitch:
            cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, 12, 2)
        else:
            cv2.circle(canvas, (x, y), radius, color, -1 if r.h_quality == "ok" else 2)
            cv2.circle(canvas, (x, y), radius, (0, 0, 0), 1)


def render_minimap(
    video: Path, positions: Sequence[PositionRecord], keypoints: dict[int, FrameKeypoints],
    states: Sequence["State"], refs: Sequence[FrameRef], dims: PitchDims,
    cfg: "PitchStageConfig", out_path: Path,
) -> int:  # fmt: skip
    """Write the side-by-side video for the configured window; return the frames written."""
    mm = cfg.pitch.minimap
    background, to_px = pitch_canvas(dims, mm.panel_height)
    by_frame: dict[int, list[PositionRecord]] = defaultdict(list)
    for p in positions:
        by_frame[p.frame].append(p)
    state_of = {ref.index: s for ref, s in zip(refs, states, strict=False)}
    window = [r for r in refs if mm.start_sec <= r.t_sec < mm.start_sec + mm.duration_sec]
    height = background.shape[0]
    writer, n = None, 0
    for ref, image in iter_frames(video, window):
        matrix, quality = state_of.get(ref.index, (None, "missing"))
        if mm.overlay:
            _overlay(image, keypoints.get(ref.index), matrix, dims)
        left = cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
        right = background.copy()
        cv2.rectangle(right, (0, 0), (right.shape[1], 8), QUALITY_BGR[quality], -1)
        _draw_minimap(right, by_frame.get(ref.index, []), to_px)
        frame = np.hstack([left, right])
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (frame.shape[1], frame.shape[0])
            writer = cv2.VideoWriter(str(out_path), fourcc, cfg.video.stride_fps, size)
        writer.write(frame)
        n += 1
    if writer is not None:
        writer.release()
    return n
