"""Annotated debug video: ellipses coloured by team, track ids and a ball marker."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from match_tagger.detection.config import DebugRenderConfig
from match_tagger.detection.schema import DetectionRecord
from match_tagger.video.frames import VideoConfig, iter_frames, read_meta, select_frame_indices

# Colour slots: team 0, team 1, player/keeper without team, referee, ball.
PALETTE = ["#00BFFF", "#FF1493", "#A0A0A0", "#FFD700", "#FFFFFF"]
NO_TEAM, REFEREE, BALL = 2, 3, 4


def color_slot(record: DetectionRecord) -> int:
    """Return the palette index for a record."""
    if record.cls == "ball":
        return BALL
    if record.cls == "referee":
        return REFEREE
    return NO_TEAM if record.team is None else record.team


def label(record: DetectionRecord) -> str:
    """Track id, prefixed with K (kapus) for goalkeepers."""
    prefix = "K" if record.cls == "goalkeeper" else ""
    return f"{prefix}{record.track_id}"


def _annotate(image: np.ndarray, records: Sequence[DetectionRecord], tools: tuple) -> np.ndarray:
    import supervision as sv

    ellipse, labeler, triangle = tools
    people = [r for r in records if r.cls != "ball"]
    balls = [r for r in records if r.cls == "ball"]
    for group, annotators in ((people, (ellipse, labeler)), (balls, (triangle,))):
        if not group:
            continue
        dets = sv.Detections(
            xyxy=np.array([[r.x1, r.y1, r.x2, r.y2] for r in group], dtype=np.float32),
            class_id=np.array([color_slot(r) for r in group]),
        )
        for annotator in annotators:
            if annotator is labeler:
                image = annotator.annotate(image, dets, labels=[label(r) for r in group])
            else:
                image = annotator.annotate(image, dets)
    return image


def render_debug(
    video: Path,
    records: Sequence[DetectionRecord],
    stride_fps: float,
    cfg: DebugRenderConfig,
    out_path: Path,
) -> int:
    """Write the annotated window ``[start, start + duration)`` at ``stride_fps``; return frames."""
    import supervision as sv

    palette = sv.ColorPalette.from_hex(PALETTE)
    tools = (
        sv.EllipseAnnotator(color=palette, thickness=2),
        sv.LabelAnnotator(
            color=palette, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER
        ),
        sv.TriangleAnnotator(color=palette, base=20, height=17, outline_thickness=1),
    )
    by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)
    for record in records:
        by_frame[record.frame].append(record)
    meta = read_meta(video)
    window = VideoConfig(
        stride_fps=stride_fps, start_sec=cfg.start_sec, duration_sec=cfg.duration_sec
    )
    info = sv.VideoInfo(width=meta.width, height=meta.height, fps=round(stride_fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with sv.VideoSink(str(out_path), info) as sink:
        for ref, image in iter_frames(video, select_frame_indices(meta, window)):
            sink.write_frame(_annotate(image, by_frame.get(ref.index, []), tools))
            n_written += 1
    return n_written
