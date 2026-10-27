"""Detection stage: video -> detect -> track -> team labels -> detections.parquet."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from match_tagger.config import DefaultConfig, load_config
from match_tagger.detection.config import DetectionConfig
from match_tagger.detection.detector import build_detector, resolve_device
from match_tagger.detection.render import render_debug
from match_tagger.detection.schema import CLASSES, DetectionRecord, write_detections
from match_tagger.detection.summary import DetectionSummary, coverage
from match_tagger.tracking.classes import ClassVotes, apply_track_classes
from match_tagger.tracking.config import TrackingConfig
from match_tagger.tracking.teams import CropCollector, TeamClassifier, label_teams, vote_per_track
from match_tagger.tracking.tracker import build_tracker, track
from match_tagger.video.frames import (
    FrameRef,
    VideoConfig,
    iter_frames,
    processing_frames,
    read_meta,
)

if TYPE_CHECKING:
    import supervision as sv

BALL = CLASSES.index("ball")


@dataclass
class StageConfig:
    """All configs the detection stage reads."""

    paths: DefaultConfig
    video: VideoConfig
    detection: DetectionConfig
    tracking: TrackingConfig

    @classmethod
    def load(cls, config_dir: Path) -> "StageConfig":
        """Load default, video, detection and tracking YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            video=load_config("video", VideoConfig, config_dir),
            detection=load_config("detection", DetectionConfig, config_dir),
            tracking=load_config("tracking", TrackingConfig, config_dir),
        )


@dataclass
class _Timer:
    totals: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, start: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start


def _record(
    ref: FrameRef, cls_idx: int, xyxy: np.ndarray, conf: float, tid: int | None
) -> DetectionRecord:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return DetectionRecord(
        frame=ref.index, t_sec=ref.t_sec, track_id=tid, cls=CLASSES[cls_idx],
        x1=x1, y1=y1, x2=x2, y2=y2, conf=float(conf),
    )  # fmt: skip


def _ball_record(ref: FrameRef, dets: "sv.Detections") -> list[DetectionRecord]:
    """Keep only the highest-confidence ball of the frame, without a track id."""
    balls = dets[dets.class_id == BALL]
    if len(balls) == 0:
        return []
    best = int(np.argmax(balls.confidence))
    return [_record(ref, BALL, balls.xyxy[best], balls.confidence[best], None)]


def _detect_and_track(
    video: Path, refs: list[FrameRef], output_idx: set[int], cfg: StageConfig, device: str,
    timer: _Timer,
) -> tuple[list[DetectionRecord], CropCollector, ClassVotes]:  # fmt: skip
    """Single pass over the processed frames; returns untagged output records, crops and votes.

    Every processed frame feeds the tracker, the crop collector and the class votes; only
    frames in ``output_idx`` produce records.
    """
    detector = build_detector(cfg.detection, cfg.paths.paths.models_dir, device)
    track_fps = cfg.video.detect_fps or cfg.video.stride_fps
    tracker = build_tracker(cfg.tracking.bytetrack, track_fps)
    collector, votes = CropCollector(cfg.tracking.team), ClassVotes()
    records: list[DetectionRecord] = []
    for ref, image in iter_frames(video, refs):
        start = time.perf_counter()
        dets = detector.detect(image)
        timer.add("detect", start)
        start = time.perf_counter()
        is_output = ref.index in output_idx
        records += _ball_record(ref, dets) if is_output else []
        people = dets[dets.class_id != BALL].with_nms(cfg.detection.nms_iou, class_agnostic=True)
        tracked = track(tracker, people)
        for xyxy, conf, cls_idx, tid in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id, strict=True
        ):
            if is_output:
                records.append(_record(ref, int(cls_idx), xyxy, conf, int(tid)))
            votes.add(int(tid), CLASSES[cls_idx], float(conf))
            collector.add(image, int(tid), xyxy)
        timer.add("track_and_crop", start)
    return records, collector, votes


def _classify_teams(
    collector: CropCollector, player_tracks: set[int], cfg: StageConfig, device: str
) -> dict[int, int]:
    """Cluster the crops of player tracks into two teams and vote one team per track."""
    track_ids, crops = collector.subset(player_tracks)
    if len(crops) < 3:
        return {}
    labels = TeamClassifier(cfg.tracking.team, device).fit_predict(crops)
    return vote_per_track(track_ids, labels)


def run_detection(video: Path, match_id: str, cfg: StageConfig) -> DetectionSummary:
    """Run the stage and write detections.parquet, the summary and the debug video."""
    started = time.perf_counter()
    timer, out_dir = _Timer(), cfg.paths.paths.processed_dir / match_id
    device = resolve_device(cfg.detection.device)
    refs, output_idx = processing_frames(read_meta(video), cfg.video)
    records, collector, votes = _detect_and_track(video, refs, output_idx, cfg, device, timer)
    start = time.perf_counter()
    track_cls = votes.resolve()
    records = apply_track_classes(records, track_cls)
    player_tracks = {tid for tid, cls in track_cls.items() if cls == "player"}
    records = label_teams(records, _classify_teams(collector, player_tracks, cfg, device))
    timer.add("teams", start)
    write_detections(records, out_dir / "detections.parquet")
    if cfg.detection.debug.enabled:
        start = time.perf_counter()
        render_debug(
            video,
            records,
            cfg.video.stride_fps,
            cfg.detection.debug,
            out_dir / "debug_detection.mp4",
        )
        timer.add("render", start)
    timer.add("total", started)
    summary = DetectionSummary(
        match_id=match_id, video=str(video), device=device,
        timings_sec={k: round(v, 2) for k, v in timer.totals.items()},
        n_processed_frames=len(refs),
        **coverage(records, len(output_idx)),
    )  # fmt: skip
    (out_dir / "detection_summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary
