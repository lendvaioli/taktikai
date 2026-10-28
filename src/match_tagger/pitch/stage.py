"""Pitch stage: detections.parquet + video -> per-frame homography -> positions.parquet."""

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from match_tagger.config import DefaultConfig, load_config
from match_tagger.detection.detector import resolve_device
from match_tagger.detection.schema import DetectionRecord, load_detection_records
from match_tagger.pitch.config import PitchConfig
from match_tagger.pitch.geometry import PitchDims, keypoint_vertices
from match_tagger.pitch.homography import HomographyFit, fit_homography, select_keypoints
from match_tagger.pitch.keypoints import build_keypoint_detector
from match_tagger.pitch.minimap import FrameKeypoints, render_minimap
from match_tagger.pitch.project import project_frame
from match_tagger.pitch.schema import (
    HomographyRecord,
    PositionRecord,
    write_homographies,
    write_positions,
)
from match_tagger.pitch.sequence import (
    HQuality,
    assign_quality,
    pitch_grid,
    smooth_homographies,
)
from match_tagger.pitch.summary import PitchSummary, player_plausibility, quality_shares
from match_tagger.video.frames import (
    FrameRef,
    VideoConfig,
    iter_frames,
    read_meta,
    select_frame_indices,
)

State = tuple[np.ndarray | None, HQuality]


@dataclass
class PitchStageConfig:
    """All configs the pitch stage reads."""

    paths: DefaultConfig
    video: VideoConfig
    pitch: PitchConfig

    @classmethod
    def load(cls, config_dir: Path) -> "PitchStageConfig":
        """Load default, video and pitch YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            video=load_config("video", VideoConfig, config_dir),
            pitch=load_config("pitch", PitchConfig, config_dir),
        )


def _estimate(
    video: Path, refs: list[FrameRef], cfg: PitchStageConfig, vertices: np.ndarray, device: str
) -> tuple[list[HomographyFit | None], dict[int, FrameKeypoints]]:
    """Detect keypoints and fit one homography per output frame (None if not possible)."""
    detector = build_keypoint_detector(cfg.pitch, cfg.paths.paths.models_dir, device)
    fits: list[HomographyFit | None] = []
    keypoints: dict[int, FrameKeypoints] = {}
    for ref, image in iter_frames(video, refs):
        xy, conf = detector.detect(image)
        mask = select_keypoints(xy, conf, (image.shape[1], image.shape[0]), cfg.pitch.homography)
        fits.append(fit_homography(xy[mask], vertices[mask], cfg.pitch.homography))
        keypoints[ref.index] = FrameKeypoints(xy=xy, mask=mask)
    return fits, keypoints


def _positions(
    records: list[DetectionRecord], refs: list[FrameRef], states: list[State],
    cfg: PitchStageConfig, match_id: str,
) -> list[PositionRecord]:  # fmt: skip
    """Project every detection row with its frame's homography state."""
    by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)
    for r in records:
        by_frame[r.frame].append(r)
    unknown = set(by_frame) - {ref.index for ref in refs}
    if unknown:
        raise ValueError(f"{len(unknown)} detection frames lie outside the video window")
    dims = cfg.pitch.dims_for(match_id)
    out: list[PositionRecord] = []
    for ref, (matrix, quality) in zip(refs, states, strict=True):
        rows = by_frame.get(ref.index, [])
        out += project_frame(rows, matrix, quality, dims, cfg.pitch.out_margin_m) if rows else []
    return out


def run_pitch(video: Path, match_id: str, cfg: PitchStageConfig) -> PitchSummary:
    """Run the stage and write positions.parquet, the summary and the minimap video."""
    started, timings = time.perf_counter(), {}
    out_dir, dims = cfg.paths.paths.processed_dir / match_id, cfg.pitch.dims_for(match_id)
    device = resolve_device(cfg.pitch.device)
    records = load_detection_records(out_dir / "detections.parquet")
    refs = select_frame_indices(read_meta(video), cfg.video)
    fits, keypoints = _estimate(video, refs, cfg, keypoint_vertices(dims), device)
    timings["keypoints_and_fit"] = time.perf_counter() - started
    times = [ref.t_sec for ref in refs[: len(fits)]]
    raw = [f.matrix if f else None for f in fits]
    smoothed = smooth_homographies(times, raw, cfg.pitch.smooth_window_sec, pitch_grid(dims))
    states = assign_quality(times, smoothed, cfg.pitch.max_reuse_sec)
    positions = _positions(records, refs[: len(fits)], states, cfg, match_id)
    write_positions(positions, out_dir / "positions.parquet")
    write_homographies(
        [
            HomographyRecord(frame=ref.index, t_sec=ref.t_sec, h_quality=quality,
                             matrix=None if m is None else m.ravel().tolist())
            for ref, (m, quality) in zip(refs, states, strict=False)
        ],
        out_dir / "homographies.parquet",
    )  # fmt: skip
    if cfg.pitch.minimap.enabled:
        start = time.perf_counter()
        render_minimap(
            video, positions, keypoints, states, refs, dims, cfg, out_dir / "minimap.mp4"
        )
        timings["render"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - started
    summary = _summary(match_id, device, dims, refs, fits, keypoints, states, positions, cfg)
    summary.timings_sec = {k: round(v, 2) for k, v in timings.items()}
    path = out_dir / "pitch_summary.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary


def _summary(
    match_id: str, device: str, dims: PitchDims, refs: list[FrameRef],
    fits: list[HomographyFit | None], keypoints: dict[int, FrameKeypoints], states: list[State],
    positions: list[PositionRecord], cfg: PitchStageConfig,
) -> PitchSummary:  # fmt: skip
    """Collect coverage and plausibility numbers for pitch_summary.json."""
    ok_errors = [f.mean_error_m for f in fits if f is not None]
    next_frame = {a.index: b.index for a, b in zip(refs, refs[1:], strict=False)}
    return PitchSummary(
        match_id=match_id, device=device, pitch_length_m=dims.length_m,
        pitch_width_m=dims.width_m, n_frames=len(states),
        h_quality_frame_share=quality_shares([q for _, q in states]),
        mean_confident_keypoints=float(np.mean([k.mask.sum() for k in keypoints.values()])),
        mean_inlier_error_m=float(np.mean(ok_errors)) if ok_errors else None,
        **player_plausibility(positions, dims, next_frame, cfg.pitch.max_speed_mps),
    )  # fmt: skip
