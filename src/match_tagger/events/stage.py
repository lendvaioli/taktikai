"""Events stage v1: detections + positions + homographies -> events.json (pass, ball_loss)."""

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from match_tagger.config import DefaultConfig, load_config
from match_tagger.detection.schema import read_detections
from match_tagger.events.ball import (
    airborne_flags,
    expected_ball_px,
    ground_ratio_baseline,
    interpolate_gaps,
)
from match_tagger.events.config import BallConfig, EventsConfig, MatchesConfig
from match_tagger.events.possession import FramePlayers, Segment, frame_candidate, segments
from match_tagger.events.rules import events_from_segments
from match_tagger.events.schema import Event, EventsFile, Team
from match_tagger.events.teams import TeamMapping, kit_colour, map_clusters, parse_kit
from match_tagger.pitch.schema import read_homographies, read_positions
from match_tagger.tracking.teams import crop_box
from match_tagger.video.frames import FrameRef, VideoConfig, iter_frames

PEOPLE = ("player", "goalkeeper")


@dataclass
class EventsStageConfig:
    """All configs the events stage reads."""

    paths: DefaultConfig
    video: VideoConfig
    events: EventsConfig
    matches: MatchesConfig

    @classmethod
    def load(cls, config_dir: Path) -> "EventsStageConfig":
        """Load default, video, events and matches YAML files."""
        return cls(
            paths=load_config("default", DefaultConfig, config_dir),
            video=load_config("video", VideoConfig, config_dir),
            events=load_config("events", EventsConfig, config_dir),
            matches=load_config("matches", MatchesConfig, config_dir),
        )


@dataclass
class ClipData:
    """Per-clip inputs of the event rules, loaded once (also reused by tuning)."""

    clip_id: str
    times: np.ndarray  # t_sec of every output frame
    frames: np.ndarray  # frame index of every output frame
    h_quality: list[str]
    ball_xy: np.ndarray  # projected ball, NaN where no usable ball row
    ball_size_ratio: np.ndarray  # observed / expected ball width, NaN if unknown
    players: list[FramePlayers]
    mapping: TeamMapping
    tracks: pd.DataFrame  # track_id, team, first/last t_sec and position (diagnostics)


@dataclass
class Detection:
    """Output of the rules on one clip plus intermediate state for diagnostics."""

    events: list[Event]
    segments: list[Segment]
    ball_xy: np.ndarray
    airborne: np.ndarray
    interpolated: np.ndarray
    possessed: np.ndarray
    ratio_baseline: float | None = None


def _team_mapping(
    video: Path, det: pd.DataFrame, kits: dict[Team, str], cfg: EventsConfig
) -> TeamMapping:
    """Sample player crops per cluster from the video and map clusters to home/away."""
    players = det[(det.cls == "player") & det.team.notna()]
    sample = pd.concat(
        g.sample(min(len(g), cfg.team_mapping.crops_per_team), random_state=cfg.team_mapping.seed)
        for _, g in players.groupby("team")
    ).sort_values("frame")
    refs = [
        FrameRef(index=int(f), t_sec=float(t))
        for f, t in sample[["frame", "t_sec"]].drop_duplicates().itertuples(index=False)
    ]
    crops: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for ref, image in iter_frames(video, refs):
        for r in sample[sample.frame == ref.index].itertuples():
            crop = crop_box(image, (r.x1, r.y1, r.x2, r.y2))
            if crop is not None:
                crops[int(r.team)].append(crop)
    if not crops[0] or not crops[1]:
        raise ValueError("Team mapping needs player crops of both team clusters")
    palette = cfg.team_mapping.palette
    lab = {c: kit_colour(v) for c, v in crops.items()}
    return map_clusters(lab, {t: parse_kit(n, palette) for t, n in kits.items()})


def _ball_table(
    pos: pd.DataFrame,
    det: pd.DataFrame,
    frames: np.ndarray,
    matrices: dict[int, np.ndarray],
    diameter_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per output frame: projected ball position and observed/expected size ratio."""
    xy = np.full((len(frames), 2), np.nan)
    ratio = np.full(len(frames), np.nan)
    index = {int(f): i for i, f in enumerate(frames)}
    ball = pos[(pos.cls == "ball") & pos.x_m.notna() & ~pos.out_of_pitch]
    width = det[det.cls == "ball"].set_index("frame").eval("x2 - x1")
    for r in ball.itertuples():
        i = index.get(int(r.frame))
        if i is None:
            continue
        xy[i] = (r.x_m, r.y_m)
        if int(r.frame) in matrices and int(r.frame) in width.index:
            expected = expected_ball_px(matrices[int(r.frame)], r.x_m, r.y_m, diameter_m)
            ratio[i] = float(width.loc[int(r.frame)]) / max(expected, 1e-6)
    return xy, ratio


def _players(pos: pd.DataFrame, frames: np.ndarray, mapping: TeamMapping) -> list[FramePlayers]:
    """Players and goalkeepers with team and position, per output frame."""
    ppl = pos[pos.cls.isin(PEOPLE) & pos.team.notna() & pos.x_m.notna()]
    groups = dict(tuple(ppl.groupby("frame")))
    out = []
    for f in frames:
        g = groups.get(int(f))
        if g is None:
            out.append(FramePlayers(np.empty(0, int), [], np.empty((0, 2))))
            continue
        teams = [mapping.mapping[int(t)] for t in g.team]
        out.append(FramePlayers(g.track_id.to_numpy(int), teams, g[["x_m", "y_m"]].to_numpy()))
    return out


def load_clip(match_dir: Path, video: Path, kits: dict[Team, str], cfg: EventsConfig) -> ClipData:
    """Load detections, positions and homographies of one clip into ``ClipData``."""
    det = read_detections(match_dir / "detections.parquet")
    pos = read_positions(match_dir / "positions.parquet")
    homs = read_homographies(match_dir / "homographies.parquet")
    frames = np.array([h.frame for h in homs])
    matrices = {h.frame: np.array(h.matrix).reshape(3, 3) for h in homs if h.matrix is not None}
    mapping = _team_mapping(video, det, kits, cfg)
    ball_xy, ratio = _ball_table(pos, det, frames, matrices, cfg.ball.ball_diameter_m)
    ppl = pos[pos.cls.isin(PEOPLE) & pos.team.notna() & pos.x_m.notna()].sort_values("t_sec")
    tracks = ppl.groupby("track_id").agg(
        team=("team", "first"),
        first_t=("t_sec", "first"),
        last_t=("t_sec", "last"),
        first_x=("x_m", "first"),
        first_y=("y_m", "first"),
        last_x=("x_m", "last"),
        last_y=("y_m", "last"),
    )
    return ClipData(
        match_dir.name,
        np.array([h.t_sec for h in homs]),
        frames,
        [h.h_quality for h in homs],
        ball_xy,
        ratio,
        _players(pos, frames, mapping),
        mapping,
        tracks,
    )


def airborne_mask(clip: ClipData, cfg: BallConfig) -> tuple[np.ndarray, float | None]:
    """Airborne flag per frame for the configured mode, and the calibration baseline used.

    ``calibrated`` divides the size ratio by the clip's ground-ball baseline; without enough
    ground balls for a baseline it falls back to the fixed rule (baseline None).
    """
    airborne = np.zeros(len(clip.times), dtype=bool)
    seen = ~np.isnan(clip.ball_xy[:, 0])
    if cfg.airborne_mode == "off" or not seen.any():
        return airborne, None
    ratio, baseline = clip.ball_size_ratio, None
    if cfg.airborne_mode == "calibrated":
        baseline = ground_ratio_baseline(
            clip.ball_xy,
            ratio,
            [p.xy for p in clip.players],
            cfg.calibration_near_m,
            cfg.calibration_min_samples,
        )
        ratio = ratio / baseline if baseline else ratio
    airborne[seen] = airborne_flags(
        clip.times[seen],
        clip.ball_xy[seen],
        ratio[seen],
        cfg.airborne_size_ratio,
        cfg.max_ground_speed_mps,
    )
    return airborne, baseline


def detect_events(clip: ClipData, cfg: EventsConfig) -> Detection:
    """Run the v1 rules on loaded clip data (pure; used by the stage and by tuning)."""
    airborne, baseline = airborne_mask(clip, cfg.ball)
    ball = clip.ball_xy.copy()
    ball[airborne] = np.nan
    ball, interpolated = interpolate_gaps(
        clip.times, ball, cfg.ball.max_interp_gap_sec, blocked=airborne
    )
    radius = cfg.possession.control_radius_m
    cands = [frame_candidate(b, p, radius) for b, p in zip(ball, clip.players, strict=True)]
    segs = segments(clip.times, cands, cfg.possession.min_hold_sec)
    events = events_from_segments(segs, clip.times, ball, clip.clip_id, cfg.possession, cfg.rules)
    possessed = np.array([c is not None for c in cands])
    return Detection(events, segs, ball, airborne, interpolated, possessed, baseline)


class EventsSummary(BaseModel):
    """events_summary.json: team mapping, ball coverage, possession and event counts."""

    match_id: str
    team_mapping: dict[str, str]
    cluster_kit_lab: dict[str, tuple[float, float, float]]
    mapping_margin: float
    kits: dict[str, str]
    n_frames: int
    ball_observed_share: float
    airborne_mode: str
    ball_ratio_baseline: float | None
    ball_airborne_share: float
    ball_interpolated_share: float
    ball_known_share: float
    possession_share: float
    n_segments: int
    events_per_type: dict[str, int]
    timings_sec: dict[str, float]


def run_events(video: Path, match_id: str, cfg: EventsStageConfig) -> EventsSummary:
    """Run the stage and write events.json and events_summary.json."""
    started = time.perf_counter()
    if match_id not in cfg.matches.matches:
        raise ValueError(f"No kits for {match_id!r} in configs/matches.yaml")
    kits = cfg.matches.matches[match_id].kits
    out_dir = cfg.paths.paths.processed_dir / match_id
    clip = load_clip(out_dir, video, kits, cfg.events)
    loaded = time.perf_counter() - started
    result = detect_events(clip, cfg.events)
    events_file = EventsFile(clip_id=match_id, events=result.events)
    (out_dir / "events.json").write_text(events_file.model_dump_json(indent=2), encoding="utf-8")
    n, observed = len(clip.times), ~np.isnan(clip.ball_xy[:, 0])
    summary = EventsSummary(
        match_id=match_id,
        team_mapping={str(k): v for k, v in clip.mapping.mapping.items()},
        cluster_kit_lab={str(k): v for k, v in clip.mapping.cluster_lab.items()},
        mapping_margin=round(clip.mapping.margin, 1),
        kits=dict(kits),
        n_frames=n,
        ball_observed_share=float(observed.mean()),
        airborne_mode=cfg.events.ball.airborne_mode,
        ball_ratio_baseline=result.ratio_baseline,
        ball_airborne_share=float(result.airborne.sum() / max(observed.sum(), 1)),
        ball_interpolated_share=float(result.interpolated.mean()),
        ball_known_share=float((~np.isnan(result.ball_xy[:, 0])).mean()),
        possession_share=float(result.possessed.mean()),
        n_segments=len(result.segments),
        events_per_type=dict(Counter(e.type for e in result.events)),
        timings_sec={"load": round(loaded, 2), "total": round(time.perf_counter() - started, 2)},
    )
    path = out_dir / "events_summary.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary
