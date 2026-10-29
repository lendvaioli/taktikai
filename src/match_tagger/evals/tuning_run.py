"""``make tune``: leave-one-clip-out tuning and error analysis of events v1 on a golden set."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from match_tagger.evals.diagnostics import (
    MISS_CAUSES,
    false_pass_causes,
    miss_cause,
    observable,
)
from match_tagger.evals.golden import GoldenSet, load_golden_set
from match_tagger.evals.matching import match_events
from match_tagger.evals.metrics import Counts, MetricRow
from match_tagger.evals.run import EvalConfig, evaluate_clip, git_commit
from match_tagger.evals.tune import (
    Fold,
    Spread,
    distance_to_start,
    fold_spread,
    grid_combos,
    held_out_total,
    leave_one_out,
    select_best,
    summed,
)
from match_tagger.events.stage import ClipData, EventsStageConfig, detect_events, load_clip


class MappingCheck(BaseModel):
    """Predicted vs reference cluster -> team mapping of one clip."""

    clip_id: str
    predicted: dict[str, str]
    reference: dict[str, str] | None
    correct: bool | None
    margin: float


class Observability(BaseModel):
    """Held-out golden passes split by whether a ball position was known around them."""

    window_sec: float
    raw_observable: int
    raw_not_observable: int
    processed_observable: int
    processed_not_observable: int
    recall_raw_observable: float | None
    recall_raw_not_observable: float | None
    recall_processed_observable: float | None
    recall_processed_not_observable: float | None


class ModeResult(BaseModel):
    """Leave-one-clip-out restricted to one airborne mode, for a side-by-side comparison."""

    mode: str
    held_out_pass: MetricRow
    spread: Spread
    fold_params: dict[str, dict[str, Any]]
    airborne_share: float = Field(description="Flagged share of observed balls, start values.")
    ratio_baselines: dict[str, float | None]


class ClipPassResult(BaseModel):
    """Pass metrics of one clip next to the tracking and ball statistics of its detections."""

    clip_id: str
    golden_passes: int
    held_out: MetricRow = Field(description="From the fold that held this clip out.")
    in_sample: MetricRow = Field(description="With the final thresholds tuned on all clips.")
    n_player_tracks: int
    median_player_track_sec: float | None
    ball_frame_share: float | None


def clip_pass_results(
    folds: list[Fold],
    pass_table: list[dict[str, Counts]],
    final: int,
    golden: GoldenSet,
    processed_dir: Path,
) -> list[ClipPassResult]:
    """Per clip: held-out and in-sample pass metrics plus detection_summary.json statistics."""
    out = []
    for fold in folds:
        clip_id = fold.held_out
        summary = json.loads(
            (processed_dir / clip_id / "detection_summary.json").read_text(encoding="utf-8")
        )
        out.append(
            ClipPassResult(
                clip_id=clip_id,
                golden_passes=sum(e.type == "pass" for e in golden.clip(clip_id).events),
                held_out=fold.held,
                in_sample=MetricRow.from_counts(pass_table[final][clip_id]),
                n_player_tracks=summary["n_player_tracks"],
                median_player_track_sec=summary.get("median_player_track_sec"),
                ball_frame_share=summary.get("ball_frame_share"),
            )
        )
    return out


class TuningReport(BaseModel):
    """Result of one tuning run (evals/results/tuning/<timestamp>.json)."""

    created_at: datetime
    git_commit: str | None
    label: str | None = None
    grid: dict[str, list[Any]] = {}
    set_id: str
    clips: list[str]
    tolerance_sec: float
    n_combos: int
    folds: list[Fold]
    folds_ball_loss: dict[str, Counts]
    held_out_pass: MetricRow
    held_out_pass_spread: Spread
    held_out_ball_loss_anecdotal: MetricRow
    final_params_in_sample: dict[str, Any]
    final_pass_in_sample: MetricRow
    final_ball_loss_in_sample: MetricRow
    team_mapping: list[MappingCheck]
    mapping_accuracy: float | None
    held_out_miss_causes: dict[str, int]
    held_out_false_pass_causes: dict[str, int]
    observability: Observability
    by_airborne_mode: list[ModeResult]
    per_clip_pass: list[ClipPassResult]


def compare_airborne_modes(
    tables: dict[str, list[dict[str, Counts]]],
    combos: list[dict[str, Any]],
    tiebreak: list[int],
    clips: dict[str, ClipData],
    cfg: EventsStageConfig,
) -> list[ModeResult]:
    """Run leave-one-clip-out separately within each airborne mode of the grid."""
    out = []
    for mode in cfg.events.grid.get("ball.airborne_mode", []):
        idx = [i for i, c in enumerate(combos) if c["ball.airborne_mode"] == mode]
        folds = leave_one_out(
            [tables["pass"][i] for i in idx],
            list(clips),
            [combos[i] for i in idx],
            [tiebreak[i] for i in idx],
        )
        flagged = observed = 0
        baselines: dict[str, float | None] = {}
        for clip_id, clip in clips.items():
            det = detect_events(clip, cfg.events.with_params({"ball.airborne_mode": mode}))
            flagged += int(det.airborne.sum())
            observed += int((~np.isnan(clip.ball_xy[:, 0])).sum())
            baselines[clip_id] = det.ratio_baseline
        out.append(
            ModeResult(
                mode=mode,
                held_out_pass=held_out_total(folds),
                spread=fold_spread(folds),
                fold_params={f.held_out: f.params for f in folds},
                airborne_share=flagged / max(observed, 1),
                ratio_baselines=baselines,
            )
        )
    return out


def start_values(cfg: EventsStageConfig) -> dict[str, Any]:
    """Current config values of the grid parameters."""
    data = cfg.events.model_dump()
    return {k: data[k.split(".")[0]][k.split(".")[1]] for k in cfg.events.grid}


def load_clips(golden: GoldenSet, cfg: EventsStageConfig) -> dict[str, ClipData]:
    """Load every clip of the split (must be complete in the golden set)."""
    out = {}
    for clip_id in cfg.events.split.clips:
        g = golden.clip(clip_id)
        if g is None or not g.complete:
            raise ValueError(f"Clip {clip_id} is not a complete clip of golden set {golden.set_id}")
        match_dir = cfg.paths.paths.processed_dir / clip_id
        out[clip_id] = load_clip(
            match_dir, Path(g.video), cfg.matches.matches[clip_id].kits, cfg.events
        )
    return out


def score_grid(
    clips: dict[str, ClipData],
    golden: GoldenSet,
    cfg: EventsStageConfig,
    combos: list[dict[str, Any]],
    tol: float,
) -> dict[str, list[dict[str, Counts]]]:
    """Counts per event type, grid combination and clip."""
    tables: dict[str, list[dict[str, Counts]]] = {"pass": [], "ball_loss": []}
    for combo in combos:
        params = cfg.events.with_params(combo)
        rows: dict[str, dict[str, Counts]] = {"pass": {}, "ball_loss": {}}
        for clip_id, clip in clips.items():
            counts = evaluate_clip(golden.clip(clip_id), detect_events(clip, params).events, tol)
            for t in rows:
                rows[t][clip_id] = counts[t]
        for t in tables:
            tables[t].append(rows[t])
    return tables


def mapping_checks(clips: dict[str, ClipData], reference_path: Path) -> list[MappingCheck]:
    """Compare predicted mappings with a visually confirmed reference, if one exists."""
    ref = (
        yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        if reference_path.exists()
        else {}
    )
    checks = []
    for clip_id, clip in clips.items():
        pred = {str(k): v for k, v in clip.mapping.mapping.items()}
        r = {str(k): v for k, v in ref[clip_id].items()} if clip_id in (ref or {}) else None
        checks.append(
            MappingCheck(
                clip_id=clip_id,
                predicted=pred,
                reference=r,
                correct=None if r is None else pred == r,
                margin=round(clip.mapping.margin, 1),
            )
        )
    return checks


def analyse_folds(
    folds: list[Fold],
    clips: dict[str, ClipData],
    golden: GoldenSet,
    cfg: EventsStageConfig,
    checks: list[MappingCheck],
    tol: float,
) -> tuple[Counter[str], Counter[str], Counter[tuple]]:
    """Miss causes, false-pass causes and ball observability over the held-out folds."""
    diag, correct = cfg.events.diagnostics, {c.clip_id: c.correct for c in checks}
    misses: Counter[str] = Counter()
    false_passes: Counter[str] = Counter()
    obs: Counter[tuple] = Counter()
    for fold in folds:
        clip = clips[fold.held_out]
        det = detect_events(clip, cfg.events.with_params(fold.params))
        gold = [e for e in golden.clip(fold.held_out).events if e.type == "pass"]
        preds = [e for e in det.events if e.type == "pass"]
        result = match_events(preds, gold, tol)
        matched = {g for _, g, _ in result.pairs}
        for g in gold:
            raw = observable(
                clip.times, ~np.isnan(clip.ball_xy[:, 0]), g.t_sec, diag.observable_window_sec
            )
            proc = observable(
                clip.times, ~np.isnan(det.ball_xy[:, 0]), g.t_sec, diag.observable_window_sec
            )
            hit = g.id in matched
            obs[("raw", raw, hit)] += 1
            obs[("proc", proc, hit)] += 1
            if not hit:
                misses[miss_cause(g, clip, det, preds, correct.get(fold.held_out), tol, diag)] += 1
        fps = [p for p in preds if p.id in set(result.unmatched_predictions)]
        false_passes.update(false_pass_causes(fps, clip, det, diag))
    return misses, false_passes, obs


def observability(obs: Counter, window: float) -> Observability:
    """Turn (kind, observable, hit) counts into recall per observability group."""

    def n(kind: str, seen: bool) -> int:
        return obs[(kind, seen, True)] + obs[(kind, seen, False)]

    def recall(kind: str, seen: bool) -> float | None:
        total = n(kind, seen)
        return obs[(kind, seen, True)] / total if total else None

    return Observability(
        window_sec=window,
        raw_observable=n("raw", True),
        raw_not_observable=n("raw", False),
        processed_observable=n("proc", True),
        processed_not_observable=n("proc", False),
        recall_raw_observable=recall("raw", True),
        recall_raw_not_observable=recall("raw", False),
        recall_processed_observable=recall("proc", True),
        recall_processed_not_observable=recall("proc", False),
    )


def run_tuning(
    cfg: EventsStageConfig, eval_cfg: EvalConfig, reference_dir: Path, label: str | None = None
) -> TuningReport:
    """Score the grid, run the folds, analyse errors and assemble the report."""
    set_id, tol = cfg.events.split.set_id, eval_cfg.tolerance_sec
    golden = load_golden_set(eval_cfg.golden_dir, set_id)
    clips = load_clips(golden, cfg)
    ids = list(clips)
    combos = grid_combos(cfg.events.grid)
    start = start_values(cfg)
    tiebreak = [distance_to_start(c, start, cfg.events.grid) for c in combos]
    tables = score_grid(clips, golden, cfg, combos, tol)
    folds = leave_one_out(tables["pass"], ids, combos, tiebreak)
    loss = {f.held_out: tables["ball_loss"][f.combo_index][f.held_out] for f in folds}
    final = select_best(tables["pass"], ids, tiebreak)
    checks = mapping_checks(clips, reference_dir / f"team_mapping_{set_id}.yaml")
    known = [c.correct for c in checks if c.correct is not None]
    misses, false_passes, obs = analyse_folds(folds, clips, golden, cfg, checks, tol)
    return TuningReport(
        created_at=datetime.now(UTC).replace(microsecond=0),
        git_commit=git_commit(),
        label=label,
        grid=cfg.events.grid,
        set_id=set_id,
        clips=ids,
        tolerance_sec=tol,
        n_combos=len(combos),
        folds=folds,
        folds_ball_loss=loss,
        held_out_pass=held_out_total(folds),
        held_out_pass_spread=fold_spread(folds),
        held_out_ball_loss_anecdotal=MetricRow.from_counts(sum(loss.values(), Counts())),
        final_params_in_sample=combos[final],
        final_pass_in_sample=MetricRow.from_counts(summed(tables["pass"][final], ids)),
        final_ball_loss_in_sample=MetricRow.from_counts(summed(tables["ball_loss"][final], ids)),
        team_mapping=checks,
        mapping_accuracy=sum(known) / len(known) if known else None,
        held_out_miss_causes={c: misses[c] for c in MISS_CAUSES},
        held_out_false_pass_causes={c: false_passes[c] for c in ("track_break", "other")},
        observability=observability(obs, cfg.events.diagnostics.observable_window_sec),
        by_airborne_mode=compare_airborne_modes(tables, combos, tiebreak, clips, cfg),
        per_clip_pass=clip_pass_results(
            folds, tables["pass"], final, golden, cfg.paths.paths.processed_dir
        ),
    )


def write_report(report: TuningReport, results_dir: Path) -> Path:
    """Write the report to <results_dir>/tuning/<YYYYMMDDTHHMMSSZ>.json."""
    out = results_dir / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report.created_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
