"""Evaluation CLI.

``python -m match_tagger.evals.cli eval [--set ID]``, ``compare A B``, ``tune`` and
``team-sheets``.
"""

import argparse
import sys
from pathlib import Path

from match_tagger.config import CONFIG_DIR, load_config
from match_tagger.evals.golden import list_golden_sets, load_golden_set
from match_tagger.evals.report import compare_runs, compare_table, fmt, set_table
from match_tagger.evals.run import (
    EvalConfig,
    EvalRun,
    evaluate_set,
    file_prediction_loader,
    new_run,
    write_run,
)

REFERENCE_DIR = Path("evals/reference")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="match_tagger.evals")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="Evaluate predictions against golden sets.")
    ev.add_argument("--set", dest="set_id", help="Golden set id; default: all sets.")
    cmp = sub.add_parser("compare", help="Per-metric difference between two result files.")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    tune = sub.add_parser(
        "tune", help="Leave-one-clip-out tuning of events v1 (configs/events.yaml)."
    )
    tune.add_argument("--label", help="Short name stored in the report, e.g. a variant name.")
    sub.add_parser("team-sheets", help="Contact sheets of the team-colour mapping per clip.")
    return parser.parse_args(argv)


def run_eval(set_id: str | None, config_dir: Path) -> int:
    """Evaluate one or all golden sets, print tables and write the result file."""
    cfg = load_config("evals", EvalConfig, config_dir)
    set_ids = [set_id] if set_id else list_golden_sets(cfg.golden_dir)
    if not set_ids:
        print(f"No golden sets in {cfg.golden_dir}.", file=sys.stderr)
        return 1
    if set_id and set_id not in list_golden_sets(cfg.golden_dir):
        print(f"Golden set not found: {set_id}", file=sys.stderr)
        return 1
    loader = file_prediction_loader(cfg.predictions)
    results = [
        evaluate_set(load_golden_set(cfg.golden_dir, s), loader, cfg.tolerance_sec) for s in set_ids
    ]
    for result in results:
        print(set_table(result), end="\n\n")
    path = write_run(new_run(results, cfg.tolerance_sec), cfg.results_dir)
    print(f"Written: {path}")
    return 0


def run_compare(a: Path, b: Path) -> int:
    """Print the per-metric difference B - A between two result files."""
    runs = [EvalRun.model_validate_json(p.read_text(encoding="utf-8")) for p in (a, b)]
    print(compare_table(compare_runs(*runs)))
    return 0


def run_tune(config_dir: Path, label: str | None = None) -> int:
    """Run leave-one-clip-out tuning, print the summary and write the report."""
    from match_tagger.evals.tuning_run import run_tuning, write_report
    from match_tagger.events.stage import EventsStageConfig

    cfg = EventsStageConfig.load(config_dir)
    eval_cfg = load_config("evals", EvalConfig, config_dir)
    report = run_tuning(cfg, eval_cfg, REFERENCE_DIR, label)
    print(f"{'held out':<10} {'P':>6} {'R':>6} {'F1':>6}  params")
    for f in report.folds:
        print(
            f"{f.held_out:<10} {fmt(f.held.precision):>6} {fmt(f.held.recall):>6} "
            f"{fmt(f.held.f1):>6}  {f.params}"
        )
    print(
        f"{'clip':<10} {'gold':>4} {'held R':>6} {'held F1':>7} {'in R':>6} {'in F1':>6} "
        f"{'tracks':>6} {'med s':>6} {'ball':>5}"
    )
    for c in report.per_clip_pass:
        print(
            f"{c.clip_id:<10} {c.golden_passes:>4} {fmt(c.held_out.recall):>6} "
            f"{fmt(c.held_out.f1):>7} {fmt(c.in_sample.recall):>6} {fmt(c.in_sample.f1):>6} "
            f"{c.n_player_tracks:>6} {c.median_player_track_sec or 0:>6.1f} "
            f"{c.ball_frame_share or 0:>5.2f}"
        )
    h, s = report.held_out_pass, report.held_out_pass_spread
    print(
        f"pass, held-out micro: P {fmt(h.precision)} R {fmt(h.recall)} F1 {fmt(h.f1)} "
        f"(TP {h.tp} FP {h.fp} FN {h.fn}); fold F1 mean {fmt(s.mean)} std {fmt(s.std)}"
    )
    print(f"final params (in-sample, NOT held-out): {report.final_params_in_sample}")
    print(f"{'airborne mode':<14} {'P':>6} {'R':>6} {'F1':>6} {'F1 std':>7} {'flagged':>8}")
    for m in report.by_airborne_mode:
        r = m.held_out_pass
        print(
            f"{m.mode:<14} {fmt(r.precision):>6} {fmt(r.recall):>6} {fmt(r.f1):>6} "
            f"{fmt(m.spread.std):>7} {m.airborne_share:>8.1%}"
        )
    print(f"written: {write_report(report, eval_cfg.results_dir)}")
    return 0


def run_team_sheets(config_dir: Path) -> int:
    """Write data/processed/<clip>/team_sheet.jpg for every clip of the tuning split."""
    import cv2

    from match_tagger.events.sheets import team_sheet
    from match_tagger.events.stage import EventsStageConfig, load_clip

    cfg = EventsStageConfig.load(config_dir)
    golden = load_golden_set(
        load_config("evals", EvalConfig, config_dir).golden_dir, cfg.events.split.set_id
    )
    for clip_id in cfg.events.split.clips:
        video, match_dir = Path(golden.clip(clip_id).video), cfg.paths.paths.processed_dir / clip_id
        kits = cfg.matches.matches[clip_id].kits
        clip = load_clip(match_dir, video, kits, cfg.events)
        sheet = team_sheet(video, match_dir, clip.mapping, kits, cfg.events.team_mapping.seed)
        cv2.imwrite(str(match_dir / "team_sheet.jpg"), sheet)
        print(
            f"{clip_id}: {clip.mapping.mapping} margin {clip.mapping.margin:.1f} -> "
            f"{match_dir / 'team_sheet.jpg'}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.command == "eval":
        return run_eval(args.set_id, args.config_dir)
    if args.command == "compare":
        return run_compare(args.a, args.b)
    if args.command == "tune":
        return run_tune(args.config_dir, args.label)
    return run_team_sheets(args.config_dir)


if __name__ == "__main__":
    sys.exit(main())
