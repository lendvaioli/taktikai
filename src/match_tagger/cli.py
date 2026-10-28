"""Command-line entry point: ``python -m match_tagger.cli run --stage <stage> --video <path>``."""

import argparse
import sys
from pathlib import Path

from match_tagger.config import CONFIG_DIR

IMPLEMENTED_STAGES = ("detection", "pitch", "events")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="match_tagger")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one pipeline stage.")
    run.add_argument("--stage", required=True)
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--match", help="Match id; defaults to the video file name without suffix.")
    run.add_argument("--start", type=float, help="Override video.start_sec.")
    run.add_argument("--duration", type=float, help="Override video.duration_sec.")
    run.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    return parser.parse_args(argv)


def run_stage(args: argparse.Namespace) -> int:
    """Run the requested stage; return a process exit code."""
    if args.stage not in IMPLEMENTED_STAGES:
        print(f"Stage {args.stage!r} is not implemented yet.", file=sys.stderr)
        return 2
    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1
    if args.stage == "detection":
        from match_tagger.detection.stage import StageConfig as Config
        from match_tagger.detection.stage import run_detection as run
    elif args.stage == "pitch":
        from match_tagger.pitch.stage import PitchStageConfig as Config
        from match_tagger.pitch.stage import run_pitch as run
    else:
        from match_tagger.events.stage import EventsStageConfig as Config
        from match_tagger.events.stage import run_events as run

    cfg = Config.load(args.config_dir)
    overrides = {"start_sec": args.start, "duration_sec": args.duration}
    cfg.video = cfg.video.model_validate(
        {**cfg.video.model_dump(), **{k: v for k, v in overrides.items() if v is not None}}
    )
    summary = run(args.video, args.match or args.video.stem, cfg)
    print(summary.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return run_stage(args)


if __name__ == "__main__":
    sys.exit(main())
