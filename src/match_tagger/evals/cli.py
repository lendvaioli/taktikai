"""Evaluation CLI: ``python -m match_tagger.evals.cli eval [--set ID]`` and ``compare A B``."""

import argparse
import sys
from pathlib import Path

from match_tagger.config import CONFIG_DIR, load_config
from match_tagger.evals.golden import list_golden_sets, load_golden_set
from match_tagger.evals.report import compare_runs, compare_table, set_table
from match_tagger.evals.run import (
    EvalConfig,
    EvalRun,
    evaluate_set,
    file_prediction_loader,
    new_run,
    write_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="match_tagger.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="Evaluate predictions against golden sets.")
    ev.add_argument("--set", dest="set_id", help="Golden set id; default: all sets.")
    ev.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    cmp = sub.add_parser("compare", help="Per-metric difference between two result files.")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
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


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.command == "eval":
        return run_eval(args.set_id, args.config_dir)
    return run_compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
