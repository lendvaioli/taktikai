"""Regenerate the tables in README.md between their marker comments (make readme)."""

import sys
from pathlib import Path

from match_tagger.config import CONFIG_DIR, DefaultConfig, load_config
from match_tagger.evals.readme_tables import (
    events_table,
    latest_run,
    models_table,
    replace_block,
    report_table,
)
from match_tagger.evals.run import EvalConfig
from match_tagger.report.runs import all_records
from match_tagger.report.score import SCORES_FILE, read_scores


def main() -> int:
    """Entry point: python scripts/readme_tables.py."""
    evals = load_config("evals", EvalConfig, CONFIG_DIR)
    processed = load_config("default", DefaultConfig, CONFIG_DIR).paths.processed_dir
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    text = replace_block(text, "results:events", events_table(latest_run(evals.results_dir)))
    scores = read_scores(evals.results_dir / SCORES_FILE)
    text = replace_block(text, "results:report", report_table(all_records(processed), scores))
    models = Path("docs/MODELS.md").read_text(encoding="utf-8")
    text = replace_block(text, "models", models_table(models))
    readme.write_text(text, encoding="utf-8")
    print("README.md tables updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
