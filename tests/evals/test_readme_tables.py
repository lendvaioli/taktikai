from datetime import UTC, datetime

import pytest

from match_tagger.evals.metrics import Counts, MetricRow
from match_tagger.evals.readme_tables import events_table, models_table, replace_block
from match_tagger.evals.run import ClipResult, EvalRun, SetResult

MODELS_MD = """# Models

| model or dataset | source URL | licence | used in |
|------------------|------------|---------|---------|
| weights.pt | https://x | **Non-commercial use only.** See caveats below. | detection |
| some data | https://y | CC BY 4.0 according to search results; page not opened. | pitch |

Text between the tables.

| asset | source | terms | used in |
|---|---|---|---|
| logo.png | portal | Required. | Space |
"""


def _row(tp, fp, fn):
    return MetricRow.from_counts(Counts(tp=tp, fp=fp, fn=fn))


def test_models_table_reads_only_the_first_table_and_cuts_after_a_sentence():
    table = models_table(MODELS_MD).splitlines()
    assert table[2] == "| weights.pt | **Non-commercial use only.** | detection |"
    assert (
        table[3]
        == "| some data | CC BY 4.0 according to search results; page not opened. | pitch |"
    )
    assert len(table) == 4  # the asset table is not included


def test_events_table_has_aggregates_only_and_skips_unpredicted_types():
    clip = ClipResult(clip_id="secret_clip", predictions_found=True, counts={"pass": Counts(tp=1)})
    run = EvalRun(
        created_at=datetime(2026, 10, 29, tzinfo=UTC), git_commit="abc1234", tolerance_sec=2.0,
        sets=[SetResult(
            set_id="dfl", source="s", n_clips_evaluated=5, skipped_incomplete=[],
            missing_predictions=[], per_type={"pass": _row(32, 4, 11), "shot": _row(0, 0, 1)},
            overall=_row(32, 4, 12), clips=[clip],
        )],
    )  # fmt: skip
    table = events_table(run)
    assert "| dfl (5 clips) | pass | 43 | 0.889 | 0.744 | 0.810 |" in table
    assert "shot" not in table.split("\n\n")[0]  # no predictions -> no row
    assert "secret_clip" not in table  # per-clip results are never written


def test_replace_block_between_markers():
    text = "a\n<!-- t:start -->\nold\n<!-- t:end -->\nb"
    assert replace_block(text, "t", "new") == "a\n<!-- t:start -->\nnew\n<!-- t:end -->\nb"
    with pytest.raises(ValueError, match="no <!-- x:start -->"):
        replace_block(text, "x", "new")


def test_dataset_schema_matches_the_golden_set_model():
    import json
    from pathlib import Path

    from match_tagger.evals.golden import GoldenSet

    path = Path(__file__).resolve().parents[2] / "dataset" / "golden_set.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == GoldenSet.model_json_schema()
