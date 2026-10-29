"""Build the static Space: index.html, charts and the Hugging Face README into space/.

Inputs (local only, never published themselves): data/processed/<match>/metrics.json and the
accepted report run record. Outputs: only allow-listed files (tests/test_publication.py).
"""

from pathlib import Path

from match_tagger.config import CONFIG_DIR, DefaultConfig, load_config
from match_tagger.metrics.stage import MatchMetrics
from match_tagger.report.runs import RunRecord, runs_dir
from match_tagger.space.charts import losses_by_third, pass_networks
from match_tagger.space.content import SpaceConfig, SpaceContent, build_content
from match_tagger.space.page import render_page

SPACE_README = """---
title: TaktikAI
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: true
license: agpl-3.0
short_description: Football match analysis for club video analysts (demo)
---

# TaktikAI – demo

Static demo of [TaktikAI]({repo}): tactical metrics and a validated coaching report (Hungarian)
for one full match of StatsBomb Open Data. Numbers are computed by code; the report text is
written by a local open model and checked against the tool results.

Data: StatsBomb Open Data (github.com/statsbomb/open-data), used under the StatsBomb Public Data
User Agreement for non-commercial analysis; this Space contains aggregates only, no raw data.
Code: AGPL-3.0-or-later.
"""


def load_inputs(cfg: SpaceConfig, processed_dir: Path) -> tuple[MatchMetrics, RunRecord]:
    """metrics.json of the match and the configured run record."""
    match_dir = processed_dir / cfg.match_id
    metrics = MatchMetrics.model_validate_json((match_dir / "metrics.json").read_text("utf-8"))
    record_path = runs_dir(match_dir) / f"{cfg.run_id}.json"
    return metrics, RunRecord.model_validate_json(record_path.read_text("utf-8"))


def write_charts(content: SpaceContent, cfg: SpaceConfig, assets: Path) -> dict[str, str]:
    """Render the charts into assets/; returns alt text -> file name."""
    logo = assets / cfg.logo_light
    charts = {}
    if content.networks:
        path = pass_networks(
            list(content.networks.values()),
            logo,
            assets / "chart_pass_network.png",
            content.attribution,
        )
        charts["Passzhálózat csapatonként"] = path.name
    path = losses_by_third(
        list(content.rows.values()), logo, assets / "chart_ball_losses.png", content.attribution
    )
    charts["Labdavesztések harmadonként"] = path.name
    return charts


def build(config_dir: Path = CONFIG_DIR) -> Path:
    """Build the Space; returns the path of index.html."""
    cfg = load_config("space", SpaceConfig, config_dir)
    processed = load_config("default", DefaultConfig, config_dir).paths.processed_dir
    content = build_content(*load_inputs(cfg, processed), cfg)
    assets = cfg.out_dir / "assets"
    for logo in (cfg.logo_light, cfg.logo_dark):
        if not (assets / logo).exists():
            raise FileNotFoundError(f"StatsBomb logo missing: {assets / logo} (media pack)")
    charts = write_charts(content, cfg, assets)
    (cfg.out_dir / "README.md").write_text(SPACE_README.format(repo=cfg.repo_url), encoding="utf-8")
    index = cfg.out_dir / "index.html"
    index.write_text(render_page(content, cfg, charts), encoding="utf-8")
    return index
