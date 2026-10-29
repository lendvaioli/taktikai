from datetime import UTC, datetime

import pytest

from match_tagger.events.schema import Event
from match_tagger.metrics.config import MetricsConfig
from match_tagger.metrics.context import MatchContext
from match_tagger.metrics.stage import compute_metrics
from match_tagger.report.runs import RunRecord
from match_tagger.space.content import SpaceConfig, build_content, network_view
from match_tagger.space.page import num, pct, render_page

CFG = SpaceConfig(
    match_id="sb_1", run_id="r1", title_hu="Döntő", logo_light="hudl-statsbomb-logo-black.png",
    logo_dark="hudl-statsbomb-logo-white.png", top_edges=2, repo_url="https://example.org/repo",
    report_note_hu="Kézi megjegyzés.",
)  # fmt: skip


def _ctx(source="statsbomb", attribution="Adatforrás: StatsBomb Open Data"):
    return MatchContext(
        match_id="sb_1", source=source, duration_sec=5400, teams={"home": "A", "away": "B"},
        teams_hu={"home": "Ahon", "away": "Bhon"}, has_video=False, home_attacks="right",
        attribution=attribution,
    )  # fmt: skip


def _pass(i, player, recipient, x, y):
    return Event(id=f"p{i}", t_sec=float(i), type="pass", team="home", x_m=x, y_m=y,
                 player=player, recipient=recipient, outcome="complete")  # fmt: skip


# A -> B three times, B -> A twice, C -> A once: with top_edges 2, C drops out.
EVENTS = [
    *[_pass(i, "A", "B", 30, 20) for i in range(3)],
    *[_pass(10 + i, "B", "A", 50, 40) for i in range(2)],
    _pass(20, "C", "A", 70, 30),
]


def _record(status="ok", match_id="sb_1"):
    finding = {
        "title": "Cím <b>", "claim": "Állítás & más", "clip_ids": [], "training": "Gyakorlat",
        "numbers": [{"label": "passzok", "value": 6, "unit": None, "source": {
            "tool": "get_metric", "arguments": {"name": "summary", "team": "home"},
            "path": "summary.passes"}}],
    }  # fmt: skip
    return RunRecord(
        run_id="r1", created_at=datetime(2026, 10, 29, tzinfo=UTC), match_id=match_id,
        backend="ollama", model="qwen", prompt_version="report_system v3.1", prompt_sha256="x",
        status=status, steps=4, errors=[],
        report={"match_id": match_id, "findings": [finding, finding]} if status == "ok" else None,
    )  # fmt: skip


def _metrics(ctx=None):
    return compute_metrics(EVENTS, ctx or _ctx(), MetricsConfig())


def test_network_view_keeps_top_edges_and_their_players():
    view = network_view(_metrics(), "home", top=2)
    assert view.edges == [("A", "B", 3), ("B", "A", 2)]
    assert set(view.nodes) == {"A", "B"}  # C only appears in a dropped edge
    assert view.nodes["A"] == (30.0, 20.0, 3)  # mean pass position and passes made


def test_content_refuses_non_statsbomb_or_rejected_input():
    with pytest.raises(ValueError, match="StatsBomb"):
        build_content(_metrics(_ctx(source="pipeline")), _record(), CFG)
    with pytest.raises(ValueError, match="StatsBomb"):
        build_content(_metrics(_ctx(attribution=None)), _record(), CFG)
    with pytest.raises(ValueError, match="not an accepted report"):
        build_content(_metrics(), _record(status="rejected"), CFG)
    with pytest.raises(ValueError, match="not an accepted report"):
        build_content(_metrics(), _record(match_id="other"), CFG)


def test_page_has_logo_escapes_model_text_and_shows_sources():
    content = build_content(_metrics(), _record(), CFG)
    html = render_page(content, CFG, {"Ábra": "chart_x.png"})
    assert html.count("assets/hudl-statsbomb-logo-black.png") == 2  # header and footer
    assert "Cím &lt;b&gt;" in html and "Állítás &amp; más" in html and "<b>" not in html
    assert "summary.passes" in html and "Kézi megjegyzés." in html
    assert "<h1>Ahon – Bhon</h1>" in html and ">A<" not in html  # Hungarian team labels
    assert "<script" not in html


def test_hungarian_number_format():
    assert (num(34.1), num(10.05), num(1.10), num(None), num(8, 0)) == (
        "34,1",
        "10,05",
        "1,1",
        "–",
        "8",
    )
    assert (pct(0.703), pct(None)) == ("70,3 %", "–")


def test_charts_carry_the_logo_in_the_png_metadata(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    from match_tagger.space.charts import _stamp_and_save

    logo = tmp_path / "hudl-statsbomb-logo-black.png"
    plt.imsave(logo, [[[0.0, 0.0, 0.0, 1.0]] * 4] * 2)  # a 4 x 2 black image as a stand-in logo
    fig, _ = plt.subplots(figsize=(2, 1))
    out = _stamp_and_save(fig, logo, tmp_path / "chart_x.png", "Adatforrás: StatsBomb")
    assert b"hudl-statsbomb-logo-black.png stamped" in out.read_bytes()
