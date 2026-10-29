"""The Space's index.html (Hungarian, plain HTML and CSS, no scripts).

Every text from data or from the model is escaped. The StatsBomb logo appears in the header
and next to the attribution in the footer; charts carry it in the image.
"""

from html import escape

from match_tagger.space.content import SpaceConfig, SpaceContent, TeamRow

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
  color:#1d1d1f;background:#fafafa}
main{max-width:980px;margin:0 auto;padding:16px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
header img{height:28px}
h1{font-size:1.6rem;margin:8px 0}
h2{font-size:1.2rem;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px 16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:right}
th:first-child,td:first-child{text-align:left}
img.chart{width:100%;height:auto;border:1px solid #eee;border-radius:6px;background:#fff}
.source{color:#666;font-size:.8rem}
.placeholder{border:2px dashed #bbb;border-radius:8px;padding:24px;text-align:center;color:#555}
footer{margin-top:40px;font-size:.85rem;color:#555;display:flex;gap:12px;
  align-items:center;flex-wrap:wrap}
footer img{height:20px}
@media (max-width:600px){h1{font-size:1.3rem}th,td{padding:4px}}
"""


def num(value: float | None, decimals: int = 2) -> str:
    """Hungarian number: decimal comma, "–" for unknown."""
    if value is None:
        return "–"
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{value:.0f}"
    return text.replace(".", ",")


def pct(share: float | None) -> str:
    """A share as Hungarian percentage with one decimal."""
    return "–" if share is None else f"{num(100 * share, 1)} %"


def _metrics_table(rows: list[TeamRow]) -> str:
    lines = [
        ("Passzok (nyílt játékból)", [str(r.passes) for r in rows]),
        ("Passzpontosság", [pct(r.completion_share) for r in rows]),
        ("Labdavesztések", [str(r.ball_losses) for r in rows]),
        ("… saját harmadban", [str(r.losses_by_third.get("own", 0)) for r in rows]),
        ("… középső harmadban", [str(r.losses_by_third.get("middle", 0)) for r in rows]),
        ("… támadóharmadban", [str(r.losses_by_third.get("final", 0)) for r in rows]),
        (
            "Letámadási mutató (PPDA-proxy, kisebb = erősebb letámadás)",
            [num(r.pressing) for r in rows],
        ),
        ("Lövések", [num(r.shots, 0) for r in rows]),
        ("Gólok lövésből", [num(r.shot_goals, 0) for r in rows]),
        ("Várható gólok (xG) összesen", [num(r.xg_total) for r in rows]),
    ]
    head = "".join(f"<th>{escape(r.name_hu)}</th>" for r in rows)
    body = "".join(
        f"<tr><td>{escape(label)}</td>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"
        for label, values in lines
    )
    return f"<table><thead><tr><th>Mutató</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def _source(src: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in src.get("arguments", {}).items())
    return f"{src.get('tool')}({args}) → {src.get('path')}"


def _report(c: SpaceContent) -> str:
    parts = [f"<p><strong>Mintaméret:</strong> {escape(c.sample_note_hu)}</p>"]
    for i, f in enumerate(c.report["findings"], 1):
        numbers = "".join(
            f"<li>{escape(n['label'])}: <strong>{num(n['value'], 3)}</strong>"
            f"{' ' + escape(n['unit']) if n.get('unit') else ''} "
            f"<span class='source'>(forrás: {escape(_source(n['source']))})</span></li>"
            for n in f["numbers"]
        )
        parts.append(
            f"<div class='card'><h3>{i}. {escape(f['title'])}</h3><p>{escape(f['claim'])}</p>"
            f"<ul>{numbers}</ul>"
            f"<p><strong>Edzésjavaslat:</strong> {escape(f['training'])}</p></div>"
        )
    return "".join(parts)


def _how_it_works(c: SpaceContent, cfg: SpaceConfig) -> str:
    run = c.run
    return (
        "<ul>"
        "<li>Minden számot kód számol (Python), a StatsBomb eseményadatokból.</li>"
        f"<li>A riport szövegét egy helyben futó nyílt modell írta ({escape(run['model'])}, "
        f"{escape(run['prompt_version'] or '')}, {run['steps']} lépés), eszközhívásokkal.</li>"
        "<li>Egy ellenőrző elutasítja a riportot, ha egy szám nem szerepel az eszközök "
        "kimenetében, ha a forrásként megadott mező értéke eltér, ha egy összehasonlítás iránya "
        "hibás, vagy ha angol szó, kulcs vagy fordítatlan csapatnév marad benne.</li>"
        f"<li>Forráskód: <a href='{escape(cfg.repo_url)}'>{escape(cfg.repo_url)}</a> "
        "(AGPL-3.0).</li>"
        "</ul>"
    )


def render_page(c: SpaceContent, cfg: SpaceConfig, charts: dict[str, str]) -> str:
    """The whole page; ``charts`` maps chart keys to file names under assets/."""
    teams = f"{c.teams_hu['home']} – {c.teams_hu['away']}"
    images = "".join(
        f"<img class='chart' src='assets/{escape(name)}' alt='{escape(alt)}'>"
        for alt, name in charts.items()
    )
    logo = f"assets/{escape(cfg.logo_light)}"
    return f"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TaktikAI – mérkőzéselemzés</title><style>{CSS}</style></head>
<body><main>
<header><div><div class="source">TaktikAI · automatikus mérkőzéselemzés videóelemzőknek</div>
<h1>{escape(teams)}</h1><div>{escape(c.title_hu)} · {escape(c.duration_hu)}
· megbízhatóság: {escape(c.reliability_hu)}</div></div>
<img src="{logo}" alt="Hudl StatsBomb logó"></header>
<p>Ez a demó a TaktikAI elemzési lépéseit mutatja be egy teljes mérkőzés nyílt eseményadatain:
a számokat kód számolja, a szöveges riportot egy helyben futó nyelvi modell írja,
ellenőrzés mellett.</p>
<h2>Fő mutatók</h2><div class="card">{_metrics_table(list(c.rows.values()))}</div>
<h2>Ábrák</h2>{images}
<h2>Edzői riport (gépi, ellenőrzött)</h2>{_report(c)}
<p class="source"><strong>Megjegyzés:</strong> {escape(cfg.report_note_hu)}</p>
<h2>Videós demó</h2><div class="placeholder">Videós demó – hamarosan, saját felvételből,
a klub írásos
hozzájárulásával: minitérkép, szűrhető eseménylista klipekkel és a riport.</div>
<h2>Hogyan készült?</h2>{_how_it_works(c, cfg)}
<footer><img src="{logo}" alt="Hudl StatsBomb logó"><span>{escape(c.attribution)}.
Az adatok kutatási és nem kereskedelmi célra használhatók; a nyers adatokat ez az oldal nem
tartalmazza.</span></footer>
</main></body></html>
"""
