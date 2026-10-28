"""Hungarian Markdown rendering of a validated report."""

from match_tagger.metrics.context import MatchContext
from match_tagger.report.schema import Report


def _number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def render_markdown(report: Report, ctx: MatchContext, generated_by: str) -> str:
    """report.md: sample note, findings with numbers, clip links and training focus."""
    lines = [
        f"# Elemzői riport: {ctx.teams['home']} – {ctx.teams['away']}",
        "",
        f"**Mintaméret:** {report.sample_note}",
        "",
    ]
    for i, f in enumerate(report.findings, 1):
        lines += [f"## {i}. {f.title}", "", f.claim, "", "**Számok:**"]
        lines += [
            f"- {n.label}: {_number(n.value)}{' ' + n.unit if n.unit else ''}" for n in f.numbers
        ]
        if f.clip_ids:
            links = ", ".join(f"[{c}](../clips/{c}.mp4)" for c in f.clip_ids)
            lines += ["", f"**Klipek:** {links}"]
        lines += ["", f"**Edzésjavaslat:** {f.training}", ""]
    lines += [
        "---",
        f"*Készítette: {generated_by}. A számokat kód számolta, a szöveget nyelvi modell írta.*",
    ]
    if ctx.attribution:
        lines.append(f"*{ctx.attribution}*")
    return "\n".join(lines) + "\n"
