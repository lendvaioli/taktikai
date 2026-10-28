"""Hungarian Markdown rendering of a validated report."""

from match_tagger.metrics.context import MatchContext, team_labels_hu
from match_tagger.metrics.sample import Sample
from match_tagger.report.sample_note import sample_note
from match_tagger.report.schema import Report, SupportingNumber


def _number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def _number_line(n: SupportingNumber) -> str:
    unit = f" {n.unit}" if n.unit else ""
    return f"- {n.label}: {_number(n.value)}{unit} *(forrás: `{n.source.describe()}`)*"


def render_markdown(report: Report, ctx: MatchContext, sample: Sample, generated_by: str) -> str:
    """report.md: sample note (written by code), findings with sourced numbers, clips, training."""
    teams = team_labels_hu(ctx)
    lines = [
        f"# Elemzői riport: {teams['home']} – {teams['away']}",
        "",
        f"**Mintaméret:** {sample_note(sample)}",
        "",
    ]
    for i, f in enumerate(report.findings, 1):
        lines += [f"## {i}. {f.title}", "", f.claim, "", "**Számok:**"]
        lines += [_number_line(n) for n in f.numbers]
        if f.clip_ids:
            links = ", ".join(f"[{c}](../../clips/{c}.mp4)" for c in f.clip_ids)
            lines += ["", f"**Klipek:** {links}"]
        lines += ["", f"**Edzésjavaslat:** {f.training}", ""]
    lines += [
        "---",
        f"*Készítette: {generated_by}. A számokat kód számolta, a szöveget nyelvi modell írta.*",
    ]
    if ctx.attribution:
        lines.append(f"*{ctx.attribution}*")
    return "\n".join(lines) + "\n"
