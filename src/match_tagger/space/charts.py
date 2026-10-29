"""Space charts: pass networks and ball losses per third, each stamped with the StatsBomb logo.

The logo is drawn into every image and its file name is written into the PNG metadata, which
tests/test_publication.py checks. matplotlib and mplsoccer come with the ``ml`` extra.
"""

from pathlib import Path

from match_tagger.space.content import NetworkView, TeamRow

THIRDS_HU = {"own": "Saját harmad", "middle": "Középső harmad", "final": "Támadóharmad"}
COLOURS = ("#1f4e9c", "#c0392b")  # home, away


def _stamp_and_save(fig: object, logo: Path, out: Path, attribution: str) -> Path:
    """Draw the logo bottom right, save with the logo name in the PNG metadata."""
    import matplotlib.image as mpimg

    image = mpimg.imread(logo)
    height = 0.05
    width = height * image.shape[1] / image.shape[0] * fig.get_figheight() / fig.get_figwidth()
    ax = fig.add_axes((0.99 - width, 0.01, width, height))
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.01, 0.015, attribution, fontsize=7, color="#555555")
    fig.savefig(
        out,
        dpi=150,
        metadata={"Description": f"{logo.name} stamped; {attribution}", "Software": "TaktikAI"},
    )
    return out


def _upper_of_close_pair(name: str, view: NetworkView, within_m: float = 8.0) -> bool:
    """True if another node lies within ``within_m`` below this one (its label goes above)."""
    x, y, _ = view.nodes[name]
    return any(
        other != name and ((ox - x) ** 2 + (oy - y) ** 2) ** 0.5 < within_m and oy < y
        for other, (ox, oy, _) in view.nodes.items()
    )


def pass_networks(views: list[NetworkView], logo: Path, out: Path, attribution: str) -> Path:
    """One pitch per team (attacking right), nodes at mean pass positions, top links as lines."""
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#888888")
    fig, axes = pitch.draw(nrows=1, ncols=len(views), figsize=(6.5 * len(views), 5.2))
    for ax, view, colour in zip(list(getattr(axes, "flat", [axes])), views, COLOURS, strict=False):
        top = max(c for _, _, c in view.edges) if view.edges else 1
        for s, t, count in view.edges:
            (x1, y1, _), (x2, y2, _) = view.nodes[s], view.nodes[t]
            pitch.lines(x1, y1, x2, y2, ax=ax, lw=1 + 5 * count / top, color=colour, alpha=0.5)
        for name, (x, y, passes) in view.nodes.items():
            pitch.scatter(
                x, y, s=60 + 4 * passes, ax=ax, color=colour, edgecolors="white", zorder=3
            )
            dy = 9 if _upper_of_close_pair(name, view) else -12
            ax.annotate(
                name, (x, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=6
            )
        ax.set_title(
            f"{view.team_hu}: a {len(view.edges)} leggyakoribb passzkapcsolat", fontsize=10
        )
    fig.suptitle("Passzhálózat (támadási irány: jobbra; pont: átlagos passzhely)", fontsize=11)
    path = _stamp_and_save(fig, logo, out, attribution)
    plt.close(fig)
    return path


def losses_by_third(rows: list[TeamRow], logo: Path, out: Path, attribution: str) -> Path:
    """Grouped bars: ball losses per third of the losing team's attack, per team."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    for i, (row, colour) in enumerate(zip(rows, COLOURS, strict=False)):
        values = [row.losses_by_third.get(k, 0) for k in THIRDS_HU]
        xs = [j + (i - 0.5) * width for j in range(len(THIRDS_HU))]
        bars = ax.bar(xs, values, width, label=row.name_hu, color=colour)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(range(len(THIRDS_HU)), list(THIRDS_HU.values()))
    ax.set_ylabel("Labdavesztések (db)")
    ax.set_title(
        "Labdavesztések harmadonként (a labdát vesztő csapat támadási irányában)", fontsize=10
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.18)
    path = _stamp_and_save(fig, logo, out, attribution)
    plt.close(fig)
    return path
