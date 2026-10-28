"""The report's sample note, written by code from the sample block (Hungarian template).

The model never writes it: duration, event counts and reliability are facts, so they are
rendered exactly as metrics.json has them.
"""

from match_tagger.metrics.sample import Sample

TYPE_HU = {"pass": "passz", "ball_loss": "labdavesztés", "shot": "lövés", "set_piece": "pontrúgás"}
RELIABILITY_HU = {"very_low": "nagyon alacsony", "low": "alacsony", "ok": "megfelelő"}
MEANING_HU = {
    "very_low": "rövid részlet, a megállapítások videón ellenőrizendő megfigyelések, "
    "nem a csapat jellemző mintázatai",
    "low": "részleges mérkőzés, a megállapításokat óvatosan kell kezelni",
    "ok": "egy teljes mérkőzésnyi adat",
}


def _number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def duration_hu(seconds: float) -> str:
    """Seconds below two minutes ("30 másodperc"), else minutes with decimal comma ("95,1 perc")."""
    if seconds < 120:
        return f"{_number(round(seconds))} másodperc"
    return f"{_number(round(seconds / 60, 1))} perc"


def sample_note(sample: Sample) -> str:
    """E.g. "30 másodperc, 18 esemény (12 labdavesztés, 6 passz). Megbízhatóság: ..."."""
    per_type = ", ".join(
        f"{n} {TYPE_HU.get(t, t)}"
        for t, n in sorted(sample.events_per_type.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    counts = f"{sample.n_events} esemény" + (f" ({per_type})" if per_type else "")
    return (
        f"{duration_hu(sample.duration_sec)}, {counts}. Megbízhatóság: "
        f"{RELIABILITY_HU[sample.reliability]} – {MEANING_HU[sample.reliability]}."
    )
