import numpy as np
import pytest

from match_tagger.config import load_config
from match_tagger.events.config import EventsConfig, MatchesConfig
from match_tagger.events.teams import kit_colour, map_clusters, parse_kit

PALETTE = EventsConfig().team_mapping.palette or {"fehér": (240, 240, 240), "piros": (200, 30, 40)}


def _crops(bgr, n=3):
    crop = np.zeros((100, 40, 3), np.uint8)
    crop[:] = (60, 140, 60)  # grass around the player
    crop[20:45, 10:30] = bgr  # kit region (20-45 % height, middle 50 % width)
    return [crop] * n


def test_kit_colour_reads_the_upper_body_only():
    white = kit_colour(_crops((240, 240, 240)))
    assert white[0] == pytest.approx(94.7, abs=1.0)  # Lab L* of sRGB 240 grey
    assert abs(white[1]) < 1 and abs(white[2]) < 1


def test_white_and_red_clusters_map_to_their_kits():
    palette = {"fehér": (240, 240, 240), "piros": (200, 30, 40)}
    lab = {0: kit_colour(_crops((240, 240, 240))), 1: kit_colour(_crops((40, 30, 200)))}  # BGR red
    kits = {"home": parse_kit("fehér", palette), "away": parse_kit("piros", palette)}
    assert map_clusters(lab, kits).mapping == {0: "home", 1: "away"}
    swapped = {"home": kits["away"], "away": kits["home"]}
    result = map_clusters(lab, swapped)
    assert result.mapping == {0: "away", 1: "home"}
    assert result.margin > 50


def test_unknown_kit_name_is_rejected():
    with pytest.raises(ValueError, match="lila-csíkos"):
        parse_kit("lila-csíkos", {"fehér": (240, 240, 240)})


def test_kit_names_are_case_and_space_insensitive():
    assert parse_kit(" Fehér ", {"fehér": (240, 240, 240)}) == pytest.approx(
        parse_kit("fehér", {"fehér": (240, 240, 240)})
    )


def test_repo_configs_parse_and_every_kit_is_in_the_palette():
    events = load_config("events", EventsConfig)
    matches = load_config("matches", MatchesConfig)
    assert len(events.split.clips) == 5 and set(events.split.clips) == set(matches.matches)
    for info in matches.matches.values():
        for name in info.kits.values():
            parse_kit(name, events.team_mapping.palette)
    combos = 1
    for values in events.grid.values():
        combos *= len(values)
    assert combos == 4 * 3 * 3 * 3 * 3 * 3
    assert events.grid["ball.airborne_mode"] == ["off", "fixed", "calibrated"]
