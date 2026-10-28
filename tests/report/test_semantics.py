import pytest

from match_tagger.config import load_config
from match_tagger.report.config import ReportConfig, SemanticsConfig
from match_tagger.report.schema import NumberSource, SupportingNumber, ToolCall
from match_tagger.report.semantics import (
    check_comparisons,
    english_errors,
    paths_with_value,
    resolve,
    sentences,
    source_error,
)
from match_tagger.report.tools import normalize_arguments

CFG = SemanticsConfig()
WORDS = ["the", "pass", "ball", "loss", "losses", "final", "very", "low"]


def test_sentences_keep_decimals():
    assert sentences("Arány 70,3% volt. Másik 81.2 érték; harmadik") == [
        "Arány 70,3% volt",
        "Másik 81.2 érték",
        "harmadik",
    ]


@pytest.mark.parametrize(
    ("text", "errors", "unchecked"),
    [
        # the v1 StatsBomb sentence: 51 is not more than 54
        ("A francia csapat 51 labdavesztést regisztrált, ami több, mint a horvát csapat 54.", 1, 0),
        ("A francia csapat 51 labdavesztést szenvedett, ami kevesebb, mint a horvát 54.", 0, 0),
        ("A passzpontosság 70,3%, ami alacsonyabb, mint a vendégcsapat 81,2%-a.", 0, 0),
        ("A passzpontosság 70,3%, ami magasabb, mint a vendégcsapat 81,2%-a.", 1, 0),
        ("51 labdavesztés nem több, mint 54.", 0, 0),  # negation flips
        ("A horvátok 54-éhez képest a franciák 51 labdavesztése kevesebb.", 0, 0),  # swap
        ("A horvátok 54-éhez képest a franciák 51 labdavesztése több.", 1, 0),
        ("3 és 4 közül több a jobb sávban, mint 2.", 0, 1),  # three numbers: unchecked
        ("Több labdavesztés a jobb sávban.", 0, 1),  # no numbers: unchecked
        ("A többi játékos 3 passzt adott 4 helyett.", 0, 0),  # "többi" is not a comparison
        ("4 egyenlő, és több, mint 4.", 1, 0),  # equal numbers with a comparative
        # the full v1 sentence: the ", bár" clause is checked on its own
        (
            "A francia csapat 51 labdavesztést regisztrált, ami több, mint a horvát csapat 54 "
            "labdavesztése, bár a teljesítési arányuk alacsonyabb.",
            1,
            1,
        ),
    ],
)
def test_comparison_direction(text, errors, unchecked):
    found, skipped = check_comparisons(text, CFG)
    assert (len(found), skipped) == (errors, unchecked)


def test_english_words_keys_and_team_names():
    text = (
        "A CROATIA csapat very_low mintán, ball loss a final harmadban; Franciaország jól játszott."
    )
    found = english_errors(text, WORDS, ["Croatia", "France"])
    assert 'metric key "very_low"' in found
    assert {'English word "ball"', 'English word "loss"', 'English word "final"'} <= set(found)
    assert 'team name "Croatia" (use the Hungarian label)' in found
    assert not any("France" in f for f in found)  # "Franciaország" is not "France"


def test_hungarian_words_are_not_english_with_the_configured_list():
    words = load_config("report", ReportConfig).semantics.english_words
    assert {"the", "ball", "final"} <= set(words)
    text = "Ez is a bal sávban történt, ok nélkül; a ballábas játékos per percben passzolt."
    assert english_errors(text, words, []) == []


def test_resolve_paths():
    data = {"thirds": {"middle": {"count": 4}}, "events": [{"t_sec": 30.0}, {"t_sec": 40.0}]}
    assert resolve(data, "thirds.middle.count") == 4
    assert resolve(data, "events.1.t_sec") == 40.0
    with pytest.raises(KeyError):
        resolve(data, "thirds.final.count")
    with pytest.raises(KeyError):
        resolve(data, "events.5.t_sec")


# A v2 failure: real tool numbers with the wrong meaning. The model wrote
# "3 losses in the middle third, 3 in the final third"; per third the counts are 4 and 2.
LOSSES = {
    "zones": {
        "middle_centre": {"count": 3},
        "middle_left": {"count": 1},
        "final_centre": {"count": 1},
    },
    "thirds": {"middle": {"count": 4, "share": 0.667}, "final": {"count": 2, "share": 0.333}},
}
ARGS = {"name": "ball_losses_by_zone", "team": "home"}
CALLS = [(ToolCall(id="c1", name="get_metric", arguments=ARGS), LOSSES)]


def _num(label, value, path, args=ARGS, tool="get_metric"):
    return SupportingNumber(
        label=label, value=value, source=NumberSource(tool=tool, arguments=args, path=path)
    )


def test_source_rejects_the_v2_wrong_meaning():
    wrong = _num("labdavesztések a középső harmadban", 3, "thirds.middle.count")
    assert source_error(wrong, CALLS, normalize_arguments) == (
        '"labdavesztések a középső harmadban": value 3 but '
        "get_metric(name=ball_losses_by_zone, team=home) -> thirds.middle.count is 4"
        "; 3 is found at: get_metric(name=ball_losses_by_zone, team=home) -> "
        "zones.middle_centre.count"
    )
    right = _num("labdavesztések a középső harmadban", 4, "thirds.middle.count")
    assert source_error(right, CALLS, normalize_arguments) is None
    zone = _num("labdavesztések a középső harmad középső sávjában", 3, "zones.middle_centre.count")
    assert source_error(zone, CALLS, normalize_arguments) is None


def test_source_needs_a_call_of_this_run_and_an_existing_field():
    other_team = _num(
        "x", 4, "thirds.middle.count", args={"name": "ball_losses_by_zone", "team": "away"}
    )
    assert "was not made in this run" in source_error(other_team, CALLS, normalize_arguments)
    missing = _num("x", 4, "thirds.own.count")
    assert 'has no field "thirds.own.count"' in source_error(missing, CALLS, normalize_arguments)
    reordered = _num(
        "x", 4, "thirds.middle.count", args={"team": "home", "name": "ball_losses_by_zone"}
    )
    assert source_error(reordered, CALLS, normalize_arguments) is None
    share_as_percent = _num("x", 66.7, "thirds.middle.share")
    assert "is 0.667" in source_error(share_as_percent, CALLS, normalize_arguments)


def test_hint_names_the_path_from_the_v3_sb_failure():
    # v3 sb_8658: source path "pass_completion_share" instead of "summary.pass_completion_share"
    summary = {"summary": {"passes": 249, "pass_completion_share": 0.703}, "has_video": False}
    calls = [
        (
            ToolCall(id="c0", name="get_metric", arguments={"name": "summary", "team": "home"}),
            summary,
        )
    ]
    cited = _num(
        "passzpontosság", 0.703, "pass_completion_share", args={"name": "summary", "team": "home"}
    )
    error = source_error(cited, calls, normalize_arguments)
    assert error.endswith(
        "0.703 is found at: get_metric(name=summary, team=home) -> summary.pass_completion_share"
    )
    nowhere = _num(
        "passzpontosság", 0.5, "pass_completion_share", args={"name": "summary", "team": "home"}
    )
    assert source_error(nowhere, calls, normalize_arguments).endswith(
        "0.5 occurs in no tool result of this run"
    )


def test_paths_with_value_walks_lists_and_ignores_bools():
    data = {"a": [1, {"b": 1.0}], "c": True, "d": "1"}
    assert paths_with_value(data, 1) == ["a.0", "a.1.b"]
