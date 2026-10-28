import pytest

from match_tagger.report.numbers import matches, numbers_in_text, tool_values
from match_tagger.report.schema import Report, ToolCall
from match_tagger.report.tools import normalize_arguments
from match_tagger.report.validate import ValidationInput, validate_report

from .conftest import SUMMARY_HOME, report_args

SUMMARY = {"summary": {"ball_losses": 3}, "sample": {"duration_min": 10, "n_events": 6}}


def _input(has_video=True, clips=("m-e3",), match_id="m", foreign=()):
    calls = [(ToolCall(id="c0", name="get_metric", arguments=SUMMARY_HOME), SUMMARY)]
    return ValidationInput(
        calls=calls,
        returned_clips=set(clips),
        has_video=has_video,
        match_id=match_id,
        normalize=normalize_arguments,
        foreign_names=list(foreign),
    )


def test_numbers_in_text():
    text = "A 12-es zónában 3,5 passz, 33% és -2 különbség; U19 csapat, e0012 azonosító, 2,3."
    assert numbers_in_text(text) == ["12", "3,5", "33", "-2", "2,3"]


@pytest.mark.parametrize(
    ("written", "known", "ok"),
    [
        ("3,5", [3.5], True),
        ("3.5", [3.5], True),
        ("0,67", [0.667], True),  # rounded to the written precision
        ("0,66", [0.667], False),
        ("67", [0.667], True),  # share written as percent
        ("66,7", [0.667], True),
        ("65", [0.667], False),
        ("4", [3.6], True),  # 3.6 rounds to 4
        ("4", [3.4], False),
        ("12", [12], True),
        ("13", [12], False),
    ],
)
def test_matches_with_rounding_tolerance(written, known, ok):
    assert matches(written, known, 0.5) is ok


def test_tool_values_reads_numbers_numeric_strings_and_keys():
    out = {"zones": {"12": {"count": 2, "share": 0.5}}, "zone": "7", "flag": True, "id": "m-e3"}
    assert sorted(tool_values(out)) == [0.5, 2.0, 7.0, 12.0]


def test_fabricated_number_in_free_text_is_reported():
    good = Report.model_validate(report_args())
    assert validate_report(good, _input()).errors == []
    bad = Report.model_validate(report_args(note_minutes="90"))
    assert validate_report(bad, _input()).errors == [
        "sample_note: 90 does not appear in any tool result"
    ]
    fabricated = report_args()
    fabricated["findings"][1]["training"] = "Heti 4 alkalommal."
    errors = validate_report(Report.model_validate(fabricated), _input()).errors
    assert errors == ["finding 2 training: 4 does not appear in any tool result"]


def test_clip_rules_depend_on_video():
    no_clips = report_args()
    for f in no_clips["findings"]:
        f["clip_ids"] = []
    report = Report.model_validate(no_clips)
    assert validate_report(report, _input(has_video=False, clips=())).errors == []
    assert len(validate_report(report, _input(clips=())).errors) == 2
    with_clips = Report.model_validate(report_args())
    errors = validate_report(with_clips, _input(has_video=False)).errors
    assert all("no video" in e for e in errors) and len(errors) == 2
    assert validate_report(with_clips, _input(match_id="other")).errors == [
        "match_id must be 'other'"
    ]


def test_semantic_errors_reject_the_report():
    args = report_args()
    args["findings"][0]["claim"] = (
        "A Croatia csapat 3 labdát veszített, több, mint a 10 perc alatt."
    )
    args["findings"][1]["numbers"][0]["label"] = "ball_losses"
    result = validate_report(Report.model_validate(args), _input(foreign=["Croatia"]))
    assert any('team name "Croatia"' in e for e in result.errors)
    assert any('"több" does not hold for 3 and 10' in e for e in result.errors)
    assert any('metric key "ball_losses"' in e for e in result.errors)
