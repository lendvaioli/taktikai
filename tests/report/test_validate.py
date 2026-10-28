import pytest

from match_tagger.report.schema import Report
from match_tagger.report.validate import (
    matches,
    numbers_in_text,
    tool_values,
    validate_report,
)

from .conftest import report_args


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


def test_fabricated_number_in_any_field_is_reported():
    outputs = [{"total": 3, "minutes": 10, "events": 6}]
    good = Report.model_validate(report_args())
    assert validate_report(good, outputs, {"m-e3"}, True, "m", 0.5) == []
    bad = Report.model_validate(report_args(note_minutes="90"))
    errors = validate_report(bad, outputs, {"m-e3"}, True, "m", 0.5)
    assert errors == ["sample_note: 90 does not appear in any tool result"]
    fabricated = report_args()
    fabricated["findings"][1]["training"] = "Heti 4 alkalommal."
    errors = validate_report(Report.model_validate(fabricated), outputs, {"m-e3"}, True, "m", 0.5)
    assert errors == ["finding 2 training: 4 does not appear in any tool result"]


def test_clip_rules_depend_on_video():
    outputs = [{"total": 3, "minutes": 10, "events": 6}]
    no_clips = report_args()
    for f in no_clips["findings"]:
        f["clip_ids"] = []
    report = Report.model_validate(no_clips)
    assert validate_report(report, outputs, set(), False, "m", 0.5) == []
    assert len(validate_report(report, outputs, set(), True, "m", 0.5)) == 2
    with_clips = Report.model_validate(report_args())
    errors = validate_report(with_clips, outputs, {"m-e3"}, False, "m", 0.5)
    assert all("no video" in e for e in errors) and len(errors) == 2
    assert validate_report(with_clips, outputs, {"m-e3"}, True, "other", 0.5) == [
        "match_id must be 'other'"
    ]
