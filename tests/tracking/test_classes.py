import pytest
from pydantic import ValidationError

from match_tagger.detection.schema import DetectionRecord
from match_tagger.tracking.classes import ClassVotes, apply_track_classes
from match_tagger.tracking.teams import label_teams


def _rec(frame, track_id, cls, x_center=10.0, team=None):
    return DetectionRecord(
        frame=frame, t_sec=frame / 25, track_id=track_id, cls=cls, team=team,
        x1=x_center - 5, y1=0.0, x2=x_center + 5, y2=20.0, conf=0.9,
    )  # fmt: skip


GK_SHARE = 0.3


def _votes(*sightings):
    votes = ClassVotes()
    for track_id, cls, conf in sightings:
        votes.add(track_id, cls, conf)
    return votes.resolve(GK_SHARE)


def test_confidence_weighting_beats_raw_count():
    # track 2: two weak player sightings (0.2 + 0.2 = 0.4) lose to one strong referee (0.9)
    assert _votes((2, "player", 0.2), (2, "referee", 0.9), (2, "player", 0.2)) == {2: "referee"}


def test_tie_goes_to_first_seen_class():
    assert _votes((3, "player", 0.5), (3, "referee", 0.5)) == {3: "player"}


def test_even_player_goalkeeper_split_resolves_to_goalkeeper():
    # goalkeeper share = 1.0 / 2.0 = 0.5 >= 0.3; a plain majority would pick player (seen first)
    sightings = [(1, "player", 0.5), (1, "goalkeeper", 0.5), (1, "player", 0.5)]
    sightings.append((1, "goalkeeper", 0.5))
    assert _votes(*sightings) == {1: "goalkeeper"}


def test_player_with_ten_percent_goalkeeper_votes_stays_player():
    # goalkeeper share = 0.9 / (8.1 + 0.9) = 0.1 < 0.3
    sightings = [(4, "player", 0.9)] * 9 + [(4, "goalkeeper", 0.9)]
    assert _votes(*sightings) == {4: "player"}


def test_tracks_without_goalkeeper_votes_are_unaffected():
    sightings = [(5, "referee", 0.8), (5, "player", 0.3), (6, "player", 0.9), (6, "referee", 0.2)]
    assert _votes(*sightings) == {5: "referee", 6: "player"}


def test_goalkeeper_share_at_threshold_counts():
    # share exactly 0.3: 0.3 / (0.7 + 0.3)
    assert _votes((7, "player", 0.7), (7, "goalkeeper", 0.3)) == {7: "goalkeeper"}


def test_apply_rewrites_flickering_rows_and_leaves_ball():
    ball = DetectionRecord(frame=0, t_sec=0, cls="ball", x1=1, y1=1, x2=2, y2=2, conf=0.5)
    records = [_rec(0, 1, "goalkeeper"), _rec(5, 1, "player"), _rec(5, 2, "player"), ball]
    out = apply_track_classes(records, {1: "goalkeeper", 2: "player"})
    assert [r.cls for r in out] == ["goalkeeper", "goalkeeper", "player", "ball"]


def test_apply_revalidates_changed_rows():
    # a player row with a team cannot silently become a referee with a team
    with pytest.raises(ValidationError):
        apply_track_classes([_rec(0, 1, "player", team=0)], {1: "referee"})


def test_keeper_track_gets_one_team_after_class_fix():
    # Before the fix track 9 had a player row (own team vote) and a goalkeeper row.
    records = [
        _rec(0, 1, "player", 10.0), _rec(0, 2, "player", 100.0),
        _rec(0, 9, "goalkeeper", 0.0), _rec(5, 1, "player", 10.0),
        _rec(5, 2, "player", 100.0), _rec(5, 9, "player", 5.0),
    ]  # fmt: skip
    fixed = apply_track_classes(records, {1: "player", 2: "player", 9: "goalkeeper"})
    labelled = label_teams(fixed, {1: 0, 2: 1})
    keeper_rows = [r for r in labelled if r.track_id == 9]
    assert [(r.cls, r.team) for r in keeper_rows] == [("goalkeeper", 0), ("goalkeeper", 0)]
