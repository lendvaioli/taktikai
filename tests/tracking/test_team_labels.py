import numpy as np

from match_tagger.detection.schema import DetectionRecord
from match_tagger.tracking.teams import (
    CropCollector,
    TeamConfig,
    crop_box,
    label_teams,
    vote_per_track,
)


def _rec(frame, track_id, cls, x_center, team=None):
    return DetectionRecord(
        frame=frame, t_sec=frame / 25, track_id=track_id, cls=cls, team=team,
        x1=x_center - 5, y1=0.0, x2=x_center + 5, y2=20.0, conf=0.9,
    )  # fmt: skip


def test_crop_box_clips_to_image():
    image = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    crop = crop_box(image, [-5.0, 2.0, 4.0, 20.0])
    assert crop.shape == (8, 4, 3)
    assert crop_box(image, [12.0, 0.0, 15.0, 5.0]) is None


def test_crop_collector_samples_every_nth_and_caps():
    collector = CropCollector(TeamConfig(crop_every=2, crops_per_track=2))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for _ in range(7):  # sightings 0..6 -> crops at 0 and 2, capped at 2
        collector.add(image, 3, [0, 0, 5, 5])
    collector.add(image, 4, [0, 0, 5, 5])
    assert collector.track_ids == [3, 3, 4]


def test_crop_collector_subset_keeps_only_given_tracks():
    collector = CropCollector(TeamConfig(crop_every=1, crops_per_track=5))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    for track_id in (1, 2, 1, 3):
        collector.add(image, track_id, [0, 0, 5, 5])
    track_ids, crops = collector.subset({1, 3})
    assert track_ids == [1, 1, 3]
    assert len(crops) == 3


def test_vote_per_track():
    assert vote_per_track([1, 1, 1, 2], [0, 1, 0, 1]) == {1: 0, 2: 1}


def test_label_teams_players_keepers_referees():
    records = [
        _rec(0, 1, "player", 10.0),
        _rec(0, 2, "player", 100.0),
        _rec(0, 3, "player", 500.0),  # no team: track had no crop
        _rec(0, 9, "goalkeeper", 0.0),  # nearest centroid is team 0 at x=10
        _rec(5, 9, "goalkeeper", 90.0),  # nearest is team 1 at x=100, but frame 5 has no players
        _rec(0, 7, "referee", 50.0),
        DetectionRecord(frame=0, t_sec=0, cls="ball", x1=1, y1=1, x2=2, y2=2, conf=0.5),
    ]
    labelled = label_teams(records, {1: 0, 2: 1})
    assert [r.team for r in labelled] == [0, 1, None, 0, 0, None, None]
