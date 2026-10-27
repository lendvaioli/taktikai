import numpy as np

from match_tagger.tracking.teams import assign_goalkeeper, majority_vote


def test_majority_vote_picks_most_frequent():
    assert majority_vote([0, 0, 1]) == 0
    assert majority_vote([1, 0, 1, 1]) == 1


def test_majority_vote_tie_goes_to_first_seen():
    assert majority_vote([1, 0, 0, 1]) == 1


def test_majority_vote_empty_is_none():
    assert majority_vote([]) is None


def test_goalkeeper_joins_nearest_team_centroid():
    players = np.array([[8.0, 0.0], [12.0, 0.0], [90.0, 0.0], [110.0, 0.0]])
    teams = np.array([0, 0, 1, 1])  # centroids (10, 0) and (100, 0)
    assert assign_goalkeeper(np.array([0.0, 0.0]), players, teams) == 0
    assert assign_goalkeeper(np.array([80.0, 0.0]), players, teams) == 1


def test_goalkeeper_with_only_one_team_visible():
    players = np.array([[500.0, 0.0]])
    assert assign_goalkeeper(np.array([0.0, 0.0]), players, np.array([1])) == 1


def test_goalkeeper_without_labelled_players_is_none():
    players = np.empty((0, 2))
    assert assign_goalkeeper(np.array([0.0, 0.0]), players, np.empty(0, dtype=int)) is None
