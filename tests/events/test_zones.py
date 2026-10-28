import pytest

from match_tagger.events.zones import zone, zone_bounds
from match_tagger.pitch.geometry import PitchDims

DIMS = PitchDims()  # 105 x 68: bands of 17.5 m along, 22.67 m across


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0.0, 0.0, 1),  # bottom-left corner
        (0.0, 68.0, 3),  # top-left corner
        (105.0, 0.0, 16),
        (105.0, 68.0, 18),  # top-right corner
        (52.5, 34.0, 11),  # centre spot: band 4 (52.5 / 17.5 = 3), middle row
        (17.49, 22.6, 1),
        (17.5, 22.7, 5),  # just across both boundaries
        (-3.0, 70.0, 3),  # slightly outside: nearest zone
    ],
)
def test_zone_numbers(x, y, expected):
    assert zone(x, y, DIMS) == expected


def test_no_position_no_zone():
    assert zone(None, None, DIMS) is None


def test_smaller_pitch_uses_its_own_bands():
    small = PitchDims(length_m=90, width_m=60)  # bands of 15 m and 20 m
    assert zone(15.0, 20.0, small) == 5
    assert zone(14.9, 19.9, small) == 1


def test_zone_bounds():
    assert zone_bounds(1, DIMS) == pytest.approx((0, 17.5, 0, 68 / 3))
    assert zone_bounds(18, DIMS) == pytest.approx((87.5, 105, 2 * 68 / 3, 68))
    with pytest.raises(ValueError):
        zone_bounds(19, DIMS)
