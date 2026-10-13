"""Rule checks against hand-constructed series (center 0, sigma 1)."""

import numpy as np

from secom_spc import check_rules


def flags(values, rule):
    return check_rules(np.array(values, dtype=float), 0.0, 1.0)[rule].to_numpy()


def test_in_control_series_has_no_flags():
    x = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.1, 0.2, -0.2]
    assert not check_rules(np.array(x), 0.0, 1.0).to_numpy().any()


def test_we1_single_point_beyond_three_sigma():
    f = flags([0, 0, 3.5, 0, -3.2, 0], "WE1")
    assert f.tolist() == [False, False, True, False, True, False]


def test_we1_exactly_three_sigma_is_not_beyond():
    assert not flags([3.0, -3.0], "WE1").any()


def test_we2_two_of_three_beyond_two_sigma_same_side():
    f = flags([0, 2.5, 0.5, 2.4, 0], "WE2")
    assert f.tolist() == [False, False, False, True, False]


def test_we2_opposite_sides_do_not_count():
    assert not flags([2.5, 0, -2.5], "WE2").any()


def test_we3_four_of_five_beyond_one_sigma():
    f = flags([1.5, 1.2, 0.2, 1.1, 1.3, 0], "WE3")
    assert f.tolist() == [False, False, False, False, True, False]


def test_we4_eight_in_a_row_same_side():
    f = flags([0.5] * 8 + [-0.5], "WE4")
    assert f.tolist() == [False] * 7 + [True, False]


def test_we4_seven_is_not_enough():
    assert not flags([0.5] * 7, "WE4").any()


def test_we4_nan_breaks_the_run():
    assert not flags([0.5] * 4 + [np.nan] + [0.5] * 4, "WE4").any()


def test_n5_six_points_trending_up():
    f = flags([0, 1, 2, 3, 4, 5, 4.5], "N5")
    assert f.tolist() == [False] * 5 + [True, False]


def test_n5_five_points_is_not_a_trend():
    assert not flags([0, 1, 2, 3, 4], "N5").any()


def test_n6_fourteen_points_alternating():
    x = [0.1 if i % 2 == 0 else -0.1 for i in range(14)]
    f = flags(x, "N6")
    assert f.tolist() == [False] * 13 + [True]


def test_n6_thirteen_points_is_not_enough():
    x = [0.1 if i % 2 == 0 else -0.1 for i in range(13)]
    assert not flags(x, "N6").any()
