import numpy as np
import pytest

from secom_spc import (
    bootstrap_ci,
    capability_indices,
    nonparametric_capability,
    recall_at_fpr,
    threshold_for_fpr,
    time_split,
)


def test_capability_centered_process():
    rng = np.random.default_rng(1)
    x = rng.normal(10, 1, 5000)
    cap = capability_indices(x, lsl=7, usl=13)
    assert cap["pp"] == pytest.approx(1.0, abs=0.05)
    assert cap["ppk"] == pytest.approx(1.0, abs=0.05)


def test_capability_off_center_lowers_cpk_not_cp():
    rng = np.random.default_rng(2)
    x = rng.normal(11, 1, 5000)
    cap = capability_indices(x, lsl=7, usl=13)
    assert cap["ppk"] < cap["pp"]
    assert cap["ppk"] == pytest.approx(2 / 3, abs=0.05)


def test_capability_rejects_inverted_limits():
    with pytest.raises(ValueError):
        capability_indices([1, 2, 3, 4], lsl=5, usl=1)


def test_nonparametric_matches_normal_for_normal_data():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200000)
    np_cap = nonparametric_capability(x, -3, 3)
    assert np_cap["pp"] == pytest.approx(1.0, abs=0.03)


def test_time_split_is_ordered_and_disjoint():
    train, test = time_split(10, 0.7)
    assert train.tolist() == list(range(7))
    assert test.tolist() == [7, 8, 9]
    assert train.max() < test.min()


def test_time_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        time_split(10, 1.0)


def test_threshold_and_recall_at_fpr():
    neg = np.arange(100, dtype=float)
    thr = threshold_for_fpr(neg, 0.1)
    y = np.array([0] * 100 + [1] * 4)
    score = np.concatenate([neg, [50, 95, 99, 120]])
    recall, fpr = recall_at_fpr(y, score, thr)
    assert fpr == pytest.approx(0.1, abs=0.02)
    assert recall == pytest.approx(0.75)


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(4)
    y = (rng.random(400) < 0.2).astype(int)
    score = y + rng.normal(0, 1, 400)

    def acc(yy, ss):
        return float(((ss > 0.5) == yy.astype(bool)).mean())

    lo, hi = bootstrap_ci(y, score, acc, n_boot=300, seed=7)
    assert lo < acc(y, score) < hi
    assert (lo, hi) == bootstrap_ci(y, score, acc, n_boot=300, seed=7)
