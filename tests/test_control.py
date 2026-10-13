import numpy as np
import pandas as pd
import pytest

from secom_spc import (
    cusum,
    estimate_imr_limits,
    ewma,
    ewma_limits,
    lag1_autocorrelation,
)


def test_imr_limits_from_known_moving_range():
    x = [10, 12, 10, 12, 10, 12]  # every moving range is 2
    lim = estimate_imr_limits(x)
    assert lim.center == pytest.approx(11.0)
    assert lim.mr_bar == pytest.approx(2.0)
    assert lim.sigma == pytest.approx(2.0 / 1.128)
    assert lim.ucl == pytest.approx(11.0 + 3 * 2.0 / 1.128)
    assert lim.mr_ucl == pytest.approx(3.267 * 2.0)


def test_imr_ignores_missing_values():
    lim = estimate_imr_limits([1.0, np.nan, 2.0, 3.0, 4.0])
    assert lim.center == pytest.approx(2.5)


def test_imr_needs_three_points():
    with pytest.raises(ValueError):
        estimate_imr_limits([1.0, 2.0])


def test_lag1_autocorrelation_signs():
    alternating = [1, -1] * 20
    trend = list(range(40))
    assert lag1_autocorrelation(alternating) < -0.9
    assert lag1_autocorrelation(trend) > 0.9
    assert np.isnan(lag1_autocorrelation([5.0] * 10))


def test_ewma_constant_input_stays_at_center():
    s = pd.Series([2.0] * 5)
    assert ewma(s, center=2.0).tolist() == pytest.approx([2.0] * 5)


def test_ewma_step_response():
    z = ewma(pd.Series([1.0, 1.0]), center=0.0, lam=0.5)
    assert z.tolist() == pytest.approx([0.5, 0.75])


def test_ewma_carries_forward_over_missing():
    z = ewma(pd.Series([1.0, np.nan]), center=0.0, lam=0.5)
    assert z.tolist() == pytest.approx([0.5, 0.5])


def test_ewma_limits_widen_toward_asymptote():
    lo, hi = ewma_limits(0.0, 1.0, 50, lam=0.2)
    assert hi[0] < hi[-1]
    assert hi[-1] == pytest.approx(3 * np.sqrt(0.2 / 1.8), rel=1e-3)
    assert lo == pytest.approx(-hi)


def test_cusum_detects_sustained_shift_and_not_noise():
    quiet = pd.Series([0.2, -0.3, 0.1, -0.2, 0.0] * 4)
    assert not cusum(quiet, 0.0, 1.0)["alarm"].any()
    shifted = pd.Series([0.0] * 5 + [1.5] * 10)
    res = cusum(shifted, 0.0, 1.0)
    assert res["alarm"].iloc[-1]
    assert not res["alarm"].iloc[:5].any()
    assert (res["c_minus"] == 0).all()
