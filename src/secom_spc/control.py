"""Control limits and sequential charts (I-MR, EWMA, CUSUM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

D2 = 1.128  # expected range of two standard normal draws
D4 = 3.267  # moving-range chart upper limit factor for n = 2


@dataclass(frozen=True)
class IMRLimits:
    """Phase I limits for an individuals and moving-range chart."""

    center: float
    sigma: float
    mr_bar: float

    @property
    def ucl(self) -> float:
        return self.center + 3 * self.sigma

    @property
    def lcl(self) -> float:
        return self.center - 3 * self.sigma

    @property
    def mr_ucl(self) -> float:
        return D4 * self.mr_bar


def estimate_imr_limits(phase1: pd.Series | np.ndarray) -> IMRLimits:
    """Estimate center and short-term sigma from a phase I sample.

    Sigma is the average moving range divided by d2, not the sample standard
    deviation, so slow drift inside the window does not inflate the limits.
    Missing values are dropped before the moving ranges are computed.
    """
    x = np.asarray(pd.Series(phase1).dropna(), dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 non-missing points to estimate limits")
    mr_bar = float(np.mean(np.abs(np.diff(x))))
    return IMRLimits(center=float(x.mean()), sigma=mr_bar / D2, mr_bar=mr_bar)


def lag1_autocorrelation(x: pd.Series | np.ndarray) -> float:
    """Lag-1 sample autocorrelation after dropping missing values."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    v = v - v.mean()
    denom = float(np.sum(v * v))
    if denom == 0:
        return float("nan")
    return float(np.sum(v[1:] * v[:-1]) / denom)


def ewma(x: pd.Series, center: float, lam: float = 0.2) -> pd.Series:
    """Exponentially weighted moving average started at the phase I center.

    Missing observations carry the previous statistic forward.
    """
    out = np.empty(len(x))
    z = center
    for i, v in enumerate(x.to_numpy(dtype=float)):
        if not np.isnan(v):
            z = lam * v + (1 - lam) * z
        out[i] = z
    return pd.Series(out, index=x.index, name="ewma")


def ewma_limits(
    center: float, sigma: float, n: int, lam: float = 0.2, width: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Time-varying EWMA control limits for observation numbers 1..n."""
    k = np.arange(1, n + 1)
    var = sigma**2 * (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * k))
    half = width * np.sqrt(var)
    return center - half, center + half


def cusum(
    x: pd.Series, center: float, sigma: float, k: float = 0.5, h: float = 5.0
) -> pd.DataFrame:
    """Tabular two-sided CUSUM in sigma units with reference value k and decision interval h.

    Returns columns c_plus, c_minus and alarm (True where either exceeds h).
    The statistics keep accumulating after an alarm; they are not reset.
    """
    z = (x.to_numpy(dtype=float) - center) / sigma
    cp = np.zeros(len(z))
    cm = np.zeros(len(z))
    up = dn = 0.0
    for i, v in enumerate(z):
        if not np.isnan(v):
            up = max(0.0, up + v - k)
            dn = max(0.0, dn - v - k)
        cp[i], cm[i] = up, dn
    return pd.DataFrame({"c_plus": cp, "c_minus": cm, "alarm": (cp > h) | (cm > h)}, index=x.index)
