"""Western Electric and Nelson run rules for an individuals chart.

Every rule flags the point at which the pattern completes. A flag at index i
means the pattern ended at i, not that every point in the pattern is flagged.
Missing values (NaN) never count toward a pattern and break runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE_NAMES = {
    "WE1": "1 point beyond 3 sigma",
    "WE2": "2 of 3 consecutive points beyond 2 sigma, same side",
    "WE3": "4 of 5 consecutive points beyond 1 sigma, same side",
    "WE4": "8 consecutive points on the same side of the center",
    "N5": "6 consecutive points steadily increasing or decreasing",
    "N6": "14 consecutive points alternating up and down",
}


def _k_of_n(flags: np.ndarray, k: int, n: int) -> np.ndarray:
    """True at i when at least k of the n points ending at i are True."""
    counts = pd.Series(flags.astype(float)).rolling(n, min_periods=n).sum().to_numpy()
    return np.nan_to_num(counts, nan=0.0) >= k


def check_rules(x: pd.Series | np.ndarray, center: float, sigma: float) -> pd.DataFrame:
    """Evaluate the rules in RULE_NAMES against fixed limits.

    Returns a boolean DataFrame with one column per rule, indexed like x.
    """
    s = pd.Series(x, dtype=float)
    z = ((s - center) / sigma).to_numpy()
    valid = ~np.isnan(z)
    zz = np.where(valid, z, 0.0)

    def beyond(threshold: float, sign: int) -> np.ndarray:
        return valid & (sign * zz > threshold)

    out = pd.DataFrame(index=s.index)
    out["WE1"] = beyond(3, 1) | beyond(3, -1)
    out["WE2"] = _k_of_n(beyond(2, 1), 2, 3) & beyond(2, 1) | (
        _k_of_n(beyond(2, -1), 2, 3) & beyond(2, -1)
    )
    out["WE3"] = _k_of_n(beyond(1, 1), 4, 5) & beyond(1, 1) | (
        _k_of_n(beyond(1, -1), 4, 5) & beyond(1, -1)
    )
    out["WE4"] = _k_of_n(beyond(0, 1), 8, 8) | _k_of_n(beyond(0, -1), 8, 8)

    d = np.diff(s.to_numpy())
    up = d > 0  # comparisons with NaN are False, which breaks the run
    down = d < 0
    # Five consecutive same-direction steps span six points.
    trend = np.zeros(len(s), dtype=bool)
    trend[1:] = _k_of_n(up, 5, 5) | _k_of_n(down, 5, 5)
    out["N5"] = trend
    # Thirteen alternating steps span fourteen points.
    sign = np.where(up, 1, np.where(down, -1, 0))
    alt_pair = np.zeros(len(sign), dtype=bool)
    alt_pair[1:] = (sign[1:] * sign[:-1]) == -1
    alt = np.zeros(len(s), dtype=bool)
    alt[1:] = _k_of_n(alt_pair, 12, 12)
    out["N6"] = alt
    return out
