"""Process capability indices.

SECOM publishes no specification limits. These functions take limits as
arguments; any limits used with this dataset are assumptions for illustration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from secom_spc.control import D2

# Percentiles that bound a normal distribution at +/- 3 sigma.
_P_LOW, _P_HIGH = 0.135, 99.865


def _clean(x: pd.Series | np.ndarray, lsl: float, usl: float) -> np.ndarray:
    if usl <= lsl:
        raise ValueError("usl must be greater than lsl")
    return np.asarray(pd.Series(x).dropna(), dtype=float)


def capability_indices(x: pd.Series | np.ndarray, lsl: float, usl: float) -> dict[str, float]:
    """Cp and Cpk (sigma from moving ranges) plus Pp and Ppk (overall sample sigma)."""
    v = _clean(x, lsl, usl)
    mean = float(v.mean())
    sigma_within = float(np.mean(np.abs(np.diff(v))) / D2)
    sigma_overall = float(v.std(ddof=1))

    def pair(sigma: float) -> tuple[float, float]:
        return (usl - lsl) / (6 * sigma), min(usl - mean, mean - lsl) / (3 * sigma)

    cp, cpk = pair(sigma_within)
    pp, ppk = pair(sigma_overall)
    return {"cp": cp, "cpk": cpk, "pp": pp, "ppk": ppk, "mean": mean}


def nonparametric_capability(x: pd.Series | np.ndarray, lsl: float, usl: float) -> dict[str, float]:
    """Percentile-method Pp and Ppk that do not assume a normal distribution.

    The 6-sigma spread becomes the 0.135th to 99.865th percentile range and the
    mean becomes the median.
    """
    v = _clean(x, lsl, usl)
    lo, med, hi = np.percentile(v, [_P_LOW, 50.0, _P_HIGH])
    if hi <= lo:
        return {"pp": float("inf"), "ppk": float("inf"), "median": float(med)}
    pp = (usl - lsl) / (hi - lo)
    upper = (usl - med) / (hi - med) if hi > med else float("inf")
    lower = (med - lsl) / (med - lo) if med > lo else float("inf")
    return {"pp": float(pp), "ppk": float(min(upper, lower)), "median": float(med)}
