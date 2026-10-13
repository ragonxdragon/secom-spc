"""Threshold selection and bootstrap intervals for the fail classifier."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def threshold_for_fpr(scores_neg: np.ndarray, fpr: float) -> float:
    """Score threshold that alarms on about `fpr` of the negative (pass) runs.

    Alarm means score >= threshold. Uses the (1 - fpr) quantile of negatives.
    """
    return float(np.quantile(np.asarray(scores_neg, dtype=float), 1 - fpr))


def recall_at_fpr(y_true: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    """Return (recall, realized false-alarm rate) for a fixed score threshold."""
    y = np.asarray(y_true).astype(bool)
    alarm = np.asarray(score) >= threshold
    pos, neg = y.sum(), (~y).sum()
    recall = float((alarm & y).sum() / pos) if pos else float("nan")
    fpr = float((alarm & ~y).sum() / neg) if neg else float("nan")
    return recall, fpr


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling runs with replacement.

    Resamples that contain no positive runs are skipped. Resampling treats runs
    as exchangeable, which ignores serial dependence; intervals are optimistic
    if runs are autocorrelated.
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(stat(y[idx], s[idx]))
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
