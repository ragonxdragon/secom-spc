"""Chronological splitting."""

from __future__ import annotations

import numpy as np


def time_split(n: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Positional indices for an ordered train block followed by a test block.

    Rows must already be sorted by time. The first floor(n * train_frac) rows
    train; the remainder test. No shuffling.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    cut = int(np.floor(n * train_frac))
    if cut == 0 or cut == n:
        raise ValueError("split would leave an empty block")
    return np.arange(cut), np.arange(cut, n)
