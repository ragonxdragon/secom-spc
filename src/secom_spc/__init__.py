"""Statistical process control helpers and evaluation utilities for the SECOM dataset."""

from secom_spc.capability import capability_indices, nonparametric_capability
from secom_spc.control import (
    cusum,
    estimate_imr_limits,
    ewma,
    ewma_limits,
    lag1_autocorrelation,
)
from secom_spc.data import load_secom
from secom_spc.metrics import bootstrap_ci, recall_at_fpr, threshold_for_fpr
from secom_spc.rules import RULE_NAMES, check_rules
from secom_spc.splits import time_split

__all__ = [
    "RULE_NAMES",
    "bootstrap_ci",
    "capability_indices",
    "check_rules",
    "cusum",
    "estimate_imr_limits",
    "ewma",
    "ewma_limits",
    "lag1_autocorrelation",
    "load_secom",
    "nonparametric_capability",
    "recall_at_fpr",
    "threshold_for_fpr",
    "time_split",
]
