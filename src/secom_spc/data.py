"""Loading the raw SECOM files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_secom(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read secom.data and secom_labels.data and return runs sorted by timestamp.

    Returns the sensor frame (columns s000..s589, NaN for missing) and a label
    series where 1 means fail and 0 means pass. Both share a DatetimeIndex.
    """
    data_dir = Path(data_dir)
    x = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    x.columns = [f"s{i:03d}" for i in range(x.shape[1])]
    raw = pd.read_csv(
        data_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "stamp"],
        quotechar='"',
    )
    if len(x) != len(raw):
        raise ValueError("secom.data and secom_labels.data have different row counts")
    stamps = pd.to_datetime(raw["stamp"], format="%d/%m/%Y %H:%M:%S")
    x.index = pd.DatetimeIndex(stamps, name="timestamp")
    y = pd.Series((raw["label"].to_numpy() == 1).astype(int), index=x.index, name="fail")
    order = x.index.argsort(kind="stable")
    return x.iloc[order], y.iloc[order]
