"""Optional pandas helpers (install aqwel-aion[ai])."""

from __future__ import annotations

from typing import Optional, Sequence


def require_pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "aion.dataframe requires pandas. Install with pip install aqwel-aion[ai]"
        ) from e
    return pd


def assert_columns(df, columns: Sequence[str]) -> None:
    """Raise ``ValueError`` if any column is missing."""
    pd = require_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def train_val_split_rows(
    df,
    *,
    val_fraction: float = 0.2,
    seed: Optional[int] = None,
):
    """
    Shuffle and split a DataFrame into train and validation by row fraction.
    """
    pd = require_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled.iloc[n_val:], shuffled.iloc[:n_val]
