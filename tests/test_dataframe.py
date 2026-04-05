"""Tests for aion.dataframe (requires pandas)."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

import aion.dataframe as adf

EXAMPLES = Path(adf.__file__).resolve().parent / "examples"
SAMPLE = EXAMPLES / "sample.csv"


@pytest.fixture
def df():
    return adf.read_csv_with_columns(
        SAMPLE,
        ["id", "feature_a", "feature_b", "label"],
    )


def test_read_csv_with_columns_ok(df) -> None:
    assert len(df) == 8
    assert list(df.columns) == ["id", "feature_a", "feature_b", "label"]


def test_read_csv_with_columns_missing() -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        adf.read_csv_with_columns(SAMPLE, ["id", "nope"])


def test_assert_no_nulls(df) -> None:
    adf.assert_no_nulls(df)
    bad = df.copy()
    bad.loc[0, "label"] = None
    with pytest.raises(ValueError, match="Null"):
        adf.assert_no_nulls(bad)


def test_train_val_split_rows(df) -> None:
    tr, va = adf.train_val_split_rows(df, val_fraction=0.25, seed=1)
    assert len(tr) + len(va) == len(df)
    assert len(va) >= 1


def test_stratified_train_val_split(df) -> None:
    tr, va = adf.stratified_train_val_split(df, "label", val_fraction=0.25, seed=2)
    assert len(tr) + len(va) == len(df)
    assert set(tr["label"]) <= set(df["label"])


def test_drop_constant_columns() -> None:
    wide = pd.DataFrame({"a": [1, 2], "b": [3, 3], "c": ["x", "y"]})
    slim = adf.drop_constant_columns(wide)
    assert "b" not in slim.columns


def test_basic_summary(df) -> None:
    s = adf.basic_summary(df)
    assert s["n_rows"] == 8
    assert "memory_mib" in s
