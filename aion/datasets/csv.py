"""Load CSV files as rows of string-valued dicts (header-driven)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterator, List, Union

PathLike = Union[str, os.PathLike[str]]

# Row type: all values are strings unless a future dialect adds converters.
CsvRow = Dict[str, str]


def iter_csv_rows(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    **csv_kwargs,
) -> Iterator[CsvRow]:
    """
    Stream CSV rows as dictionaries (first row is the header).

    Parameters
    ----------
    path : str or path-like
        CSV file path.
    encoding : str, default "utf-8"
    delimiter : str, default ","
    **csv_kwargs
        Extra arguments for ``csv.DictReader`` (e.g. ``quotechar``).

    Yields
    ------
    dict[str, str]
        One row per iteration; keys are column names from the header.
    """
    p = Path(path)
    with p.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter, **csv_kwargs)
        for row in reader:
            if row is None:
                continue
            # DictReader may include None key for extra fields; drop it.
            clean = {k: v for k, v in row.items() if k is not None}
            yield clean


def load_csv(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    **csv_kwargs,
) -> List[CsvRow]:
    """
    Load an entire CSV file into a list of row dicts.

    Returns
    -------
    list[dict[str, str]]
        All data rows (header excluded from rows; names come from header).
    """
    return list(iter_csv_rows(path, encoding=encoding, delimiter=delimiter, **csv_kwargs))
