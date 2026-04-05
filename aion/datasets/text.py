"""Load plain text files (whole file or line-by-line)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Union

PathLike = Union[str, os.PathLike[str]]


def load_text(path: PathLike, *, encoding: str = "utf-8", errors: str = "strict") -> str:
    """
    Read an entire text file.

    Parameters
    ----------
    path : str or path-like
        File to read.
    encoding : str, default "utf-8"
        Text encoding.
    errors : str, default "strict"
        Passed to ``open`` (e.g. ``"replace"`` for lossy reads).

    Returns
    -------
    str
        Full file contents.
    """
    p = Path(path)
    return p.read_text(encoding=encoding, errors=errors)


def iter_text_lines(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    keepends: bool = False,
) -> Iterator[str]:
    """
    Stream text file line by line (memory-efficient for large files).

    Yields
    ------
    str
        One line per iteration (without trailing newline unless ``keepends``).
    """
    p = Path(path)
    with p.open(encoding=encoding, errors=errors) as f:
        for line in f:
            yield line if keepends else line.rstrip("\n\r")
