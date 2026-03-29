"""Load JSON Lines (one JSON value per line)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, List, Union

PathLike = Union[str, os.PathLike[str]]


def iter_jsonl(
    path: PathLike,
    *,
    encoding: str = "utf-8",
) -> Iterator[Any]:
    """
    Stream JSONL: each non-empty line is parsed with ``json.loads``.

    Blank lines and lines that are only whitespace are skipped.

    Yields
    ------
    Any
        Parsed JSON value per line (typically ``dict``).
    """
    p = Path(path)
    with p.open(encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def load_jsonl(path: PathLike, *, encoding: str = "utf-8") -> List[Any]:
    """
    Load all JSONL records into a list.

    Returns
    -------
    list
        Parsed objects in file order.
    """
    return list(iter_jsonl(path, encoding=encoding))


def load_json_lines_file(path: PathLike, *, encoding: str = "utf-8") -> List[Any]:
    """
    Load a JSON Lines file (``.jsonl``): one JSON value per non-empty line.

    Same behavior as :func:`load_jsonl`. Prefer this name when mirroring a
    line-by-line read tutorial. Do not confuse with :func:`json.loads`, which
    parses a single JSON string, not a file path.
    """
    return load_jsonl(path, encoding=encoding)
