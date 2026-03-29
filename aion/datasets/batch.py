"""Yield fixed-size slices from a sequence (batching for large data)."""

from __future__ import annotations

from typing import Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def batch_processor(data: Sequence[T], batch_size: int) -> Iterator[List[T]]:
    """
    Split a sequence into consecutive chunks of length ``batch_size`` (last chunk may be shorter).

    Uses a generator so large collections are not copied whole into memory.

    Parameters
    ----------
    data : sequence
        Any indexable sequence (list, tuple, etc.).
    batch_size : int
        Maximum number of items per yielded chunk. Must be positive.

    Yields
    ------
    list
        The next batch ``data[i : i + batch_size]``.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = len(data)
    for i in range(0, n, batch_size):
        yield list(data[i : i + batch_size])
