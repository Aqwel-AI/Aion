"""Lightweight validation for dataset records (e.g. dict rows)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def validate_schema(
    data: Mapping[str, Any],
    required_keys: Optional[Sequence[str]] = None,
) -> bool:
    """
    Return True if ``data`` contains every key in ``required_keys``.

    Typical use: check that a training example dict has ``"text"`` and ``"label"``.

    Parameters
    ----------
    data : mapping
        Usually a ``dict`` (one row / record).
    required_keys : sequence of str, optional
        Keys that must be present. Default: ``("text", "label")``.

    Returns
    -------
    bool
        ``True`` if all keys exist in ``data``, else ``False``.
    """
    if not isinstance(data, Mapping):
        return False
    keys = required_keys if required_keys is not None else ("text", "label")
    return all(k in data for k in keys)
