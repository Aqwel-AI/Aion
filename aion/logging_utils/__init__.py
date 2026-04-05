"""Opinionated ``logging`` setup without extra dependencies."""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    level: Optional[int] = None,
    *,
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure root logger once (``basicConfig``). Level from ``LOG_LEVEL``
    env (DEBUG, INFO, …) or ``level`` argument, default INFO.
    """
    if level is None:
        name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, name, logging.INFO)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
