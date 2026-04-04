"""Lightweight helpers for package metadata (maintainers)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def read_version_from_init(init_path: Optional[Path] = None) -> str:
    """Parse ``__version__ = \"x.y.z\"`` from ``aion/__init__.py``."""
    root = Path(__file__).resolve().parents[1]
    p = init_path or (root / "__init__.py")
    text = p.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise ValueError("Could not find __version__ in aion/__init__.py")
    return m.group(1)
