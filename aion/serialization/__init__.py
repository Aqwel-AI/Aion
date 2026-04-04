"""JSON-safe helpers for experiment configs (avoid pickle for untrusted data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def dumps_json_safe(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` with ``default=str`` for non-JSON-native values."""
    return json.dumps(obj, default=str, **kwargs)


def loads_json(s: str, **kwargs: Any) -> Any:
    """Parse JSON string."""
    return json.loads(s, **kwargs)


def write_json(path: PathLike, obj: Any, **kwargs: Any) -> None:
    """Write object as UTF-8 JSON (pretty if indent passed)."""
    p = Path(path)
    p.write_text(dumps_json_safe(obj, **kwargs), encoding="utf-8")


def read_json(path: PathLike, **kwargs: Any) -> Any:
    """Read JSON object from UTF-8 file."""
    return loads_json(Path(path).read_text(encoding="utf-8"), **kwargs)


def checkpoint_meta(
    *,
    epoch: int,
    model_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Small versioned metadata dict for training checkpoints."""
    m: Dict[str, Any] = {
        "format": "aion_checkpoint_meta_v1",
        "epoch": int(epoch),
        "model_type": str(model_type),
    }
    if extra:
        m.update(extra)
    return m
