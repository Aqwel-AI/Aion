"""Load merged configuration from TOML/YAML and environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, MutableMapping, Union

PathLike = Union[str, os.PathLike[str]]


def _load_toml_bytes(data: bytes) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(data.decode("utf-8"))
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError(
            "Reading TOML on Python <3.11 requires tomli. "
            "Install with: pip install aqwel-aion[config]"
        ) from e
    return tomllib.loads(data.decode("utf-8"))


def load_toml_file(path: PathLike) -> Dict[str, Any]:
    """Load a TOML file into a nested dict."""
    p = Path(path)
    return _load_toml_bytes(p.read_bytes())


def load_yaml_file(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file (requires PyYAML: pip install aqwel-aion[config] or [former])."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML config requires PyYAML. Install with pip install pyyaml "
            "or pip install aqwel-aion[config]"
        ) from e
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if isinstance(out, dict) else {}


def merge_env_overrides(
    cfg: MutableMapping[str, Any],
    prefix: str = "AION_",
) -> MutableMapping[str, Any]:
    """
    For each env var ``PREFIX_KEY`` (nested keys as ``PREFIX_SECTION__KEY``),
    set string values on ``cfg`` (shallow merge only for top-level keys).
    """
    plen = len(prefix)
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        sub = k[plen:].lower()
        if "__" in sub:
            section, key = sub.split("__", 1)
            if section not in cfg or not isinstance(cfg[section], MutableMapping):
                cfg[section] = {}
            cfg[section][key] = v  # type: ignore[index]
        else:
            cfg[sub] = v
    return cfg


def load_config(
    path: PathLike,
    *,
    merge_env: bool = True,
    env_prefix: str = "AION_",
) -> Dict[str, Any]:
    """Load ``.toml`` or ``.yaml`` / ``.yml`` from path."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".toml",):
        cfg = dict(load_toml_file(p))
    elif suf in (".yaml", ".yml"):
        cfg = dict(load_yaml_file(p))
    else:
        raise ValueError(f"Unsupported config extension: {suf}")
    if merge_env:
        merge_env_overrides(cfg, prefix=env_prefix)
    return cfg
