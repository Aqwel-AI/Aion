"""Tests for aion.config helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import aion.config as cfg

EXAMPLES = Path(cfg.__file__).resolve().parent / "examples"
SAMPLE_TOML = EXAMPLES / "sample.toml"
SAMPLE_YAML = EXAMPLES / "sample_override.yaml"


def test_load_toml_and_get_nested() -> None:
    d = cfg.load_toml_file(SAMPLE_TOML)
    assert cfg.get_nested(d, "db.host") == "localhost"
    assert cfg.get_nested(d, "db.port") == 5432
    assert cfg.get_nested(d, "missing", 1) == 1


def test_deep_merge() -> None:
    a = {"x": 1, "db": {"host": "a", "port": 1}}
    b = {"db": {"port": 2}, "y": 3}
    m = cfg.deep_merge(a, b)
    assert m["x"] == 1 and m["y"] == 3
    assert m["db"] == {"host": "a", "port": 2}


def test_set_nested() -> None:
    d: dict = {}
    cfg.set_nested(d, "a.b.c", 7)
    assert d["a"]["b"]["c"] == 7


def test_coerce_string_scalar() -> None:
    assert cfg.coerce_string_scalar("true") is True
    assert cfg.coerce_string_scalar("42") == 42
    assert cfg.coerce_string_scalar("3.5") == 3.5
    assert cfg.coerce_string_scalar("keep me") == "keep me"


def test_load_layered() -> None:
    merged = cfg.load_layered(SAMPLE_TOML, SAMPLE_YAML, merge_env=False)
    assert merged["app"]["name"] == "aion-demo"
    assert merged["app"]["debug"] is True
    assert merged["db"]["host"] == "db.internal.example"
    assert merged["logging"]["level"] == "INFO"


def test_require_keys_ok() -> None:
    cfg.require_keys({"a": {"b": 1}}, "a.b")


def test_require_keys_fail() -> None:
    with pytest.raises(KeyError, match="a.c"):
        cfg.require_keys({"a": {"b": 1}}, "a.c")


def test_load_first_existing() -> None:
    c, p = cfg.load_first_existing(
        [EXAMPLES / "nope.toml", SAMPLE_TOML],
        merge_env=False,
    )
    assert p == SAMPLE_TOML
    assert c["app"]["name"] == "aion-demo"


def test_merge_env_overrides() -> None:
    base = cfg.load_toml_file(SAMPLE_TOML)
    os.environ["AION_DB__HOST"] = "env-host"
    try:
        cfg.merge_env_overrides(base, prefix="AION_")
        assert base["db"]["host"] == "env-host"
    finally:
        del os.environ["AION_DB__HOST"]


def test_save_yaml_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    out = tmp_path / "out.yaml"
    cfg.save_yaml_file(out, {"a": 1, "b": {"c": 2}})
    loaded = cfg.load_yaml_file(out)
    assert loaded["a"] == 1 and loaded["b"]["c"] == 2
