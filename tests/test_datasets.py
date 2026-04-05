"""Tests for aion.datasets helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aion.datasets import (
    append_jsonl_line,
    append_text,
    batch_index_ranges,
    batch_processor,
    count_csv_rows,
    count_valid_records,
    csv_fieldnames,
    dump_jsonl,
    iter_batches,
    iter_csv_rows,
    iter_jsonl,
    iter_valid_records,
    load_csv,
    load_jsonl,
    load_text_lines,
    require_schema,
    validate_schema,
    validate_schema_types,
    write_text,
)


def test_batch_processor_and_index_ranges() -> None:
    data = list(range(10))
    batches = list(batch_processor(data, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    ranges = list(batch_index_ranges(10, 3))
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_iter_batches_from_generator() -> None:
    def gen():
        yield from range(5)

    assert list(iter_batches(gen(), 2)) == [[0, 1], [2, 3], [4]]


def test_iter_batches_bad_size() -> None:
    with pytest.raises(ValueError):
        list(iter_batches([1], 0))


def test_csv_fieldnames_count_max_rows(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert csv_fieldnames(p) == ["a", "b"]
    assert count_csv_rows(p) == 2
    rows = list(iter_csv_rows(p, max_rows=1))
    assert len(rows) == 1 and rows[0]["a"] == "1"
    assert len(load_csv(p, max_rows=1)) == 1


def test_jsonl_roundtrip_and_max(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    dump_jsonl(p, [{"k": 1}, {"k": 2}])
    assert len(load_jsonl(p)) == 2
    assert len(list(iter_jsonl(p, max_records=1))) == 1
    append_jsonl_line(p, {"k": 3})
    assert len(load_jsonl(p)) == 3


def test_text_write_load_append(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    write_text(f, "hello\n")
    append_text(f, "world")
    assert load_text_lines(f) == ["hello", "world"]


def test_schema_helpers() -> None:
    assert validate_schema({"text": "a", "label": "b"})
    assert not validate_schema({"text": "a"})
    with pytest.raises(ValueError, match="Missing"):
        require_schema({"text": "a"})
    require_schema({"text": "a", "label": 1})
    assert validate_schema_types({"n": 1, "x": 1.5}, {"n": int, "x": (int, float)})
    assert not validate_schema_types({"n": "bad"}, {"n": int})
    recs = [{"text": "a", "label": "x"}, {"bad": 1}, {"text": "b", "label": "y"}]
    assert list(iter_valid_records(recs)) == [recs[0], recs[2]]
    assert count_valid_records(recs) == 2
