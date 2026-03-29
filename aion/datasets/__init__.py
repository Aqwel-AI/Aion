"""
Canonical dataset loaders for common file formats.

Lightweight helpers for text, CSV, and JSONL. For transformer-specific
pipelines, use ``aion.former.datasets``.
"""

from .batch import batch_processor
from .text import load_text, iter_text_lines
from .csv import load_csv, iter_csv_rows
from .jsonl import load_jsonl, load_json_lines_file, iter_jsonl
from .schema import validate_schema

__all__ = [
    "batch_processor",
    "load_text",
    "iter_text_lines",
    "load_csv",
    "iter_csv_rows",
    "load_jsonl",
    "load_json_lines_file",
    "iter_jsonl",
    "validate_schema",
]
