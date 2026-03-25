"""
Canonical dataset loaders for common file formats.

Lightweight helpers for text, CSV, and JSONL. For transformer-specific
pipelines, use ``aion.former.datasets``.
"""

from .text import load_text, iter_text_lines
from .csv import load_csv, iter_csv_rows
from .jsonl import load_jsonl, iter_jsonl

__all__ = [
    "load_text",
    "iter_text_lines",
    "load_csv",
    "iter_csv_rows",
    "load_jsonl",
    "iter_jsonl",
]
