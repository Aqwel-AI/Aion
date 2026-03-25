"""
Safe file I/O: streaming reads, atomic writes, checksums.

Prefer this module for low-level patterns; use ``aion.files`` for higher-level
CRUD helpers that print on error.
"""

from .streaming import iter_lines, read_chunks
from .atomic import atomic_write, atomic_write_bytes
from .checksum import file_sha256, verify_sha256

__all__ = [
    "iter_lines",
    "read_chunks",
    "atomic_write",
    "atomic_write_bytes",
    "file_sha256",
    "verify_sha256",
]
