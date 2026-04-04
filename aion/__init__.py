#!/usr/bin/env python3
"""
Aqwel-Aion v0.1.9 - Complete AI Research & Development Library
==============================================================

A comprehensive Python utility library by Aqwel AI for AI research,
machine learning development, and advanced data science workflows.

This package provides:
- Complete mathematical and statistical operations (71+ functions)
- Algorithm utilities (search, arrays, graphs) for research and education
- Visualization (1D/2D plots, training metrics, matrices, heatmaps)
- Advanced machine learning utilities and model evaluation
- Text embeddings and AI prompt engineering tools
- Professional documentation generation system
- Code analysis and quality assessment tools
- File management and real-time monitoring
- Git integration and version control utilities

Author: Aksel Aghajanyan
Developed by: Aqwel AI Team
License: Apache-2.0
Copyright: 2025 Aqwel AI

For documentation and examples, visit:
https://aqwelai.xyz/
"""

# Define the current version of the package
__version__ = "0.1.9"

# Define the author information
__author__ = "Aksel Aghajanyan"
__developer__ = "Aqwel AI Team"

# Define the license type for the package
__license__ = "Apache-2.0"

# Define the copyright information
__copyright__ = "2025 Aqwel AI"

# Import the text processing module for text analysis and manipulation
from . import text

# Import the file management module for file operations and organization
from . import files

# Canonical dataset loaders (text / CSV / JSONL) and safe I/O primitives
from . import datasets
from . import io

# Remote LLM APIs (OpenAI, Gemini, Anthropic, OpenAI-compatible servers)
from . import providers

# Tool-calling helpers and RAG primitives
from . import tools
from . import rag

# Config, environment, logging helpers
from . import config
from . import env
from . import logging_utils

# Timing / benchmarks and safe serialization
from . import benchmarks
from . import serialization

# Optional ML metrics and pandas helpers (heavy imports deferred inside functions)
from . import metrics
from . import dataframe

# Maintainer and test utilities
from . import packaging
from . import testing

# Import the code parsing module for language detection and code analysis
from . import parser

# Import the file watching module for real-time file monitoring
from . import watcher

# Import the utilities module for general helper functions
from . import utils

# Import the command-line interface module for CLI functionality
from . import cli

# Import the Git integration module for repository management
from . import git

# Import the mathematics and statistics module for numerical computations
from . import maths

# Import additional AI/ML modules
from . import code
from . import embed
from . import evaluate
from . import prompt
from . import snippets
from . import pdf

# Import algorithms (search, arrays, graphs) and visualization (plots)
from . import algorithms
from . import visualization
from . import former

# Optional C++ extension (fast numerical ops; fallback to NumPy if not built)
from ._core import (
    fast_sum,
    fast_dot,
    fast_norm2,
    fast_mean,
    fast_variance,
    fast_argmax,
    fast_relu,
    fast_softmax,
    fast_cumsum,
    fast_matrix_vector_mul,
    fast_lower_bound,
    using_native_extension,
)

# Define the public API exports for the package
__all__ = [
    "__version__",      # Version information
    "__author__",       # Author information
    "__developer__",    # Developing team
    "__license__",      # License information
    "__copyright__",    # Copyright information
    "text",             # Text processing module
    "files",            # File management module
    "datasets",         # Text / CSV / JSONL loaders
    "io",               # Streaming, atomic writes, checksums
    "providers",        # OpenAI / Gemini / Anthropic / compatible APIs
    "tools",            # LLM tool schemas, registry, loop, retry, tokens
    "rag",              # Chunking, vector stores, simple RAG index
    "config",           # TOML/YAML config loading
    "env",              # Dotenv-style and required env vars
    "logging_utils",    # logging setup helpers
    "benchmarks",       # Timing and fast_* comparison helpers
    "serialization",    # JSON-safe experiment serialization
    "metrics",          # Optional sklearn metrics helpers
    "dataframe",        # Optional pandas helpers
    "packaging",        # Version/readme helpers for maintainers
    "testing",          # Pytest-oriented helpers
    "parser",           # Code parsing module
    "watcher",          # File watching module
    "utils",            # Utilities module
    "cli",              # Command-line interface module
    "git",              # Git integration module
    "maths",            # Mathematics and statistics module
    "code",             # Code analysis module
    "embed",            # Embedding utilities module
    "evaluate",         # Evaluation metrics module
    "prompt",           # Prompt management module
    "snippets",         # Code snippets module
    "pdf",              # PDF documentation module
    "algorithms",       # Search, arrays, graph utilities
    "visualization",    # Array/matrix/training plotting
    "former",          # Transformer training (decoder-only, NumPy autograd)
    "fast_sum",         # Fast 1D sum (C++ when built)
    "fast_dot",         # Fast dot product (C++ when built)
    "fast_norm2",       # Fast L2 norm (C++ when built)
    "fast_mean",        # Fast mean (C++ when built)
    "fast_variance",    # Fast variance (C++ when built)
    "fast_argmax",      # Fast argmax (C++ when built)
    "fast_relu",        # Fast ReLU (C++ when built)
    "fast_softmax",     # Fast softmax (C++ when built)
    "fast_cumsum",      # Fast cumulative sum (C++ when built)
    "fast_matrix_vector_mul",  # Fast matrix-vector product (C++ when built)
    "fast_lower_bound",  # Fast lower_bound on sorted array (C++ when built)
    "using_native_extension",
]