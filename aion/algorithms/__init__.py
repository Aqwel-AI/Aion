"""
Algorithms Package
==================

Public algorithmic utilities for research and educational workflows.

This package exposes a stable, minimal API for common algorithms used in data
processing, optimization baselines, and graph-based reasoning. Implementations
prioritize correctness, clarity, and predictable behavior.

Submodules (source files and their functions):
- search.py: binary_search, lower_bound, upper_bound, is_sorted, jump_search,
  find_peak_element, exponential_search, linear_search, First_Occurrence,
  Last_Occurrence, First_Last_Occurrence, roatated_search, ternary_search,
  interpolation_search
- arrays.py: flatten_array, chunk_array, remove_duplicates, moving_avarage,
  flatten_deep, sliding_window, pad_array, rolling_sum, pairwise
- graphs.py: bfs, dfs, toposort, dijkstra, connected_components, shortest_path_unweighted

Full package documentation: see README.md in this directory.
"""

# search.py
from .search import (
    binary_search,
    lower_bound,
    upper_bound,
    is_sorted,
    jump_search,
    find_peak_element,
    exponential_search,
    linear_search,
    First_Occurrence,
    Last_Occurrence,
    First_Last_Occurrence,
    roatated_search,
    ternary_search,
    interpolation_search,
)

# arrays.py
from .arrays import (
    flatten_array,
    chunk_array,
    remove_duplicates,
    moving_avarage,
    flatten_deep,
    sliding_window,
    pad_array,
    rolling_sum,
    pairwise,
)

# graphs.py
from .graphs import (
    bfs,
    connected_components,
    dfs,
    dijkstra,
    shortest_path_unweighted,
    toposort,
)

__all__ = [
    # search.py
    "binary_search",
    "lower_bound",
    "upper_bound",
    "is_sorted",
    "jump_search",
    "find_peak_element",
    "exponential_search",
    "linear_search",
    "First_Occurrence",
    "Last_Occurrence",
    "First_Last_Occurrence",
    "roatated_search",
    "ternary_search",
    "interpolation_search",
    # arrays.py
    "flatten_array",
    "chunk_array",
    "remove_duplicates",
    "moving_avarage",
    "flatten_deep",
    "sliding_window",
    "pad_array",
    "rolling_sum",
    "pairwise",
    # graphs.py
    "bfs",
    "dfs",
    "toposort",
    "dijkstra",
    "connected_components",
    "shortest_path_unweighted",
]