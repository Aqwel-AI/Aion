"""
Algorithms Package
==================

Public algorithmic utilities for research and educational workflows.

This package exposes a high-performance, professional-grade API for common 
algorithms used in data processing, optimization baselines, and graph-based reasoning.
Implementations prioritize correctness, efficiency, and project-wide consistency.

Submodules:
- search.py: Comprehensive search (binary, ternary, exponential), optimization (BS on answer), 
  string matching (KMP, Aho-Corasick), and selection (Quickselect).
- arrays.py: Sequence processing, matrix operations, statistical utilities, and padding.
- graphs.py: Advanced graph processing (BFS/DFS, shortest paths, SCCs, network flow, centrality).

Full package documentation: see README.md in this directory.
"""

# search.py
from .search import (
    binary_search,
    lower_bound,
    upper_bound,
    is_sorted,
    jump_search,
    find_all_peaks,
    exponential_search,
    linear_search,
    first_occurrence,
    last_occurrence,
    first_last_occurrence,
    rotated_search,
    ternary_search,
    interpolation_search,
    integer_sqrt,
    nth_root,
    kmp_search,
    aho_corasick_simple,
    quickselect,
    find_median_unordered,
)

# arrays.py
from .arrays import (
    flatten_array,
    chunk_array,
    remove_duplicates,
    moving_average,
    flatten_deep,
    sliding_window,
    pad_array,
    rolling_sum,
    pairwise,
    matrix_transpose,
    matrix_multiply,
    z_score_normalization,
    min_max_scaling,
)

# graphs.py
from .graphs import (
    bfs,
    dfs,
    toposort,
    dijkstra,
    a_star,
    bellman_ford,
    floyd_warshall,
    tarjan_scc,
    kosaraju_scc,
    prim_mst,
    kruskal_mst,
    ford_fulkerson,
    pagerank,
    connected_components,
)

__all__ = [
    # search.py
    "binary_search",
    "lower_bound",
    "upper_bound",
    "is_sorted",
    "jump_search",
    "find_all_peaks",
    "exponential_search",
    "linear_search",
    "first_occurrence",
    "last_occurrence",
    "first_last_occurrence",
    "rotated_search",
    "ternary_search",
    "interpolation_search",
    "integer_sqrt",
    "nth_root",
    "kmp_search",
    "aho_corasick_simple",
    "quickselect",
    "find_median_unordered",
    # arrays.py
    "flatten_array",
    "chunk_array",
    "remove_duplicates",
    "moving_average",
    "flatten_deep",
    "sliding_window",
    "pad_array",
    "rolling_sum",
    "pairwise",
    "matrix_transpose",
    "matrix_multiply",
    "z_score_normalization",
    "min_max_scaling",
    # graphs.py
    "bfs",
    "dfs",
    "toposort",
    "dijkstra",
    "a_star",
    "bellman_ford",
    "floyd_warshall",
    "tarjan_scc",
    "kosaraju_scc",
    "prim_mst",
    "kruskal_mst",
    "ford_fulkerson",
    "pagerank",
    "connected_components",
]