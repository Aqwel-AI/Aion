"""Placeholder for graph traversal and shortest-path algorithms (BFS, DFS, Dijkstra, toposort)."""
"""
Graph Algorithms
================

This module provides graph traversal and ordering algorithms for directed and
undirected graphs represented as adjacency lists. Implementations use only the
standard library and follow the same conventions as the rest of aion.algorithms:
correctness, clarity, and predictable behavior.

Scope
-----
- Breadth-first search (BFS): traverse or explore from a start node in level order.
- Depth-first search (DFS): traverse from a start node using a stack (iterative).
- Topological sort: linear ordering of vertices in a directed acyclic graph (DAG).

Graph representation
--------------------
Graphs are represented as adjacency lists: a mapping from node to list of neighbors.
- For BFS/DFS: graph[node] = list of adjacent nodes (any hashable type).
- For toposort: graph is directed; graph[node] = list of successors.
- Nodes must be hashable (e.g. int, str).

See also: aion/algorithms/README.md for full package documentation.
"""

from typing import Dict, List, Any, Set
from collections import deque


def bfs(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    """
    Breadth-first search from a start node.

    Explores the graph level by level: start, then all nodes at distance 1,
    then distance 2, and so on. Uses a queue. Nodes are returned in BFS order.
    Nodes that are not reachable from start are not included.

    Parameters
    ----------
    graph : dict
        Adjacency list: graph[node] = list of neighbors. Undirected edges can be
        represented by listing each endpoint in the other's list.
    start : hashable
        The starting node (must be a key in graph or have no outgoing edges).

    Returns
    -------
    list
        List of nodes in BFS order (order of first visit).

    Raises
    ------
    KeyError
        If start is not in graph.

    Notes
    -----
    - Time complexity O(V + E); space O(V).
    - If start is not in graph, add it with an empty list: graph.setdefault(start, []).
    """
    if start not in graph:
        raise KeyError(f"Start node {start!r} not in graph")
    visited: Set[Any] = set()
    order: List[Any] = []
    queue: deque = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    """
    Depth-first search from a start node (iterative, stack-based).

    Explores as far as possible along each branch before backtracking.
    Returns nodes in the order they are first discovered (pre-order).
    Nodes not reachable from start are not included.

    Parameters
    ----------
    graph : dict
        Adjacency list: graph[node] = list of neighbors.
    start : hashable
        The starting node.

    Returns
    -------
    list
        List of nodes in DFS order (pre-order).

    Raises
    ------
    KeyError
        If start is not in graph.

    Notes
    -----
    - Time complexity O(V + E); space O(V).
    - Iterative implementation avoids recursion depth limits on large graphs.
    """
    if start not in graph:
        raise KeyError(f"Start node {start!r} not in graph")
    visited: Set[Any] = set()
    order: List[Any] = []
    stack: List[Any] = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def toposort(graph: Dict[Any, List[Any]]) -> List[Any]:
    """
    Topological sort of a directed acyclic graph (DAG).

    Returns a linear ordering of vertices such that for every directed edge
    (u, v), u comes before v. Uses Kahn's algorithm (in-degree based).
    If the graph has a cycle, ValueError is raised.

    Parameters
    ----------
    graph : dict
        Directed adjacency list: graph[node] = list of successors. All nodes
        that appear as keys or in any value list are considered vertices.

    Returns
    -------
    list
        One possible topological order of the vertices.

    Raises
    ------
    ValueError
        If the graph contains a cycle (not a DAG).

    Notes
    -----
    - Time complexity O(V + E).
    - Nodes with no outgoing edges may be omitted from graph; they are still
      included in the ordering if they appear as successors of others.
    """
    # Collect all vertices and compute in-degrees
    vertices: Set[Any] = set()
    in_degree: Dict[Any, int] = {}
    for node, neighbors in graph.items():
        vertices.add(node)
        in_degree.setdefault(node, 0)
        for v in neighbors:
            vertices.add(v)
            in_degree[v] = in_degree.get(v, 0) + 1
    # Queue of nodes with in-degree 0
    queue: deque = deque(u for u in vertices if in_degree.get(u, 0) == 0)
    order: List[Any] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(order) != len(vertices):
        raise ValueError("Graph contains a cycle; topological sort is undefined")
    return order
