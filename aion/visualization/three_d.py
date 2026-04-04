"""3D plots for ML / teaching (matplotlib; optional [viz] extra)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def plot_3d_scatter(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    *,
    title: str = "3D scatter",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    show: bool = True,
):
    """Scatter in 3D; returns matplotlib Figure."""
    import matplotlib.pyplot as plt  # pyright: ignore [reportMissingImports]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(np.asarray(x), np.asarray(y), np.asarray(z))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()
    return fig


def plot_3d_surface(
    x: Sequence[float],
    y: Sequence[float],
    z: np.ndarray,
    *,
    title: str = "Surface",
    show: bool = True,
):
    """
    Plot a surface: 1D ``x`` (length m), 1D ``y`` (length n), 2D ``z`` with
    shape ``(n, m)`` matching ``numpy.meshgrid(x, y)``.
    """
    import matplotlib.pyplot as plt  # pyright: ignore [reportMissingImports]

    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(z, dtype=float)
    X, Y = np.meshgrid(xv, yv)
    if Z.shape != X.shape:
        raise ValueError(f"z shape {Z.shape} must match meshgrid {X.shape} (len(y) x len(x))")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
    ax.set_title(title)
    if show:
        plt.show()
    return fig
