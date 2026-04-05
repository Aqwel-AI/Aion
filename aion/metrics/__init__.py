"""Optional sklearn-based metrics (install aqwel-aion[ai] or [metrics])."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def brier_score_binary(y_true: List[int], y_prob: List[float]) -> float:
    """
    Mean squared error between labels in {0,1} and predicted probabilities.
    Requires scikit-learn when available; falls back to NumPy implementation.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_prob must have the same length")
    return float(np.mean((yp - yt) ** 2))


def reliability_bins(
    y_true: List[int],
    y_prob: List[float],
    *,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Histogram-based calibration table: bin midpoints, accuracy per bin, count.
    For visualization only; uses NumPy only.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (yp >= edges[i]) & (yp < edges[i + 1]) if i < n_bins - 1 else (yp >= edges[i]) & (yp <= edges[i + 1])
        cnt[i] = int(np.sum(m))
        if cnt[i]:
            acc[i] = float(np.mean(yt[m]))
    return mids, acc, cnt


def sklearn_classification_report_strings(y_true, y_pred) -> str:
    """Wrapper for ``sklearn.metrics.classification_report``; requires sklearn."""
    try:
        from sklearn.metrics import classification_report
    except ImportError as e:
        raise ImportError(
            "sklearn_classification_report_strings requires scikit-learn. "
            "Install with pip install aqwel-aion[ai] or aqwel-aion[metrics]"
        ) from e
    return classification_report(y_true, y_pred)
