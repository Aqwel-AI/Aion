"""Simple timing helpers and optional NumPy vs native ``fast_*`` comparison."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def timed_run(fn: Callable[[], T], *, repeats: int = 5, warmup: int = 1) -> Tuple[T, Dict[str, float]]:
    """
    Call ``fn`` ``warmup`` times then ``repeats`` times; return last result and
    stats: best_s, mean_s, worst_s (seconds).
    """
    for _ in range(warmup):
        fn()
    times: List[float] = []
    last: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return last, {
        "best_s": min(times),
        "mean_s": sum(times) / len(times),
        "worst_s": max(times),
        "repeats": float(repeats),
    }


def compare_sum_numpy_vs_fast(arr: np.ndarray, *, repeats: int = 5) -> Dict[str, Any]:
    """
    Compare ``numpy.sum`` vs ``aion.fast_sum`` when the native extension is
    available; otherwise only NumPy timing is meaningful.
    """
    from .. import fast_sum, using_native_extension

    a = np.asarray(arr, dtype=np.float64).ravel()
    _, st_np = timed_run(lambda: float(np.sum(a)), repeats=repeats)
    _, st_fast = timed_run(lambda: float(fast_sum(a)), repeats=repeats)
    return {
        "numpy": st_np,
        "fast_sum": st_fast,
        "using_native_extension": using_native_extension(),
    }
