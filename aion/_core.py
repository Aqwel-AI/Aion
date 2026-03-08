"""
Optional C++ extension bridge.

Exposes fast_sum: uses the native _aion_core extension if built,
otherwise falls back to NumPy. Install with pybind11 and a C++ compiler
to build the extension: pip install pybind11 && pip install -e .
"""

from typing import Union, Sequence

import numpy as np

try:
    from aion._aion_core import fast_sum as _fast_sum_native
    _NATIVE_AVAILABLE = True
except ImportError:
    _fast_sum_native = None
    _NATIVE_AVAILABLE = False


def fast_sum(arr: Union[Sequence[float], np.ndarray]) -> float:
    """
    Sum of a 1D array. Uses C++ implementation when the native extension
    is built for better performance on large arrays.

    Parameters
    ----------
    arr : array-like, 1D
        Values to sum (float64 or coercible).

    Returns
    -------
    float
        Sum of the array.

    Examples
    --------
    >>> from aion import fast_sum
    >>> fast_sum([1.0, 2.0, 3.0])
    6.0
    >>> fast_sum(np.arange(10))
    45.0
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("fast_sum expects a 1D array")
    if _NATIVE_AVAILABLE:
        return float(_fast_sum_native(a))
    return float(np.sum(a))


def using_native_extension() -> bool:
    """Return True if the C++ extension is loaded."""
    return _NATIVE_AVAILABLE


__all__ = ["fast_sum", "using_native_extension"]
