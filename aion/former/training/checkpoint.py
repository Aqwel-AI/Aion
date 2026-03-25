"""
Save and load transformer weights as compressed NumPy archives.

Parameter order matches ``Transformer.parameters()`` so checkpoints are
compatible across train and generation scripts.
"""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.transformer import Transformer


def save_transformer_weights(model: Transformer, path: str) -> None:
    """
    Save all trainable parameters to a ``.npz`` file.

    Parameters
    ----------
    model : Transformer
        Model whose ``parameters()`` will be serialized.
    path : str
        Output path (typically ending in ``.npz``).
    """
    params = model.parameters()
    payload = {f"p{i}": np.asarray(p._data, dtype=np.float64) for i, p in enumerate(params)}
    np.savez_compressed(path, **payload)


def load_transformer_weights(model: Transformer, path: str) -> None:
    """
    Load weights from ``save_transformer_weights`` into an existing model.

    Raises
    ------
    ValueError
        If the file does not contain the expected number of tensors or
        shapes do not match.
    """
    params = model.parameters()
    with np.load(path) as z:
        for i, p in enumerate(params):
            key = f"p{i}"
            if key not in z.files:
                raise ValueError(f"Missing {key} in checkpoint (expected {len(params)} tensors)")
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.shape != p._data.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint {arr.shape} vs model {p._data.shape}"
                )
            p._data[...] = arr
