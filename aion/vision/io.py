#!/usr/bin/env python3
"""
Image I/O helpers.
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np


def read_image(path: Union[str, Path], mode: Optional[str] = "RGB") -> np.ndarray:
    """
    Read an image from disk into a NumPy array.

    Args:
        path: File path to the image.
        mode: Optional PIL mode to convert into (e.g., "RGB", "L").

    Returns:
        np.ndarray with shape (H, W, C) for color or (H, W) for grayscale.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "read_image requires Pillow. Install with: pip install aqwel-aion[docs] "
            "or pip install pillow"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        if mode:
            img = img.convert(mode)
        arr = np.array(img)
    return arr
    
    
    
     
