#!/usr/bin/env python3
"""
Aqwel-Aion - Text Embeddings and Vector Similarity
===================================================

Text-to-vector embedding and similarity: embed_text and embed_file produce
fixed-size vectors (primary: sentence-transformers all-MiniLM-L6-v2; fallback:
hash-based vectors when the library is unavailable). cosine_similarity and
related helpers support semantic search and document comparison. File input
uses automatic encoding detection.

Author: Aksel Aghajanyan
Developed by: Aqwel AI Team
License: Apache-2.0
Copyright: 2025 Aqwel AI
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # pragma: no cover


def embed_file(filepath: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Read the file at filepath and return an embedding vector for its contents.
    Uses sentence-transformers when available (model_name, default all-MiniLM-L6-v2);
    otherwise returns a 384-dim hash-based fallback. Returns None if the file
    cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if _HAS_SENTENCE_TRANSFORMERS:
            model = SentenceTransformer(model_name)
            embedding = model.encode(content)
            print(f"Embedded file: {filepath}")
            return embedding
        else:
            print(f"Embedding file (sentence-transformers not available): {filepath}")
            hash_val = int(hashlib.md5(content.encode()).hexdigest(), 16)
            return np.array([hash_val % 1000] * 384, dtype=float)
    except Exception as e:
        print(f"Error embedding file {filepath}: {e}")
        return None


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Return a dense embedding vector for the given text. Uses sentence-transformers
    (model_name, default all-MiniLM-L6-v2) when available; otherwise returns a
    384-dimensional hash-based vector. Shape is (384,) for the default model or
    (768,) for models like all-mpnet-base-v2.
    """
    if _HAS_SENTENCE_TRANSFORMERS:
        model = SentenceTransformer(model_name)
        return model.encode(text)
    else:
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return np.array([hash_val % 1000] * 384, dtype=float)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Return the cosine of the angle between vec1 and vec2, in [-1, 1].
    Computed as (vec1 · vec2) / (||vec1|| * ||vec2||). Returns 0.0 if either
    vector has zero norm. Both arrays must have the same shape.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)