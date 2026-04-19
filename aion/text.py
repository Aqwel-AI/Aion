#!/usr/bin/env python3
"""
Aqwel-Aion - Text Processing
============================

Text utilities: cleaning, normalization, preprocessing, language detection,
sentiment and key phrase extraction, pattern matching, validation, and
formatting. Intended for use in NLP and research pipelines.

Author: Aksel Aghajanyan
Developed by: Aqwel AI Team
License: Apache-2.0
Copyright: 2025 Aqwel AI
"""

import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime


def count_words(text: str) -> int:
    """Return number of whitespace-separated words."""

    #Checking for errors
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return len(text.split())


def count_characters(text: str) -> int:
    """Return character count."""

    #Checking for errors
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return len(text)


def count_lines(text: str) -> int:
    """Return line count."""

    # Checking for errors
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return len(text.splitlines())


def reverse_text(text: str) -> str:
    """Return reversed string."""

    # Checking for errors
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return text[::-1]


def is_palindrome(text: str) -> bool:
    """Return True if text is a palindrome (ignoring non-alphanumeric, case-insensitive)."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only")

    cleaned_text = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    return cleaned_text == cleaned_text[::-1]


def extract_emails(text: str) -> List[str]:
    """Return list of email addresses found in text."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)


def extract_phone_numbers(text: str) -> List:
    """Return list of phone number matches (tuple groups) from text."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return re.findall(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b", text)


def extract_urls(text: str) -> List[str]:
    """Return list of URLs found in text."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return re.findall(
        r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?", text
    )


def clean_text(text: str) -> str:
    """Normalize whitespace to single spaces and strip."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return re.sub(r"\s+", " ", text.strip())


def detect_language(text: str) -> str:
    """Heuristic language detection (en/es/fr) from common words; returns 'unknown' if unclear."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only")

    text_lower = text.lower()
    english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
    spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le"]
    french_words = ["le", "la", "de", "et", "en", "un", "du", "que", "est", "pour", "dans", "sur"]
    english_count = sum(1 for word in english_words if word in text_lower)
    spanish_count = sum(1 for word in spanish_words if word in text_lower)
    french_count = sum(1 for word in french_words if word in text_lower)
    if english_count > spanish_count and english_count > french_count:
        return "en"
    if spanish_count > english_count and spanish_count > french_count:
        return "es"
    if french_count > english_count and french_count > spanish_count:
        return "fr"
    return "unknown"


def generate_hash(text: str, algorithm: str = "md5") -> str:
    """Return hex digest of text using md5, sha1, or sha256 (default md5)."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not isinstance(algorithm, str):
        raise TypeError(f"algorithm must be a str, got {type(algorithm)}")
    supported = {"md5", "sha1", "sha256"}
    if algorithm not in supported:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {', '.join(sorted(supported))}")

    text_bytes = text.encode("utf-8")
    if algorithm == "sha1":
        hash_object = hashlib.sha1(text_bytes)
    elif algorithm == "sha256":
        hash_object = hashlib.sha256(text_bytes)
    else:
        hash_object = hashlib.md5(text_bytes)
    return hash_object.hexdigest()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Return top max_keywords by frequency, excluding stop words and words of length <= 2."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not isinstance(max_keywords, int):
        raise TypeError(f"max_keywords must be an int, got {type(max_keywords)}")
    if max_keywords < 1:
        raise ValueError(f"max_keywords must be at least 1, got {max_keywords}")

    cleaned_text = clean_text(text.lower())
    stop_words = {
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
        "this", "that", "these", "those",
    }
    words = [w for w in cleaned_text.split() if w not in stop_words and len(w) > 2]
    word_freq: Dict[str, int] = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def is_question(text: str) -> bool:
    """Return True if text ends with '?' after stripping."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only")

    return text.strip().endswith("?")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    return re.sub(r"\s+", " ", text).strip()


def is_sensitive_text(text: str) -> bool:
    """Return True if text matches SSN, phone, email, password, or API key patterns."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b[\w.-]+?@\w+?\.\w+?\b",
        r"(password\s*[:=]\s*.+)",
        r"(api[_\-]?key\s*[:=]\s*.+)",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def text_contains_visual_language(text: str) -> bool:
    """Return True if text contains visual-language keywords (e.g. see, look, color, image)."""

    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if not text.strip():
        raise ValueError("Input text must not be empty or whitespace-only")

    visual_keywords = ["see", "look", "bright", "dark", "color", "image", "view", "vision", "shape"]
    return any(word in text.lower() for word in visual_keywords)
