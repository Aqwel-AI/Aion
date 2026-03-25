"""Construct a provider by name (for config-driven apps)."""

from __future__ import annotations

from typing import Any, List

from .anthropic_provider import AnthropicProvider
from .base import ChatProvider
from .gemini_provider import GeminiProvider
from .generic_openai import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider


def supported_providers() -> List[str]:
    """Short names accepted by :func:`create_provider`."""
    return [
        "openai",
        "gemini",
        "google",
        "anthropic",
        "claude",
        "openai_compatible",
        "compatible",
    ]


def create_provider(name: str, **kwargs: Any) -> ChatProvider:
    """
    Build a chat provider from a string name.

    Parameters
    ----------
    name : str
        One of: ``openai``, ``gemini`` / ``google``, ``anthropic`` / ``claude``,
        ``openai_compatible`` / ``compatible`` (requires ``base_url`` + ``model``).
    **kwargs
        Passed to the provider constructor (``api_key``, ``model``, ``base_url``, …).

    Returns
    -------
    ChatProvider
    """
    key = name.lower().strip()
    if key == "openai":
        return OpenAIProvider(**kwargs)
    if key in ("gemini", "google"):
        return GeminiProvider(**kwargs)
    if key in ("anthropic", "claude"):
        return AnthropicProvider(**kwargs)
    if key in ("openai_compatible", "compatible"):
        return OpenAICompatibleProvider(**kwargs)
    raise ValueError(
        f"Unknown provider {name!r}. Try one of: {supported_providers()}"
    )
