"""OpenAI Chat Completions API (REST)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from .base import ChatMessage
from .http_utils import post_json


class OpenAIProvider:
    """
    Chat via OpenAI-compatible ``/v1/chat/completions``.

    Parameters
    ----------
    api_key : str, optional
        Defaults to ``OPENAI_API_KEY`` from the environment.
    model : str, optional
        Default ``gpt-4o-mini``.
    base_url : str, optional
        API root including ``/v1`` (e.g. ``https://api.openai.com/v1`` or a proxy).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAIProvider requires api_key or OPENAI_API_KEY")
        self._api_key = key
        self._model = model
        self._base = base_url.rstrip("/")

    def complete(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        url = f"{self._base}/chat/completions"
        payload: dict = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        data = post_json(url, payload, headers=headers)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected OpenAI response shape: {data!r}") from e
