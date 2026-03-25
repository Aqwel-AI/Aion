"""OpenAI-compatible HTTP servers (LM Studio, vLLM, Ollama OpenAI bridge, etc.)."""

from __future__ import annotations

from typing import Any, List, Optional

from .base import ChatMessage
from .http_utils import post_json


class OpenAICompatibleProvider:
    """
    Talk to any API that implements OpenAI-style ``POST .../chat/completions``.

    Parameters
    ----------
    base_url : str
        Base URL including ``/v1`` if the server uses that layout, e.g.
        ``http://localhost:1234/v1``.
    model : str
        Model name accepted by that server.
    api_key : str, optional
        Sent as ``Bearer`` if provided (many local servers ignore it).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
    ):
        self._base = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

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
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = post_json(url, payload, headers=headers or None)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected chat/completions response: {data!r}") from e
