"""Anthropic Messages API (REST)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .base import ChatMessage
from .structured import AssistantTurn
from .http_utils import post_json


class AnthropicProvider:
    """
    Chat via Anthropic ``/v1/messages``.

    Parameters
    ----------
    api_key : str, optional
        Defaults to ``ANTHROPIC_API_KEY``.
    model : str, optional
        Default ``claude-3-5-sonnet-20241022`` (adjust if your account uses another id).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("AnthropicProvider requires api_key or ANTHROPIC_API_KEY")
        self._api_key = key
        self._model = model

    def complete_turn(
        self,
        messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Any = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        raise NotImplementedError(
            "AnthropicProvider.complete_turn is not implemented: Anthropic uses the "
            "Messages API with different tool/function shapes than OpenAI chat/completions. "
            "Use OpenAIProvider or OpenAICompatibleProvider with aion.tools for tool loops in v1."
        )

    def complete(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        system_chunks: List[str] = []
        api_messages: List[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_chunks.append(m["content"])
                continue
            api_messages.append({"role": m["role"], "content": m["content"]})

        url = "https://api.anthropic.com/v1/messages"
        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_chunks:
            payload["system"] = "\n\n".join(system_chunks)
        payload.update(kwargs)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        data = post_json(url, payload, headers=headers)
        try:
            blocks = data["content"]
            out = []
            for b in blocks:
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
            return "".join(out)
        except (KeyError, TypeError) as e:
            raise ValueError(f"Unexpected Anthropic response shape: {data!r}") from e
