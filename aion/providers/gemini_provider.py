"""Google Gemini generateContent API (REST)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .base import ChatMessage
from .structured import AssistantTurn
from .http_utils import post_json


class GeminiProvider:
    """
    Chat via Gemini ``generateContent``.

    Parameters
    ----------
    api_key : str, optional
        Defaults to ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
    model : str, optional
        Model id without the ``models/`` prefix, default ``gemini-1.5-flash``.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GeminiProvider requires api_key or GEMINI_API_KEY / GOOGLE_API_KEY"
            )
        self._api_key = key
        self._model = model.lstrip("models/")

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
            "GeminiProvider.complete_turn is not implemented: Gemini generateContent "
            "uses a different schema than OpenAI chat/completions. "
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
        system_parts: List[str] = []
        contents: List[dict] = []
        for m in messages:
            role = m["role"]
            text = m["content"]
            if role == "system":
                system_parts.append(text)
                continue
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}],
            }
        payload.update({k: v for k, v in kwargs.items() if k in ("safetySettings",)})
        data = post_json(url, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected Gemini response shape: {data!r}") from e
