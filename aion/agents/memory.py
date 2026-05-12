"""Conversation memory strategies for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Memory(Protocol):
    """Protocol for conversation memory."""

    def add(self, message: Dict[str, str]) -> None: ...
    def get_messages(self) -> List[Dict[str, str]]: ...
    def clear(self) -> None: ...


class SlidingWindowMemory:
    """
    Keep the last *window_size* messages (plus an optional system prompt).

    Parameters
    ----------
    window_size : int
        Maximum number of user/assistant messages to retain.
    system_prompt : str, optional
        Always-present system message at the start.
    """

    def __init__(self, window_size: int = 20, system_prompt: Optional[str] = None) -> None:
        self._window = window_size
        self._system = system_prompt
        self._messages: List[Dict[str, str]] = []

    def add(self, message: Dict[str, str]) -> None:
        self._messages.append(message)
        if len(self._messages) > self._window:
            self._messages = self._messages[-self._window:]

    def get_messages(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self._system:
            out.append({"role": "system", "content": self._system})
        out.extend(self._messages)
        return out

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class SummaryMemory:
    """
    Summarize older messages into a running summary, keeping recent ones verbatim.

    Requires a ``summarize_fn`` (e.g. an LLM call) that compresses a list of
    message dicts into a single summary string.

    Parameters
    ----------
    keep_recent : int
        Number of recent messages to keep verbatim.
    summarize_fn : callable
        ``(List[Dict]) -> str`` that produces a summary.
    """

    def __init__(
        self,
        keep_recent: int = 6,
        summarize_fn: Optional[Any] = None,
    ) -> None:
        self._keep = keep_recent
        self._summarize = summarize_fn or self._default_summarize
        self._messages: List[Dict[str, str]] = []
        self._summary: str = ""

    def add(self, message: Dict[str, str]) -> None:
        self._messages.append(message)
        if len(self._messages) > self._keep * 2:
            overflow = self._messages[: -self._keep]
            self._summary = self._summarize(overflow, self._summary)
            self._messages = self._messages[-self._keep:]

    def get_messages(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self._summary:
            out.append({
                "role": "system",
                "content": f"Previous conversation summary: {self._summary}",
            })
        out.extend(self._messages)
        return out

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    @staticmethod
    def _default_summarize(messages: List[Dict[str, str]], prev_summary: str) -> str:
        parts = []
        if prev_summary:
            parts.append(prev_summary)
        for m in messages:
            parts.append(f"{m.get('role', 'unknown')}: {m.get('content', '')[:100]}")
        return " | ".join(parts)[-500:]


class TokenBudgetMemory:
    """
    Keep as many recent messages as fit within a token budget.

    Uses a simple word-count estimate (1 token ~ 0.75 words) unless
    a custom ``count_fn`` is provided.

    Parameters
    ----------
    max_tokens : int
        Token budget for the message window.
    system_prompt : str, optional
        Always-present system message.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        count_fn: Optional[Any] = None,
    ) -> None:
        self._max = max_tokens
        self._system = system_prompt
        self._count = count_fn or self._word_estimate
        self._messages: List[Dict[str, str]] = []

    def add(self, message: Dict[str, str]) -> None:
        self._messages.append(message)
        while self._total_tokens() > self._max and len(self._messages) > 1:
            self._messages.pop(0)

    def get_messages(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self._system:
            out.append({"role": "system", "content": self._system})
        out.extend(self._messages)
        return out

    def clear(self) -> None:
        self._messages.clear()

    def _total_tokens(self) -> int:
        return sum(self._count(m.get("content", "")) for m in self._messages)

    @staticmethod
    def _word_estimate(text: str) -> int:
        return max(1, int(len(text.split()) / 0.75))
