"""Helpers for tests: fake providers, temp paths (optional pytest)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..providers.structured import AssistantTurn, NormalizedToolCall


class FakeToolProvider:
    """
    Minimal provider for tests: returns scripted AssistantTurns in order.
    """

    def __init__(self, turns: Sequence[AssistantTurn]) -> None:
        self._turns = list(turns)
        self._i = 0

    def complete_turn(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        if self._i >= len(self._turns):
            return AssistantTurn(content="done", tool_calls=[], raw={})
        t = self._turns[self._i]
        self._i += 1
        return t


def temp_dir_context():
    """Return a tempfile.TemporaryDirectory context manager path factory."""
    return tempfile.TemporaryDirectory()


def make_tool_turn(
    tool_calls: List[NormalizedToolCall],
    content: Optional[str] = None,
) -> AssistantTurn:
    return AssistantTurn(content=content, tool_calls=list(tool_calls), raw={})
