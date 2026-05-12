"""ReAct (Reason + Act) agent loop."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from .memory import Memory, SlidingWindowMemory


class ReActAgent:
    """
    ReAct agent: observe -> think -> act loop.

    Uses an LLM provider with tool-calling support to iteratively solve
    a task. The agent calls tools via :mod:`aion.tools.ToolRegistry` and
    feeds results back to the model until it produces a final answer.

    Parameters
    ----------
    provider
        Any provider supporting ``complete_turn`` (e.g. ``OpenAIProvider``).
    registry
        ``ToolRegistry`` with registered tool implementations.
    tools : list[dict]
        OpenAI-format tool schema list.
    system_prompt : str
        System prompt guiding the agent's behavior.
    memory : Memory, optional
        Conversation memory strategy. Defaults to a 20-message sliding window.
    max_steps : int
        Maximum tool-calling iterations before forcing a final answer.
    on_step : callable, optional
        ``(step_num, action, result) -> None`` callback for observability.
    """

    def __init__(
        self,
        provider: Any,
        registry: Any,
        tools: List[Dict[str, Any]],
        *,
        system_prompt: str = (
            "You are a helpful assistant. Use the provided tools when needed "
            "to answer the user's question. Think step by step."
        ),
        memory: Optional[Memory] = None,
        max_steps: int = 10,
        on_step: Optional[Callable[..., None]] = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory: Memory = memory or SlidingWindowMemory(system_prompt=system_prompt)
        self.max_steps = max_steps
        self.on_step = on_step
        self.steps: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> str:
        """
        Run the agent loop for a single user query.

        Returns the final text response.
        """
        self.steps = []
        self.memory.add({"role": "user", "content": user_input})

        for step_num in range(self.max_steps):
            messages = self.memory.get_messages()
            turn = self.provider.complete_turn(
                messages, tools=self.tools, temperature=0.2
            )

            if turn.tool_calls:
                asst_msg: Dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
                self.memory.add(asst_msg)

                for tc in turn.tool_calls:
                    result = self.registry.call(tc.name, tc.arguments_json)
                    step_info = {
                        "step": step_num,
                        "action": tc.name,
                        "args": tc.arguments_json,
                        "result": result[:500],
                    }
                    self.steps.append(step_info)
                    if self.on_step:
                        self.on_step(step_num, tc.name, result)

                    self.memory.add({
                        "role": "tool",
                        "content": result,
                    })
                continue

            answer = turn.content or ""
            self.memory.add({"role": "assistant", "content": answer})
            return answer

        final_messages = self.memory.get_messages()
        final_messages.append({
            "role": "user",
            "content": "Please provide your final answer now based on what you've learned.",
        })
        from ..providers.base import ChatMessage
        final = self.provider.complete(
            [ChatMessage(role=m["role"], content=m["content"]) for m in final_messages]
        )
        self.memory.add({"role": "assistant", "content": final})
        return final

    def chat(self, user_input: str) -> str:
        """Multi-turn chat: memory persists across calls."""
        return self.run(user_input)

    def reset(self) -> None:
        """Clear memory and step history."""
        self.memory.clear()
        self.steps = []
