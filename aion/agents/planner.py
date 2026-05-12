"""Planning agent: decompose a task into sub-steps, then execute each."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .memory import SlidingWindowMemory


@dataclass
class PlanStep:
    description: str
    status: str = "pending"  # pending | running | done | failed
    result: str = ""


class PlanningAgent:
    """
    Agent that first generates a plan (list of steps), then executes each
    step using tool calls.

    Parameters
    ----------
    provider
        LLM provider with ``complete`` and ``complete_turn``.
    registry
        Tool registry.
    tools
        OpenAI-format tool schemas.
    system_prompt : str
        Prompt instructing the model how to plan.
    max_steps_per_action : int
        Tool-call budget per plan step.
    """

    def __init__(
        self,
        provider: Any,
        registry: Any,
        tools: List[Dict[str, Any]],
        *,
        system_prompt: str = (
            "You are a planning assistant. When given a task, first output a numbered "
            "plan of steps as a JSON array of strings. Then execute each step using tools."
        ),
        max_steps_per_action: int = 5,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps_per_action = max_steps_per_action
        self.plan: List[PlanStep] = []

    def run(self, task: str) -> str:
        """Plan and execute a task. Returns the final answer."""
        self.plan = self._generate_plan(task)

        results: List[str] = []
        for step in self.plan:
            step.status = "running"
            result = self._execute_step(step, task)
            step.result = result
            step.status = "done"
            results.append(f"- {step.description}: {result}")

        from ..providers.base import ChatMessage
        summary_prompt = (
            f"Task: {task}\n\nCompleted steps:\n" + "\n".join(results) +
            "\n\nPlease provide a final comprehensive answer."
        )
        final = self.provider.complete([
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=summary_prompt),
        ])
        return final

    def _generate_plan(self, task: str) -> List[PlanStep]:
        from ..providers.base import ChatMessage
        prompt = (
            f"Break this task into 2-5 clear steps. Return ONLY a JSON array of strings.\n\n"
            f"Task: {task}"
        )
        response = self.provider.complete([
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=prompt),
        ])
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            steps = json.loads(response[start:end])
            return [PlanStep(description=s) for s in steps]
        except (ValueError, json.JSONDecodeError):
            return [PlanStep(description=task)]

    def _execute_step(self, step: PlanStep, original_task: str) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Original task: {original_task}\n"
                f"Current step: {step.description}\n"
                "Complete this step using available tools, then respond with the result."
            )},
        ]
        for _ in range(self.max_steps_per_action):
            turn = self.provider.complete_turn(messages, tools=self.tools, temperature=0.2)
            if turn.tool_calls:
                asst: Dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
                from ..tools.loop import tool_calls_to_message_payload
                asst["tool_calls"] = tool_calls_to_message_payload(turn.tool_calls)
                messages.append(asst)
                for tc in turn.tool_calls:
                    result = self.registry.call(tc.name, tc.arguments_json)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            return turn.content or ""
        return "Step completed (max iterations reached)."

    def get_plan_summary(self) -> List[Dict[str, str]]:
        return [
            {"description": s.description, "status": s.status, "result": s.result[:200]}
            for s in self.plan
        ]
