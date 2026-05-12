"""Multi-agent orchestration: delegate tasks to specialized sub-agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .react import ReActAgent


@dataclass
class AgentRole:
    """
    Defines a sub-agent's specialization.

    Parameters
    ----------
    name : str
        Unique role name (e.g. ``"researcher"``, ``"coder"``).
    description : str
        What this agent is good at (used by the orchestrator to route tasks).
    system_prompt : str
        System prompt for this agent's LLM calls.
    """

    name: str
    description: str
    system_prompt: str


class MultiAgent:
    """
    Orchestrator that routes sub-tasks to specialized agents.

    The orchestrator uses an LLM to decide which sub-agent should handle
    each part of a complex task, then combines their outputs.

    Parameters
    ----------
    provider
        LLM provider (used for orchestration and for each sub-agent).
    roles : list[AgentRole]
        Available sub-agent specializations.
    registry
        Tool registry shared by all sub-agents.
    tools
        Tool schemas shared by all sub-agents.
    """

    def __init__(
        self,
        provider: Any,
        roles: List[AgentRole],
        registry: Any,
        tools: List[Dict[str, Any]],
        *,
        max_delegations: int = 5,
    ) -> None:
        self.provider = provider
        self.roles = {r.name: r for r in roles}
        self.registry = registry
        self.tools = tools
        self.max_delegations = max_delegations
        self._agents: Dict[str, ReActAgent] = {}
        self.delegation_log: List[Dict[str, Any]] = []

        for role in roles:
            self._agents[role.name] = ReActAgent(
                provider=provider,
                registry=registry,
                tools=tools,
                system_prompt=role.system_prompt,
                max_steps=6,
            )

    def run(self, task: str) -> str:
        """
        Orchestrate a complex task across multiple agents.

        The orchestrator LLM decides which agent to delegate to and
        combines results into a final answer.
        """
        self.delegation_log = []
        import json
        from ..providers.base import ChatMessage

        role_descriptions = "\n".join(
            f"- {name}: {role.description}"
            for name, role in self.roles.items()
        )

        plan_prompt = (
            f"You have these specialist agents:\n{role_descriptions}\n\n"
            f"Task: {task}\n\n"
            f"Decide which agents to use and what sub-task each should handle. "
            f"Return a JSON array of objects with 'agent' (name) and 'task' (sub-task string). "
            f"Return ONLY the JSON array."
        )

        plan_response = self.provider.complete([
            ChatMessage(role="system", content="You are a task orchestrator."),
            ChatMessage(role="user", content=plan_prompt),
        ])

        try:
            start = plan_response.index("[")
            end = plan_response.rindex("]") + 1
            delegations = json.loads(plan_response[start:end])
        except (ValueError, json.JSONDecodeError):
            agent_name = next(iter(self.roles))
            delegations = [{"agent": agent_name, "task": task}]

        results: List[str] = []
        for d in delegations[: self.max_delegations]:
            agent_name = d.get("agent", "")
            sub_task = d.get("task", "")
            if agent_name not in self._agents:
                agent_name = next(iter(self._agents))

            agent = self._agents[agent_name]
            result = agent.run(sub_task)
            self.delegation_log.append({
                "agent": agent_name,
                "task": sub_task,
                "result": result[:500],
            })
            results.append(f"[{agent_name}] {result}")
            agent.reset()

        if len(results) == 1:
            return results[0].split("] ", 1)[-1]

        combine_prompt = (
            f"Original task: {task}\n\nResults from agents:\n" +
            "\n\n".join(results) +
            "\n\nCombine these into a single coherent answer."
        )
        final = self.provider.complete([
            ChatMessage(role="system", content="You synthesize information from multiple sources."),
            ChatMessage(role="user", content=combine_prompt),
        ])
        return final
