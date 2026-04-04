"""
LLM tool-calling helpers: schemas, registry, multi-turn loop, retry, rate limit.

Works with ``OpenAIProvider`` / ``OpenAICompatibleProvider.complete_turn``.
Optional: ``tiktoken`` via ``pip install aqwel-aion[tools]`` for token estimates.
"""

from .loop import run_tool_loop, tool_calls_to_message_payload
from .rate_limit import TokenBucket
from .registry import ToolRegistry
from .retry import post_json_with_retry
from .schemas import function_tool
from .tokens import estimate_messages_tokens_openai, estimate_text_tokens_openai

__all__ = [
    "ToolRegistry",
    "estimate_messages_tokens_openai",
    "estimate_text_tokens_openai",
    "function_tool",
    "post_json_with_retry",
    "run_tool_loop",
    "tool_calls_to_message_payload",
    "TokenBucket",
]
