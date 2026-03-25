"""Errors raised by LLM provider clients."""

from typing import Optional


class ProviderError(Exception):
    """HTTP or API error from a remote LLM provider."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
