"""Minimal JSON HTTP POST using the standard library (no extra deps)."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .errors import ProviderError


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    POST JSON and parse JSON response.

    Raises
    ------
    ProviderError
        On network failure or non-2xx response.
    """
    data = json.dumps(payload).encode("utf-8")
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise ProviderError(
            f"HTTP {e.code}: {e.reason}",
            status=e.code,
            body=body[:4000],
        ) from e
    except urllib.error.URLError as e:
        raise ProviderError(f"Request failed: {e.reason}") from e
