from __future__ import annotations

import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

MODEL_NAME = "meta/llama-3.3-70b-instruct"

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Proactive pacing.
#
# The graph itself never calls the LLM concurrently (query_router,
# capability_resolver, and explanation_writer run one after another within
# a single graph run), so the "Worker local total request limit reached"
# errors come from this process's calls landing too close together in time
# — either back-to-back scenarios in one run, or another process sharing the
# same NVIDIA_API_KEY. Rather than let calls fire immediately and retry
# after they get rejected, every call is paced through a single global gate:
# at most one in-flight request at a time, with a minimum gap enforced
# between the start of one call and the next. This keeps this agent's own
# request pattern smooth and under the cap by construction, so the
# exponential-backoff retry below becomes a rarely-needed safety net rather
# than the primary defense.
# ---------------------------------------------------------------------------
_LLM_GATE = threading.Semaphore(1)
_pace_lock = threading.Lock()
_last_call_started_at = 0.0
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NVIDIA_LLM_MIN_INTERVAL", "2.5"))


def _pace() -> None:
    """Block until at least MIN_CALL_INTERVAL_SECONDS has passed since the
    previous call was allowed to start."""
    global _last_call_started_at
    with _pace_lock:
        now = time.monotonic()
        wait = MIN_CALL_INTERVAL_SECONDS - (now - _last_call_started_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_started_at = time.monotonic()


def get_llm_client() -> BaseChatModel:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set")

    return ChatNVIDIA(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )


def _is_transient_error(exc: Exception) -> bool:
    """True for rate-limit / service-unavailable errors worth retrying.

    Anything else (auth errors, bad requests, malformed tool calls, etc.)
    is left alone so it re-raises immediately and the existing fallback
    logic in query_router / capability_resolver / explanation_writer
    kicks in exactly as before.
    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "resourceexhausted",
            "rate limit",
            "rate_limit",
            "too many requests",
            "503",
            "429",
            "service unavailable",
        )
    )


def invoke_with_retry(
    invoke_fn: Callable[[], _T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> _T:
    """Call `invoke_fn()` through the pacing gate above, retrying with
    backoff only as a fallback safety net on transient NVIDIA endpoint
    errors (e.g. "ResourceExhausted ... request limit reached" 503).

    The pacing (one in-flight call at a time, minimum gap between calls) is
    what's meant to keep this agent under the shared limit in the first
    place; the retry loop only matters if something outside this process
    is also consuming the same quota at the same moment.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        with _LLM_GATE:
            _pace()
            try:
                return invoke_fn()
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc) or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Transient NVIDIA endpoint error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises