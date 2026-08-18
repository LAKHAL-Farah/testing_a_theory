from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - Docker/Linux only in practice
    fcntl = None

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(Path(__file__).parent.parent / ".env")

# The NVIDIA client warns on every bind_tools() call for models it doesn't
# have hardcoded tool-support metadata for, and separately warns on every
# client construction when a model isn't in its known-type list. Both are
# one-time-relevant compatibility notices, not per-call signals — this
# model demonstrably works for both tool binding and inference in this
# codebase — so they're just noise on every single invocation. Silenced at
# the source rather than suppressed per-call so they can't leak from any
# import path.
warnings.filterwarnings(
    "ignore",
    message=r".*is not known to support tools.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*type is unknown and inference may fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*is unknown, check `available_models`.*",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "meta/llama-3.3-70b-instruct"

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Proactive pacing / concurrency limiting.
#
# docker-compose.yml runs full_species_resolver_demo and
# full_genome_metadata_demo (plus genome-agent-scenarios, ncbi-live-check,
# etc.) as *separate containers*, all with `env_file: .env`, so they all
# share one free-tier NVIDIA_API_KEY. An in-memory threading.Semaphore only
# serializes calls made by this one process — it does nothing to stop two+
# containers from each calling NVIDIA at once.
#
# IMPORTANT: "Worker local total request limit reached (17/16)" is a
# *concurrent in-flight requests* cap, not a requests-per-minute cap.
# Spacing out when calls *start* (the original approach) doesn't fix this —
# if a call takes longer to complete than the gap between two call starts
# (very likely: NIM can take several seconds per turn, and the tool loop
# makes several turns), multiple calls end up in flight at once regardless
# of how evenly their starts were spaced. The only thing that actually
# bounds concurrency is holding a lock for the *entire* call, not just the
# gap before it.
#
# Every container here bind-mounts the same host directory
# (`.:/app/gene_agent`), so an flock on a small state file inside that
# mount gives every process — regardless of container — a real cross-process
# mutex: at most one NVIDIA call, from any of these processes, is ever in
# flight at a time. That's a stricter guarantee than the free-tier's
# 16-concurrent cap needs, but on a free tier "well under the limit" is the
# right target, not "right up against it". The min-interval pacing is kept
# on top of that as a courtesy gap between one call finishing and the next
# starting, not as the primary defense anymore.
# ---------------------------------------------------------------------------
_LLM_GATE = threading.Semaphore(1)
_pace_lock = threading.Lock()
_last_call_started_at = 0.0
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NVIDIA_LLM_MIN_INTERVAL", "1.5"))

# Shared across every process that checks out this repo (dev machine or any
# docker-compose service), since they all mount the same gene_agent/ dir.
_PACE_STATE_FILE = Path(__file__).parent.parent / ".nim_pace_state"


@contextlib.contextmanager
def _llm_call_slot():
    """Cross-process mutex held for the *entire* duration of one NVIDIA
    call (including any retries/backoff inside it) — not just the pacing
    gap before it starts. This is what actually caps concurrent in-flight
    requests across every process sharing this key; see the module-level
    comment above for why pacing alone can't do that.

    Falls back to in-process-only locking (the original behavior) if flock
    isn't available (e.g. a non-POSIX dev environment) or the state file
    can't be opened, so a missing lock never becomes a hard failure — it
    just quietly loses the cross-process guarantee.
    """
    if fcntl is None:
        with _LLM_GATE:
            yield
        return

    try:
        _PACE_STATE_FILE.touch(exist_ok=True)
        f = open(_PACE_STATE_FILE, "r+")
    except OSError as exc:
        logger.warning(
            "Cross-process NVIDIA locking unavailable (%s) — falling back "
            "to in-process locking only for this call.",
            exc,
        )
        with _LLM_GATE:
            yield
        return

    try:
        # Blocks here until no other process anywhere has a call in flight.
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            last = float(raw) if raw else 0.0
            wait = MIN_CALL_INTERVAL_SECONDS - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            yield
        finally:
            f.seek(0)
            f.truncate()
            f.write(repr(time.time()))
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


@lru_cache(maxsize=None)
def get_llm_client() -> BaseChatModel:
    """Build (once) and reuse the ChatNVIDIA client.

    Constructing ChatNVIDIA is not free: for a model not in the library's
    hardcoded registry (this one isn't — that's why you see the "type is
    unknown" warning), its constructor calls `self.available_models`,
    which makes a real HTTP GET to NVIDIA to list every model it offers,
    just to validate the name. Without caching, `route_query`,
    `resolve_capability`, and `write_explanation` would each rebuild the
    client — tripling both the network round trips per scenario and the
    requests counted against the free-tier cap, for zero benefit, since
    nothing about the client changes between calls in this process.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set")

    return ChatNVIDIA(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        # The library defaults this to 60s, and it isn't just an HTTP
        # timeout — on a 202 ("still processing") response it polls every
        # 0.02s for up to `timeout` seconds before giving up (see
        # `_wait()` in langchain_nvidia_ai_endpoints._common). Under
        # free-tier saturation that's exactly the response you get, so
        # the default means a single call can silently hang for up to a
        # minute before our fallback logic ever gets a chance to run.
        # Failing in a few seconds instead means the deterministic
        # fallback kicks in fast, which is the whole point of having one.
        timeout=float(os.getenv("NVIDIA_LLM_TIMEOUT", "8")),
    )


def summarize_llm_error(exc: Exception) -> str:
    """Collapse the NVIDIA client's raw error text into one short line.

    The client's own error formatting (`_format_error` in
    langchain_nvidia_ai_endpoints) builds its message as
    `"[status] {title_or_dict}\\n{full_response_dict}"`. When the response
    body has no "detail" key — which is the case for the free-tier
    ResourceExhausted 503 — that second line is just the entire raw
    response dict repeated. Logging `str(exc)` as-is prints that whole
    two-line blob on every failure. This pulls out just the useful part.
    """
    if _is_rate_limited_error(exc):
        return "NVIDIA endpoint rate-limited (429, over the free-tier RPM cap) — using fallback"
    if _is_capacity_error(exc):
        return "NVIDIA endpoint worker pool saturated (503, transient) — using fallback"
    # Fall back to the first line only, so unexpected errors still surface
    # without dragging the raw response dict into the log.
    return str(exc).splitlines()[0]


def _is_capacity_error(exc: Exception) -> bool:
    """True for 503 / ResourceExhausted worker-pool saturation.

    This is a "busy right now" signal, not a rate-limit signal — the local
    NIM worker pool is momentarily full, and it commonly clears within a
    couple seconds. Worth a short retry. Matches trait_discovery_agent's
    workflows/llm/client.py::_is_capacity_error.
    """
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


def _is_rate_limited_error(exc: Exception) -> bool:
    """True for 429 — you are over your requests-per-minute cap.

    Deliberately NOT retried (see _is_transient_error below): retrying a
    429 immediately just adds another request to a bucket that's already
    over its limit, which can't help and often makes the next request also
    429. The right response to a 429 is to fail straight to the
    deterministic fallback, not to retry — same split trait_discovery_agent
    makes by excluding 429 from its _is_capacity_error.
    """
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message or "rate_limit" in message


def _is_transient_error(exc: Exception) -> bool:
    """True only for errors worth retrying — i.e. capacity (503), not rate
    limit (429). Anything else (auth errors, bad requests, malformed tool
    calls, 429s) is left alone so it re-raises immediately and the existing
    fallback logic in query_router / capability_resolver / explanation_writer
    kicks in right away.
    """
    return _is_capacity_error(exc)


def invoke_with_retry(
    invoke_fn: Callable[[], _T],
    *,
    max_retries: int = 1,
    base_delay: float = 2.5,
) -> _T:
    """Call `invoke_fn()` through the pacing gate above.

    Only retries on a capacity (503) error — see _is_transient_error /
    _is_capacity_error. A 429 (rate limit) is deliberately NOT retried here:
    it raises immediately on the first attempt regardless of max_retries,
    matching trait_discovery_agent's client.py. Every caller of this
    function (query_router, capability_resolver, explanation_writer, and
    the species_resolver / genome_metadata subagents) already has a fast,
    deterministic non-LLM fallback for exactly this case, so failing fast
    on a 429 gets to that fallback right away instead of burning several
    seconds retrying a request that's likely to 429 again.

    Pass max_retries > 1 to ride out a genuinely transient 503 capacity dip
    (worth doing — those often clear within a couple seconds). It has no
    effect on 429 handling.

    The whole attempt sequence (including backoff sleeps between retries)
    runs inside one _llm_call_slot() acquisition, so a retry here can't
    interleave with another process's call and re-trigger the same
    concurrent-request-limit error it's trying to recover from.
    """
    last_exc: Exception | None = None
    with _llm_call_slot():
        for attempt in range(max_retries):
            try:
                return invoke_fn()
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc) or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 1.0)
                logger.warning(
                    "Transient NVIDIA endpoint error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises