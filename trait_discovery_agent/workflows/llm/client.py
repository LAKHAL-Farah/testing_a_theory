import asyncio
import logging
import os
import random
from functools import lru_cache

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# python-dotenv is already a declared dependency (requirements.txt) but was
# never actually invoked anywhere in the codebase -- .env only got picked up
# automatically under docker-compose (env_file:). A bare `python
# evaluation/run_eval.py` (or any local, non-docker invocation) read nothing
# from .env and silently fell back to whatever's already exported in the
# shell -- or, worse, to DEFAULT_MODEL's hardcoded value below if nothing
# was exported at all. Loading it here, at import time of the one module
# every LLM call routes through, means every entry point gets it for free.
# override=False: real environment variables (e.g. a CI secret) still win
# over .env, matching docker-compose's own env_file semantics.
load_dotenv(override=False)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")

# Additional models to try, in order, if DEFAULT_MODEL 404/410s (retired,
# renamed, or never enabled on this key), or if it keeps timing out/500ing
# after its retry budget (see _is_advance_worthy_error). Comma-separated env
# var so an operator can widen/reorder this without a code change;
# NIM_MODEL itself always tried first.
#
# NVIDIA's free-tier NIM catalog is being pruned unusually aggressively as of
# Aug 2026 (see e.g. github.com/diegosouzapw/OmniRoute#9824) -- multiple
# unrelated model families hitting 410 within the same window, and
# ChatNVIDIA.get_available_models() itself has been observed lagging behind
# real availability, so a short 2-model list gets exhausted fast. This list
# is deliberately long and spans *independent* publishers/architectures so a
# single vendor's deprecation wave can't take out the whole chain at once.
#
# Ordering, two rules:
# 1. Plain (non-reasoning) instruct models come FIRST. Hybrid-reasoning
#    models (Nemotron, DeepSeek-R1-style, Qwen3-thinking, GPT-OSS's default
#    analysis channel, ...) were observed burning the entire max_tokens
#    budget on visible chain-of-thought before ever reaching the JSON final
#    answer -- every LLM pick failed and silently fell back to deterministic
#    heuristics, tanking answer_relevancy/context_recall in the eval.
#    Nemotron reasoning models also expose a "/no_think"/"detailed thinking
#    off" toggle (_reasoning_off_preamble below) but it did NOT reliably
#    suppress the chain-of-thought in practice, so plain instruct models --
#    which were never trained to think out loud in the first place -- are
#    the primary defense, not the toggle. Where a publisher tags separate
#    "-instruct" vs "-thinking" variants (Qwen3, Kimi-K2), the "-instruct"
#    one is used here for exactly that reason.
# 2. Smaller/lighter models come before larger ones of similar type.
#    mistral-nemotron (NVIDIA's function-calling-tuned model, previously
#    primary here) was observed timing out and 500ing repeatedly across
#    several eval runs while still showing supports_tools=True and
#    "reachable" in list_nim_models.py -- i.e. genuinely live but
#    under-provisioned/overloaded on the free tier right now, not
#    deprecated. An 8B model has a much better shot at spare shared-pool
#    capacity than a large one, so smaller plain-instruct models now lead;
#    mistral-nemotron is demoted to mid-chain rather than dropped, since
#    "overloaded right now" isn't permanent the way a 410 is.
# The hybrid-reasoning models are kept at the tail as an absolute last
# resort, not removed entirely, in case every plain instruct model in this
# list gets pruned or is having a bad day at once.
# Re-run list_nim_models.py periodically and prune/reorder this if entries
# go stale.
_EXTRA_FALLBACKS = tuple(
    m.strip() for m in os.getenv(
        "NIM_FALLBACK_MODELS",
        "meta/llama-3.1-8b-instruct,"
        "ibm/granite-3.3-8b-instruct,"
        "microsoft/phi-4-mini-instruct,"
        "mistralai/mistral-small-3.1-24b-instruct-2503,"
        "qwen/qwen3-next-80b-a3b-instruct,"
        "moonshotai/kimi-k2-instruct,"
        "mistralai/mistral-nemotron,"
        "meta/llama-3.1-70b-instruct,"
        "meta/llama-3.1-405b-instruct,"
        "nvidia/nemotron-3-super-120b-a12b,"
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    ).split(",")
    if m.strip()
)
FALLBACK_MODELS = tuple(dict.fromkeys((DEFAULT_MODEL, *_EXTRA_FALLBACKS)))

# 2048 rather than the original 512: plain instruct models rarely need this
# much, but it's cheap headroom for verbose final-answer reasoning fields
# and for the last-resort reasoning models at the tail of FALLBACK_MODELS
# above, which need real room even with thinking nominally toggled off.
MAX_TOKENS = int(os.getenv("NIM_MAX_TOKENS", "2048"))

MAX_CAPACITY_RETRIES = 3
CAPACITY_RETRY_BASE_SECONDS = 1.5

# 429s (request-rate quota) are a fundamentally different failure than 503s
# (momentary worker-pool saturation): a free-tier NIM key's quota window is
# typically per-minute, not per-request, so retrying after 1-2s (the 503
# backoff) just burns the retry budget for no benefit — it needs a much more
# patient wait, and more attempts, before the quota actually clears.
# Configurable via env since the right amount of patience depends entirely
# on your tier/quota, which this code has no way to introspect.
MAX_RATE_LIMIT_RETRIES = int(os.getenv("NIM_MAX_RATE_LIMIT_RETRIES", "5"))
RATE_LIMIT_RETRY_BASE_SECONDS = float(os.getenv("NIM_RATE_LIMIT_RETRY_BASE_SECONDS", "10"))
RATE_LIMIT_RETRY_MAX_SECONDS = float(os.getenv("NIM_RATE_LIMIT_RETRY_MAX_SECONDS", "60"))

# Optional proactive throttle: a minimum gap enforced between the *start* of
# consecutive NIM calls, process-wide, regardless of which subagent issues
# them. Off by default (0s) since most deployments aren't quota-constrained
# and shouldn't pay a latency tax for it — but for a free-tier key that's
# already hitting 429s, spacing requests out can avoid tripping the limit in
# the first place rather than only reacting to it after the fact.
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NIM_MIN_CALL_INTERVAL_SECONDS", "0"))
_last_call_started_at: float = 0.0
_throttle_lock = asyncio.Lock()


def _get_api_key() -> str | None:
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.1, model: str | None = None) -> ChatNVIDIA:
    """Create a ChatNVIDIA client with the configured API key explicitly passed in."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_NIM_API_KEY is not set. Copy .env.example to .env and add your NIM key."
        )
    return ChatNVIDIA(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )


def _reasoning_off_preamble() -> str:
    """Prefix for the first system message, telling any reasoning-capable
    model in FALLBACK_MODELS to skip its chain-of-thought and answer
    directly. Every model here is expected to produce ONLY a JSON object as
    its final answer (see each subagent's system prompt) -- there's no
    caller that wants or parses visible reasoning tokens, so thinking mode
    only costs latency/tokens and, worse, can burn the whole max_tokens
    budget on the chain-of-thought before the model ever reaches the JSON
    (observed on NIM's Nemotron family, whose reasoning mode is on by
    default unless told otherwise).

    Stacks the different per-family directives NVIDIA documents (see
    docs.nvidia.com/nim/large-language-models/latest/reasoning-model.html
    and docs.nvidia.com/rag/2.5.0/enable-nemotron-thinking.html): Nemotron
    1.5 / Nano use '/no_think', older Nemotron reasoning models use the
    literal phrase 'detailed thinking off'. A model that doesn't recognize
    either is expected to just treat this as inert prefix text ahead of the
    real system prompt -- harmless for non-reasoning models."""
    return "/no_think\ndetailed thinking off\n\n"


def _is_missing_model_error(exc: Exception) -> bool:
    """True for '404 not found' (never enabled on this key), '410 Gone'
    (retired/EOL'd model, e.g. NVIDIA NIM catalog deprecations), and the
    malformed variant observed as '[###] Unknown Error\\npage not found' --
    a garbled/unparseable status code (literal '###') paired with what's
    almost certainly a truncated 'Page not found'. This happens when the
    gateway returns something that isn't clean JSON for a route that no
    longer resolves to a real model backend, so langchain_nvidia_ai_endpoints'
    own status-code parser has nothing valid to put in the brackets. All
    three mean the same thing -- 'this model id isn't servable right now' --
    so all three should advance to the next candidate in FALLBACK_MODELS
    rather than aborting the request. 'page not found' is matched
    unconditionally (not gated on a status-code digit, since here there
    isn't a usable one) -- specific enough a phrase that it shouldn't
    collide with an unrelated error."""
    message = str(exc).lower()
    if "410" in message and "gone" in message:
        return True
    if "404" in message and ("not found" in message or "function" in message):
        return True
    return "page not found" in message


def _is_advance_worthy_error(exc: Exception) -> bool:
    """Whether _candidate_models' caller should move on to the NEXT model in
    FALLBACK_MODELS rather than raising. True for missing-model errors
    (above), a timeout that _retry_on_capacity already retried and gave up
    on, and a persistent 5xx server error -- each of these means the model
    is, for the purposes of finishing this request, exactly as unusable as
    one that 404s, and there's no reason to sacrifice the whole
    gene/pathway/protein decision (falling back to a deterministic
    heuristic) when another model in the chain might just work. Rate-limit
    and capacity (503) errors are NOT included here: those are properties of
    your account/quota or of NIM's shared worker pool, not the specific
    model, so switching models wouldn't help and would just mask the real
    signal (slow down / check quota)."""
    return _is_missing_model_error(exc) or _is_timeout_error(exc) or _is_server_error(exc)


def _is_capacity_error(exc: Exception) -> bool:
    """503s from a saturated local NIM worker pool ('request limit reached',
    ResourceExhausted) are transient — worth a short retry, not an immediate
    fallback or model swap."""
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


def _is_timeout_error(exc: Exception) -> bool:
    """Raw socket/connection timeouts ('Timeout on reading data from
    socket', asyncio.TimeoutError, etc.) rather than an HTTP status code --
    langchain_nvidia_ai_endpoints' default 60s socket timeout with no retry
    of its own means one slow/cold-starting request previously killed the
    whole gene outright. Treated like _is_capacity_error: worth a handful of
    short retries (a stuck socket is often just a stuck socket), and unlike
    capacity/rate-limit errors, ALSO an advance-worthy error for
    _candidate_models (see _is_missing_model_error) -- a model that keeps
    timing out even after retries is as unusable right now as one that
    404s, and there's no reason to give up on the whole request rather than
    trying the next candidate."""
    message = str(exc).lower()
    return "timeout" in message or isinstance(exc, (TimeoutError, asyncio.TimeoutError))


def _is_server_error(exc: Exception) -> bool:
    """Generic 5xx from NIM's own gateway -- observed as both '[500]
    Internal Server Error' and the more opaque '[500] Unknown Error', each
    paired with a body like 'Inference connection error while making
    inference request'. Deliberately excludes anything _is_capacity_error
    already owns (503 specifically): that's a distinct, well-understood
    worker-pool-exhaustion signal with its own retry budget. A bare 500 here
    means NIM's gateway itself failed to reach the underlying model
    backend -- usually just as transient as a 503, so it gets a short retry,
    and (like a persistent timeout) is advance-worthy on FALLBACK_MODELS if
    it doesn't clear."""
    if _is_capacity_error(exc):
        return False
    message = str(exc).lower()
    return (
        "500" in message
        or "internal server error" in message
        or "inference connection error" in message
    )


MAX_TIMEOUT_RETRIES = int(os.getenv("NIM_MAX_TIMEOUT_RETRIES", "2"))
TIMEOUT_RETRY_BASE_SECONDS = float(os.getenv("NIM_TIMEOUT_RETRY_BASE_SECONDS", "2"))

MAX_SERVER_ERROR_RETRIES = int(os.getenv("NIM_MAX_SERVER_ERROR_RETRIES", "2"))
SERVER_ERROR_RETRY_BASE_SECONDS = float(os.getenv("NIM_SERVER_ERROR_RETRY_BASE_SECONDS", "2"))


def _is_rate_limit_error(exc: Exception) -> bool:
    """429s — a request-quota rate limit, distinct from worker-pool
    exhaustion (_is_capacity_error above) and needing a much longer backoff.
    langchain_nvidia_ai_endpoints raises a bare Exception (see its
    _common.py:_try_raise) with no structured status code or Retry-After
    header exposed to the caller — only this string — so string matching is
    all that's available here."""
    message = str(exc).lower()
    return "429" in message or "too many requests" in message


async def _throttle() -> None:
    """Optional proactive spacing between NIM calls (see
    MIN_CALL_INTERVAL_SECONDS). No-op unless explicitly configured."""
    if MIN_CALL_INTERVAL_SECONDS <= 0:
        return
    global _last_call_started_at
    async with _throttle_lock:
        wait = _last_call_started_at + MIN_CALL_INTERVAL_SECONDS - asyncio.get_event_loop().time()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_started_at = asyncio.get_event_loop().time()


async def _retry_on_capacity(coro_fn):
    """Call coro_fn() (a zero-arg async callable), retrying with jittered
    exponential backoff on transient worker-pool exhaustion (503), request-
    rate limiting (429), or raw socket timeouts — the rate-limit case gets a
    much longer backoff and a larger retry budget since it's a
    slower-clearing condition than the other two (see MAX_RATE_LIMIT_RETRIES
    vs MAX_CAPACITY_RETRIES / MAX_TIMEOUT_RETRIES)."""
    last_error: Exception | None = None
    attempt = 0
    while True:
        await _throttle()
        try:
            return await coro_fn()
        except Exception as exc:
            is_rate_limited = _is_rate_limit_error(exc)
            is_capacity = (not is_rate_limited) and _is_capacity_error(exc)
            is_timeout = (not is_rate_limited) and (not is_capacity) and _is_timeout_error(exc)
            is_server_error = (
                not (is_rate_limited or is_capacity or is_timeout) and _is_server_error(exc)
            )
            if not (is_rate_limited or is_capacity or is_timeout or is_server_error):
                raise
            if is_rate_limited:
                max_retries = MAX_RATE_LIMIT_RETRIES
            elif is_capacity:
                max_retries = MAX_CAPACITY_RETRIES
            elif is_timeout:
                max_retries = MAX_TIMEOUT_RETRIES
            else:
                max_retries = MAX_SERVER_ERROR_RETRIES
            if attempt >= max_retries:
                raise
            last_error = exc
            if is_rate_limited:
                delay = min(
                    RATE_LIMIT_RETRY_BASE_SECONDS * (2 ** attempt), RATE_LIMIT_RETRY_MAX_SECONDS
                ) + random.uniform(0, 1.0)
                logger.warning(
                    "NIM rate limited [429] (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            elif is_capacity:
                delay = CAPACITY_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM worker pool exhausted (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            elif is_timeout:
                delay = TIMEOUT_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM socket timeout (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            else:
                delay = SERVER_ERROR_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM server error [5xx] (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            await asyncio.sleep(delay)
            attempt += 1
    raise last_error  # pragma: no cover - unreachable


def _candidate_models(model: str | None) -> list[str]:
    """Build the [explicit model, *fallbacks] list shared by every invocation path."""
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)
    return candidate_models


def _parse_json_object(content: str) -> dict | None:
    """Best-effort extraction of a single JSON object from a model's final text,
    tolerating markdown code fences."""
    import json

    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None