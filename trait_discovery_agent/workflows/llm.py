import json
import os
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

DEFAULT_MODEL = os.getenv("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
)

# Hard ceiling on tool-call/tool-result round-trips inside a single bind_tools
# decision, so a model that keeps calling tools instead of answering can't spin
# forever. One real decision (§0.1 of the guide) should resolve in 1-2 turns.
MAX_TOOL_TURNS = 4


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
        max_tokens=512,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )


def _is_missing_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message and ("not found" in message or "function" in message)


async def invoke_with_fallback(
    prompt: ChatPromptTemplate,
    payload: dict,
    *,
    temperature: float = 0.1,
    model: str | None = None,
):
    """Invoke a prompt against NIM, retrying with a known-good fallback model on 404s.

    The account behind this workspace can expose different model/function IDs over time.
    When one alias disappears, the graph should still be able to complete with an
    alternate model rather than failing hard at the first escalation node.
    """
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        llm = get_llm(temperature=temperature, model=candidate_model)
        chain = prompt | llm
        try:
            return await chain.ainvoke(payload)
        except Exception as exc:  # pragma: no cover - exercised in live NIM runs
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")


async def invoke_tool_loop_with_fallback(
    system_prompt: str,
    human_prompt: str,
    tools: list,
    *,
    temperature: float = 0.1,
    model: str | None = None,
    max_turns: int = MAX_TOOL_TURNS,
) -> tuple[dict, list[dict]]:
    """Run a bind_tools ReAct-style loop against NIM, with the same 404-driven
    model fallback as invoke_with_fallback.

    The model is given typed, direct access to `tools` (per guide §2/§5) and may
    call them zero or more times before returning a final JSON object as plain
    text. Returns (parsed_final_json, tool_call_log) — the tool_call_log lets
    the caller validate the final answer mechanically against what the tools
    actually returned (grounding rule, §0.1), rather than trusting the model's
    say-so.

    Raises RuntimeError if the model never produces a final JSON answer within
    max_turns, or if every candidate model 404s.
    """
    tools_by_name = {t.name: t for t in tools}
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        llm = get_llm(temperature=temperature, model=candidate_model)
        llm_with_tools = llm.bind_tools(tools)

        convo: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        tool_call_log: list[dict] = []

        try:
            for _ in range(max_turns):
                ai_msg = await llm_with_tools.ainvoke(convo)
                convo.append(ai_msg)

                tool_calls = getattr(ai_msg, "tool_calls", None)
                if not tool_calls:
                    parsed = _parse_json_object(ai_msg.content)
                    if parsed is None:
                        raise RuntimeError(
                            f"Model returned a non-JSON final answer: {ai_msg.content!r}"
                        )
                    return parsed, tool_call_log

                for call in tool_calls:
                    tool = tools_by_name[call["name"]]
                    result = await tool.ainvoke(call["args"])
                    tool_call_log.append(
                        {"name": call["name"], "args": call["args"], "result": result}
                    )
                    convo.append(
                        ToolMessage(
                            content=json.dumps(result, default=str),
                            tool_call_id=call["id"],
                        )
                    )

            raise RuntimeError(
                f"Tool-calling loop exceeded {max_turns} turns without a final answer"
            )
        except Exception as exc:
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")


def _parse_json_object(content: str) -> dict | None:
    """Best-effort extraction of a single JSON object from a model's final text,
    tolerating markdown code fences."""
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