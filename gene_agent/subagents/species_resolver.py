"""
Species Resolver — real NCBI Assembly eutils subagent
Resolves a species name to its genome assembly ID using NCBI eutils.
Never cached — always reflects NCBI's current record (see the
consolidated store-or-not matrix: cheap to refetch, must stay live).

Prefers RefSeq (GCF_) assemblies. Uses an esearch filter when available,
falls back to client-side GCF_-over-GCA_ preference when no RefSeq
assembly exists for a species.

Output shape matches schemas.outputs.SpeciesResolverOutput exactly
(assembly_id, scientific_name, common_name, confidence). The old mock
version of this module also carried a "taxonomic_group" field and a
get_all_species() helper used by subagents/visualization.py's
size_comparison scope — neither is part of that schema, and neither has
a cheap live-API equivalent (NCBI has no "list every species" call), so
both were dropped here. See visualization.py for how size_comparison
was adapted to work without them.
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ._ncbi_client import ncbi_get
from ..schemas.outputs import SpeciesResolverOutput
from ..workflows.llm import coerce_null_sentinels, get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


async def _search_taxonomy_core(query: str) -> list[dict]:
    """Search NCBI taxonomy for candidates matching a query string."""
    term = query.strip()
    if not term:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "taxonomy",
            "term": term,
            "retmode": "json",
            "retmax": 10,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "taxonomy",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    candidates = []
    for uid in uid_list:
        info = results.get(uid, {})
        candidates.append(
            {
                "tax_id": info.get("TaxId", uid),
                "scientific_name": info.get("ScientificName", ""),
                "common_name": info.get("CommonName", ""),
                "rank": info.get("Rank", ""),
            }
        )
    return candidates


async def _search_assembly_by_taxid_core(tax_id: str) -> list[dict]:
    """Search NCBI assembly for a given taxonomy ID.

    Tries the latest RefSeq filter first (retmax=1). If that returns
    zero results, falls back to an unfiltered search (retmax=20) with
    client-side GCF_-over-GCA_ preference.
    """
    filtered_term = f"txid{tax_id}[Organism:exp] AND latest_refseq[filter]"
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": filtered_term,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if uid_list:
        uid = uid_list[0]
        resp = await asyncio.to_thread(
            ncbi_get,
            {
                "path": "esummary.fcgi",
                "db": "assembly",
                "id": uid,
                "retmode": "json",
            },
        )
        data = resp.json()
        info = data.get("result", {}).get(uid, {})
        assembly_id = info.get("assemblyaccession", "")
        if assembly_id.startswith("GCF_"):
            return [
                {
                    "assembly_id": assembly_id,
                    "scientific_name": info.get("organism", ""),
                    "common_name": info.get("organism", ""),
                    "assembly_level": info.get("assemblylevel", ""),
                }
            ]

    unfiltered_term = f"txid{tax_id}[Organism:exp]"
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": unfiltered_term,
            "retmode": "json",
            "retmax": 20,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return []

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    assemblies = []
    for uid in uid_list:
        info = results.get(uid, {})
        assemblies.append(
            {
                "assembly_id": info.get("assemblyaccession", ""),
                "scientific_name": info.get("organism", ""),
                "common_name": info.get("organism", ""),
                "assembly_level": info.get("assemblylevel", ""),
            }
        )

    assemblies.sort(key=lambda x: (not x["assembly_id"].startswith("GCF_"), x["assembly_id"]))
    return assemblies


_SPECIES_RESOLVER_SYSTEM_PROMPT = (
    "You are the Species Resolver for the Genome Agent. Your job is to identify "
    "the correct genome assembly for a given species name using the available tools.\n\n"
    "Rules:\n"
    "1. ALWAYS call search_taxonomy first with the exact species name provided.\n"
    "2. If search_taxonomy returns multiple candidates, disambiguate by comparing "
    "common names and scientific names. Lower confidence if ambiguity remains.\n"
    "3. If a search returns empty results, try ONE reformulated query (e.g., add or "
    "remove qualifiers like 'asian', 'african', etc.).\n"
    "4. After identifying a candidate tax_id, call search_assembly_by_taxid to find assemblies.\n"
    "5. Before submitting SpeciesResolverOutput, verify that the scientific_name in "
    "your answer matches the organism name in the assembly results.\n"
    "6. If no assembly is found, submit assembly_id=null, confidence=0.0, and an "
    "honest reasoning note.\n"
    "7. NEVER fabricate an assembly_id. Only submit an assembly_id that literally "
    "appears in a search_assembly_by_taxid result.\n"
    "8. ALWAYS fill in reasoning with a one-sentence explanation of your choice, "
    "even when confidence is 1.0 and the match is unambiguous (e.g. 'search_taxonomy "
    "returned a single exact match and search_assembly_by_taxid returned its "
    "RefSeq assembly'). Never leave reasoning empty.\n"
    "9. If a tool call you already made returns the same result again, do not "
    "repeat it verbatim — either try a genuinely different query, move to the "
    "next tool, or submit assembly_id=null with an honest reasoning note.\n"
)


@tool
async def search_taxonomy(query: str) -> list[dict]:
    """Search NCBI taxonomy for candidates matching a query string."""
    return await _search_taxonomy_core(query)


@tool
async def search_assembly_by_taxid(tax_id: str) -> list[dict]:
    """Search NCBI assembly for a given taxonomy ID."""
    return await _search_assembly_by_taxid_core(tax_id)


def _check_duplicate_tool_call(
    messages: list,
    call_name: str,
    call_args: dict,
) -> tuple[bool, str | None]:
    """Check if this tool call duplicates a previous one in the message history.

    Same guard pattern as genome_metadata.py's resolver. Without this, a model
    that gets an empty result from search_taxonomy has nothing stopping it from
    calling search_taxonomy with the *same* query again on the next step —
    which is exactly the "asain elefant" x4 loop seen in the wild. On a repeat,
    we hand back the cached result instead of re-invoking the tool, and nudge
    the model to change its approach.

    Returns (is_duplicate, cached_result_or_None).
    """
    seen_calls: dict[tuple[str, str], str] = {}
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for prev_call in msg.tool_calls:
                prev_key = (prev_call["name"], json.dumps(prev_call["args"], sort_keys=True))
                for later_msg in messages:
                    if hasattr(later_msg, "tool_call_id") and later_msg.tool_call_id == prev_call["id"]:
                        seen_calls[prev_key] = str(later_msg.content)
                        break

    call_key = (call_name, json.dumps(call_args, sort_keys=True))
    if call_name in ("search_taxonomy", "search_assembly_by_taxid"):
        if call_key in seen_calls:
            return True, seen_calls[call_key]
    return False, None


async def resolve_species_llm(species_name: str) -> dict | None:
    """Use the LLM with tool calling to resolve a species to an assembly.

    Returns a dict matching SpeciesResolverOutput on success, or None if the
    LLM path fails or exhausts its retry budget.
    """
    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [search_taxonomy, search_assembly_by_taxid, SpeciesResolverOutput],
        tool_choice="auto",
    )

    messages: list = [
        SystemMessage(content=_SPECIES_RESOLVER_SYSTEM_PROMPT),
        HumanMessage(content=f"Resolve the species: {species_name}"),
    ]

    seen_tool_results: list[dict] = []
    # Human-readable trace of what happened at each step, logged on
    # exhaustion so a silent non-convergence (as opposed to an API error)
    # is actually diagnosable instead of just "SKIP" — see the log call
    # after the loop below.
    step_trace: list[str] = []

    # 8 steps, not 6: bumped again after switching MODEL_NAME to the
    # smaller/less-contended meta/llama-3.1-8b-instruct (see workflows/llm.py
    # for why). A smaller model is more prone to needing an extra retry
    # round on a rejected/ungrounded submission or a redundant tool call
    # before it converges on the same disambiguation work the 70b model did
    # in fewer steps — this is part of the same compromise (weaker
    # reasoning, so give it more room to still land correctly rather than
    # exhausting the budget on a case like "elephant" that needs several
    # genuine round trips even for a strong model).
    max_steps = 8
    for step in range(max_steps):
        try:
            response = await asyncio.to_thread(
                invoke_with_retry,
                lambda: bound.invoke(messages),
                # Covers both a transient 503 capacity dip (exponential
                # backoff, up to this many attempts) and a 429 rate limit
                # (a single delayed retry, capped regardless of this
                # number) — see workflows/llm.py's invoke_with_retry /
                # _is_rate_limited_error / _is_capacity_error split.
                max_retries=4,
            )
        except Exception as exc:
            # WARNING, not INFO: this is the one line that explains an
            # otherwise-silent None return (a "SKIP" in the scenario
            # runner, a quiet deterministic-fallback in production). Every
            # caller of this module sets the root logger to WARNING to
            # keep other subagents' chatter out of the trace (see
            # scripts/run_species_resolver_scenarios.py) — at INFO level
            # this line never reached stderr at all, so a genuine 429 and
            # a merely-slow/misconfigured client were both indistinguishable
            # from the outside.
            logger.warning("LLM species resolver failed: %s", summarize_llm_error(exc))
            return None

        tool_calls = response.tool_calls or []
        if not tool_calls:
            step_trace.append(f"step {step + 1}: model returned no tool calls (content: {str(response.content)[:120]!r})")
            continue

        messages.append(AIMessage(content="", tool_calls=tool_calls))

        for call in tool_calls:
            call_id = call["id"]
            call_name = call["name"]
            call_args = call["args"]

            is_dup, cached = _check_duplicate_tool_call(messages, call_name, call_args)
            if is_dup:
                step_trace.append(f"step {step + 1}: repeat call to {call_name}({call_args}) intercepted")
                logger.info(
                    "[guard] repeat call to %s(%s) intercepted — returning cached result",
                    call_name,
                    call_args,
                )
                messages.append(
                    ToolMessage(
                        content=(
                            f"You already called {call_name}({call_args}) — here is that "
                            f"same result again: {cached}. Calling it again with identical "
                            "arguments will not produce a different result. Either change "
                            "your query meaningfully, move on to the next tool, or submit "
                            "SpeciesResolverOutput with assembly_id=null and confidence=0.0 "
                            "if nothing further can be found."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue

            if call_name == "search_taxonomy":
                try:
                    result = await search_taxonomy.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)
                    step_trace.append(f"step {step + 1}: search_taxonomy({call_args}) -> {len(result)} result(s)")
                else:
                    step_trace.append(f"step {step + 1}: search_taxonomy({call_args}) -> {str(result)[:120]!r}")

            elif call_name == "search_assembly_by_taxid":
                try:
                    result = await search_assembly_by_taxid.ainvoke(call_args)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                if isinstance(result, list):
                    seen_tool_results.extend(result)
                    step_trace.append(
                        f"step {step + 1}: search_assembly_by_taxid({call_args}) -> {len(result)} result(s)"
                    )
                else:
                    step_trace.append(
                        f"step {step + 1}: search_assembly_by_taxid({call_args}) -> {str(result)[:120]!r}"
                    )

            elif call_name == "SpeciesResolverOutput":
                try:
                    parsed = SpeciesResolverOutput(
                        **coerce_null_sentinels(
                            call_args, {"assembly_id", "scientific_name", "common_name"}
                        )
                    )
                except Exception as exc:
                    step_trace.append(f"step {step + 1}: SpeciesResolverOutput rejected — parse error: {exc}")
                    messages.append(
                        ToolMessage(
                            content=f"Error parsing output: {exc}",
                            tool_call_id=call_id,
                        )
                    )
                    continue

                if parsed.assembly_id is not None:
                    grounded = any(
                        isinstance(item, dict) and item.get("assembly_id") == parsed.assembly_id
                        for item in seen_tool_results
                    )
                    if not grounded:
                        step_trace.append(
                            f"step {step + 1}: SpeciesResolverOutput rejected — assembly_id "
                            f"{parsed.assembly_id!r} not grounded in any tool result"
                        )
                        messages.append(
                            ToolMessage(
                                content=(
                                    f"Error: assembly_id '{parsed.assembly_id}' not found in any "
                                    "tool result. Please search for assemblies first and use an "
                                    "assembly_id from the results."
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        continue

                if not parsed.reasoning or not parsed.reasoning.strip():
                    step_trace.append(f"step {step + 1}: SpeciesResolverOutput rejected — empty reasoning")
                    messages.append(
                        ToolMessage(
                            content=(
                                "Error: reasoning must not be empty. Give a one-sentence "
                                "explanation of why this assembly_id (or null) and this "
                                "confidence are correct, then resubmit SpeciesResolverOutput."
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

                return parsed.model_dump()

            else:
                step_trace.append(f"step {step + 1}: unrecognized tool call {call_name}({call_args})")

    # WARNING, not INFO — same rationale as the exception-path log above:
    # this is the other silent-None cause (loop exhausted without ever
    # producing a valid grounded submission, vs. an exception), and at
    # INFO level under this script's WARNING-only root logger it never
    # reached stderr, making a genuine non-convergence indistinguishable
    # from any other silent skip. Includes the full step_trace so a
    # non-convergence is actually diagnosable — e.g. "kept repeating the
    # same search" vs "kept submitting ungrounded ids" vs "never called
    # SpeciesResolverOutput at all" are very different problems that a
    # bare "exhausted N steps" line can't distinguish between.
    logger.warning(
        "LLM species resolver exhausted %d steps for %r without a grounded submission. Trace:\n%s",
        max_steps,
        species_name,
        "\n".join(f"  {line}" for line in step_trace) or "  (no tool calls made at all)",
    )
    return None


async def _try_refseq_filter(species_term: str) -> tuple[str | None, str | None]:
    """Try to get the latest RefSeq assembly using an esearch filter.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism] AND latest_refseq[filter]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    uid = uid_list[0]

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})
    assembly_id = assembly_info.get("assemblyaccession", "")

    if assembly_id.startswith("GCF_"):
        return uid, assembly_id

    return None, None


async def _try_unfiltered_fallback(species_term: str) -> tuple[str | None, str | None]:
    """Fall back to unfiltered esearch, preferring GCF_ over GCA_.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 10,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    chosen_uid = None
    chosen_acc = None
    fallback_uid = None
    fallback_acc = None

    for uid in uid_list:
        info = results.get(uid, {})
        acc = info.get("assemblyaccession", "")
        if acc.startswith("GCF_") and chosen_uid is None:
            chosen_uid = uid
            chosen_acc = acc
            break
        elif acc.startswith("GCA_") and fallback_uid is None:
            fallback_uid = uid
            fallback_acc = acc

    target_uid = chosen_uid or fallback_uid
    target_acc = chosen_acc or fallback_acc

    return target_uid, target_acc


async def resolve_species(species_name: str) -> dict:
    """
    Deterministic fallback: search taxonomy first, then assemblies.
    Input: species_name (str)
    Output: dict matching SpeciesResolverOutput
            (assembly_id, scientific_name, common_name, confidence, reasoning)
    """
    key = species_name.strip()

    candidates = await _search_taxonomy_core(key)
    if candidates:
        tax_id = candidates[0].get("tax_id", "")
        scientific_name = candidates[0].get("scientific_name", "")
        common_name = candidates[0].get("common_name", "")
        assemblies = await _search_assembly_by_taxid_core(tax_id)
        if assemblies:
            assemblies.sort(key=lambda x: (not x["assembly_id"].startswith("GCF_"), x["assembly_id"]))
            chosen = assemblies[0]
            return {
                "assembly_id": chosen["assembly_id"],
                "scientific_name": scientific_name or chosen.get("scientific_name", ""),
                "common_name": common_name or chosen.get("common_name", ""),
                "confidence": 0.5,
                "reasoning": "Deterministic NCBI fallback used (no LLM)",
            }
        return {
            "assembly_id": None,
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": 0.0,
        }

    uid, assembly_id = await _try_refseq_filter(key)
    if uid is None:
        uid, assembly_id = await _try_unfiltered_fallback(key)

    if uid is None or assembly_id is None:
        return {
            "assembly_id": None,
            "scientific_name": None,
            "common_name": None,
            "confidence": 0.0,
        }

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})

    scientific_name = assembly_info.get("organism")
    common_name = assembly_info.get("organism")

    return {
        "assembly_id": assembly_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "confidence": 0.5,
        "reasoning": "Deterministic NCBI fallback used (no LLM)",
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Species Resolver live NCBI test ---")
        for species in ["tiger", "house mouse", "asian elephant", "dragon"]:
            result = await resolve_species(species)
            print(f"{species}: {result}")

    asyncio.run(_quick_test())