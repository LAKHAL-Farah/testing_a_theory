from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.go_client import (
    list_go_candidates,
    resolve_go_term_name,
    fetch_go_annotation,
)
from kb.embeddings import embed_text
from workflows.llm import invoke_with_fallback

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1

_SYSTEM_PROMPT = (
    "You are the Gene Mapper agent. You're given a trait, a species, and a list of genes "
    "already confirmed relevant to that trait — your job is to attach the correct GO "
    "biological-process annotation to each one.\n\n"
    "You have a tool that returns ALL biological_process GO candidates QuickGO has for a gene. "
    "If there's more than one candidate:\n"
    "- Pick the one whose name is most plausibly related to the trait as described.\n"
    "- If none of the candidates look trait-relevant, still pick the closest one but say so "
    "plainly in your reasoning — do not invent a better-sounding GO term.\n"
    "- Never output a go_id or go_name that didn't come back from the tool.\n\n"
    "If a gene has no candidates, report it as unmatched."
)


async def _llm_pick_candidate(
    trait_name: str,
    gene_symbol: str,
    candidates: list[dict],
) -> tuple[str, str, str]:
    """
    Ask the LLM to pick the best candidate.
    Uses invoke_with_fallback so it rotates through all known NIM models.
    Returns (go_id, go_name, reasoning).
    """
    # Pre-resolve every candidate name so the LLM has full context in one shot
    for c in candidates:
        if c.get("go_name") is None:
            c["go_name"] = await resolve_go_term_name(c["go_id"]) or "unknown"

    candidates_text = "\n".join(
        f"- {c['go_id']}: {c['go_name']}" for c in candidates
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", (
            "Trait: {trait_name}\n"
            "Gene: {gene_symbol}\n\n"
            "Candidates returned by list_go_candidates:\n{candidates_text}\n\n"
            "Pick the single most relevant GO biological-process term for this gene "
            "given the trait. Return ONLY a JSON object with exactly these keys:\n"
            '- "go_id": the chosen GO id\n'
            '- "go_name": the chosen GO name\n'
            '- "reasoning": brief explanation of why you picked this one\n\n'
            "Do not invent a GO term that is not in the candidates list."
        )),
    ])

    response = await invoke_with_fallback(
        prompt,
        {
            "trait_name": trait_name,
            "gene_symbol": gene_symbol,
            "candidates_text": candidates_text,
        },
        temperature=0.1,
    )

    content = response.content

    # ---- extract JSON from response ----
    parsed: dict | None = None
    try:
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content
        parsed = json.loads(json_str.strip())
    except json.JSONDecodeError:
        logger.warning("LLM response for %s was not valid JSON: %s", gene_symbol, content)

    if parsed is None:
        raise RuntimeError("Could not parse LLM response")

    picked_id = parsed.get("go_id")
    picked_name = parsed.get("go_name")
    reasoning = parsed.get("reasoning", "")

    return picked_id, picked_name or "unknown", reasoning


async def _embedding_pick_candidate(trait_name: str, candidates: list[dict]) -> tuple[str, str]:
    """
    When LLM is unavailable, use embedding cosine similarity to pick the closest candidate.
    Vectors are already normalized by embed_text(), so dot product == cosine similarity.
    """
    trait_vec = np.array(await embed_text(trait_name))

    best_id, best_name, best_score = None, None, -1.0

    for c in candidates:
        name = c.get("go_name") or await resolve_go_term_name(c["go_id"]) or "unknown"
        if name == "unknown":
            continue
        cand_vec = np.array(await embed_text(name))
        score = float(np.dot(trait_vec, cand_vec))
        if score > best_score:
            best_score = score
            best_id = c["go_id"]
            best_name = name

    if best_id is None:
        raise RuntimeError("Embedding fallback found no valid candidates")

    logger.info(
        "Embedding fallback picked %s (%s) for trait '%s' (score=%.3f)",
        best_id, best_name, trait_name, best_score,
    )
    return best_id, best_name


async def gene_mapper_agent(input: GeneMapperInput) -> GeneMapperOutput:
    annotations: list[GOAnnotation] = []
    unmatched: list[str] = []

    for gene in input.gene_list:
        uniprot_accession = input.context.get("uniprot_accessions", {}).get(gene)
        if not uniprot_accession:
            unmatched.append(gene)
            continue

        # ---- fetch candidates from QuickGO (with one retry) ----
        candidates: list[dict] = []
        try:
            candidates = await list_go_candidates(gene, uniprot_accession)
        except Exception as exc:
            logger.warning("QuickGO error for %s, retrying once: %s", gene, exc)
            try:
                candidates = await list_go_candidates(gene, uniprot_accession)
            except Exception as exc2:
                logger.error("QuickGO failed for %s after retry: %s", gene, exc2)
                unmatched.append(gene)
                continue

        if not candidates:
            unmatched.append(gene)
            continue

        # ---- single candidate: straight through, no LLM (§8) ----
        if len(candidates) == 1:
            go_id = candidates[0]["go_id"]
            go_name = candidates[0].get("go_name") or await resolve_go_term_name(go_id) or "unknown"
            if go_name == "unknown":
                unmatched.append(gene)
                continue
            entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)

        # ---- multi-candidate: LLM pick (§8) ----
        else:
            try:
                go_id, go_name, reasoning = await _llm_pick_candidate(
                    input.trait_name, gene, candidates
                )
                # ---- grounding rule (§0.1): validate against actual tool output ----
                valid_ids = {c["go_id"] for c in candidates}
                if go_id not in valid_ids:
                    raise RuntimeError(
                        f"LLM picked invalid go_id {go_id} not in {valid_ids}"
                    )
                logger.info("LLM picked %s (%s) for %s: %s", go_id, go_name, gene, reasoning)
                entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)
            except Exception as exc:
                # ---- Tier 2: LLM failed → embedding similarity fallback ----
                logger.warning("LLM pick failed for %s, trying embedding fallback: %s", gene, exc)
                try:
                    go_id, go_name = await _embedding_pick_candidate(input.trait_name, candidates)
                    entry = GOAnnotation(gene_symbol=gene, go_id=go_id, go_name=go_name)
                except Exception as exc2:
                    # ---- Tier 3: everything failed → deterministic first candidate (§9) ----
                    logger.warning(
                        "Embedding fallback also failed for %s, falling back to first candidate: %s",
                        gene, exc2,
                    )
                    fallback = await fetch_go_annotation(gene, uniprot_accession)
                    if fallback is None:
                        unmatched.append(gene)
                        continue
                    entry = fallback

        # ---- cache layer (§6) ----
        dedup_key = f"go:{entry.go_id}:{entry.gene_symbol}"
        cached = await get_cached("go_annotations", dedup_key)
        if cached:
            annotations.append(entry)
            continue

        await upsert_point(
            "go_annotations",
            dedup_key,
            text_to_embed=entry.go_name,
            payload={
                "gene_symbol": entry.gene_symbol,
                "go_id": entry.go_id,
                "go_name": entry.go_name,
                "source": "GO REST API (QuickGO)",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        annotations.append(entry)

    # §9: FAILED if no annotations resolved OR any gene is unmatched
    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(
        status=status,
        go_annotations=annotations,
        unmatched_genes=unmatched,
    )


# --------------------------------------------------------------------------- #
#  Mock kept for offline / CI tests (unchanged contract)
# --------------------------------------------------------------------------- #
_MOCK_GO_DB = {
    "FGF5": GOAnnotation(gene_symbol="FGF5", go_id="GO:0031069", go_name="hair follicle development"),
    "KRT71": GOAnnotation(gene_symbol="KRT71", go_id="GO:0031069", go_name="hair follicle development"),
    "HR": GOAnnotation(gene_symbol="HR", go_id="GO:0042633", go_name="hair cycle"),
    "TRPV3": GOAnnotation(gene_symbol="TRPV3", go_id="GO:0050977", go_name="sensory perception of touch"),
    "UCP1": GOAnnotation(gene_symbol="UCP1", go_id="GO:0009408", go_name="response to heat"),
}

async def mock_gene_mapper(input: GeneMapperInput) -> GeneMapperOutput:
    annotations, unmatched = [], []
    for gene in input.gene_list:
        if gene in _MOCK_GO_DB:
            annotations.append(_MOCK_GO_DB[gene])
        else:
            unmatched.append(gene)

    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(status=status, go_annotations=annotations, unmatched_genes=unmatched)