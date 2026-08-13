"""
Pathways Agent — KEGG pathway selection with LLM-guided relevance ranking.

For each gene:
  1. List all KEGG pathways linked to the gene (not just the first).
  2. If 0 links   → malformed_ids (no LLM call).
  3. If 1 link    → straight through (no real decision needed).
  4. If >1 links  → LLM reads candidate names and picks the most
                    trait-relevant pathway, with deterministic fallback
                    on LLM failure.
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from schemas.inputs import PathwaysInput
from schemas.outputs import PathwayEntry, PathwaysOutput
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.kegg_client import fetch_pathway, KEGG_LINK_URL, KEGG_GET_URL
from workflows.llm import invoke_with_fallback

import httpx

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
#  KEGG helpers (structured functions, not bound as LLM tools)                #
# --------------------------------------------------------------------------- #

async def list_pathway_candidates(kegg_gene_id: str) -> list[dict]:
    """All pathways KEGG's link endpoint returns for this gene, not just the first."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(2):
            try:
                resp = await client.get(KEGG_LINK_URL.format(kegg_gene_id=kegg_gene_id))
                resp.raise_for_status()
                candidates = []
                for line in resp.text.strip().splitlines():
                    if "\t" in line:
                        _gene_part, path_part = line.split("\t", 1)
                        pathway_id = path_part.replace("path:", "")
                        candidates.append({"pathway_id": pathway_id})
                return candidates
            except httpx.TimeoutException:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise
    return []


async def fetch_pathway_name(pathway_id: str) -> str:
    """Resolve a KEGG pathway ID to its human-readable NAME field."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(2):
            try:
                resp = await client.get(KEGG_GET_URL.format(pathway_id=pathway_id))
                resp.raise_for_status()
                for text_line in resp.text.splitlines():
                    if text_line.startswith("NAME"):
                        return text_line.replace("NAME", "").strip()
                return ""
            except httpx.TimeoutException:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise
    return ""


# --------------------------------------------------------------------------- #
#  LLM prompt                                                                 #
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are the Pathways agent. Given a gene, a list of KEGG pathways, and the trait under investigation, select the single most trait-relevant pathway.

Rules:
- Prefer the pathway whose name plausibly relates to the trait.
- If none look trait-relevant, keep the FIRST one in the list and say so in your reasoning. Never fabricate a better-sounding match.
- Never report a pathway_id or pathway_name that wasn't in the candidate list.
- Return ONLY a single JSON object with exactly this shape (no markdown fences, no extra text):

{{"pathway_id":"ko04928","pathway_name":"Thermogenesis","reasoning":"Selected because thermogenesis directly explains cold adaptation."}}
"""


# --------------------------------------------------------------------------- #
#  Per-gene selection                                                         #
# --------------------------------------------------------------------------- #

async def _select_pathway_for_gene(
    gene: str, kegg_gene_id: str, trait_name: str
) -> PathwayEntry | None:
    """
    Single-shot LLM prompt with all candidates pre-fetched.
    Returns None on malformed (zero links) or LLM failure (caller falls back).
    """

    # --- Discover candidates --------------------------------------------------
    try:
        candidates = await list_pathway_candidates(kegg_gene_id)
    except Exception as exc:
        logger.warning("list_pathway_candidates failed for %s: %s", gene, exc)
        return None

    if not candidates:
        return None                     # malformed — zero links

    if len(candidates) == 1:
        pid = candidates[0]["pathway_id"]
        try:
            name = await fetch_pathway_name(pid)
        except Exception:
            name = pid
        return PathwayEntry(
            pathway_id=pid,
            pathway_name=name or pid,
            reasoning="Only one KEGG link available.",
        )

    # --- Pre-fetch all names for the prompt -----------------------------------
    candidate_lines = []
    for c in candidates:
        pid = c["pathway_id"]
        try:
            pname = await fetch_pathway_name(pid)
        except Exception:
            pname = "(name unavailable)"
        candidate_lines.append(f"{pid}: {pname}")

    # --- Single-shot LLM call -------------------------------------------------
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", (
            "Gene symbol: {gene}\n"
            "KEGG gene ID: {kegg_gene_id}\n"
            "Trait under investigation: {trait_name}\n\n"
            "Candidate pathways (in order returned by KEGG):\n{candidates}\n\n"
            "Select the most relevant pathway and return JSON only."
        )),
    ])

    try:
        response = await invoke_with_fallback(
            prompt,
            {
                "gene": gene,
                "kegg_gene_id": kegg_gene_id,
                "trait_name": trait_name,
                "candidates": "\n".join(f"  {i+1}. {line}" for i, line in enumerate(candidate_lines)),
            },
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("LLM invocation failed for %s: %s", gene, exc)
        return None

    # --- Parse JSON from LLM response -----------------------------------------
    content = response.content if hasattr(response, "content") else str(response)
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```json\s*", "", content)
        cleaned = re.sub(r"```\s*", "", cleaned)
        # Find JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        data = json.loads(cleaned[start:end + 1])

        # Validate against candidate list
        selected_id = data.get("pathway_id", "")
        selected_name = data.get("pathway_name", "")
        if selected_id not in {c["pathway_id"] for c in candidates}:
            logger.warning("LLM returned unknown pathway_id %s for %s", selected_id, gene)
            return None

        return PathwayEntry(
            pathway_id=selected_id,
            pathway_name=selected_name,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as parse_exc:
        logger.warning("Failed to parse LLM answer for %s: %s\nContent: %s", gene, parse_exc, content)
        return None


# --------------------------------------------------------------------------- #
#  Public agent entrypoint                                                    #
# --------------------------------------------------------------------------- #

async def pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways: list[PathwayEntry] = []
    malformed: list[str] = []

    for gene in input.gene_list:
        kegg_gene_id = input.context.get("kegg_gene_ids", {}).get(gene)
        if not kegg_gene_id:
            malformed.append(gene)
            continue

        # --- LLM-guided selection -------------------------------------------
        entry = await _select_pathway_for_gene(gene, kegg_gene_id, input.trait_name)

        # --- Deterministic fallback (spec §9) --------------------------------
        if entry is None:
            logger.info("Falling back to deterministic fetch_pathway for %s", gene)
            try:
                entry = await fetch_pathway(kegg_gene_id)
            except Exception as exc:
                logger.warning("Deterministic fallback also failed for %s: %s", gene, exc)
                entry = None

        if entry is None:
            malformed.append(gene)
            continue

        # --- Cache / dedup (same pattern as before) ------------------------
        dedup_key = f"kegg:{entry.pathway_id}:{gene}"
        cached = await get_cached("kegg_pathways", dedup_key)
        if cached:
            pathways.append(entry)
            continue

        await upsert_point(
            "kegg_pathways",
            dedup_key,
            text_to_embed=entry.pathway_name,
            payload={
                "gene_symbol": gene,
                "pathway_id": entry.pathway_id,
                "pathway_name": entry.pathway_name,
                "source": "KEGG REST API",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        pathways.append(entry)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)


# =========================================================================== #
#  Mock agent (offline / CI)                                                  #
# =========================================================================== #

_MOCK_KEGG_DB = {
    "UCP1": PathwayEntry(
        pathway_id="ko00071",
        pathway_name="Fatty acid degradation",
        reasoning="Mock: relevant to metabolic heat production.",
    ),
    "PRDM16": PathwayEntry(
        pathway_id="ko04928",
        pathway_name="Thermogenesis",
        reasoning="Mock: directly related to brown adipose tissue.",
    ),
    "FGF5": PathwayEntry(
        pathway_id="ko04010",
        pathway_name="MAPK signaling pathway",
        reasoning="Mock: growth factor signaling.",
    ),
}


async def mock_pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways, malformed = [], []
    for gene in input.gene_list:
        if gene in _MOCK_KEGG_DB:
            pathways.append(_MOCK_KEGG_DB[gene])
        else:
            malformed.append(gene)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)