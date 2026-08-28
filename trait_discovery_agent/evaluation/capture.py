"""
Step 4 — run retrieval and capture what actually came back.

One shared helper, `capture_node`, used for all four retrieval-facing nodes
(gene_mapper, pathways, protein_data, literature_support) instead of
duplicating the same "run node -> pull context -> pull answer" shape four
times, per the Sprint 4 guide.

Each node has a different "context" surface (see Step 1 of the guide):
  - gene_mapper       -> Qdrant `go_annotations`
  - pathways          -> Qdrant `kegg_pathways`
  - protein_data      -> Qdrant `uniprot_proteins`
  - literature_support -> cross-agent call to the Literature Agent (no
                           Qdrant collection -- kb/sources/literature_agent_client.py
                           explicitly never caches evidence content)

so `capture_node` takes a small per-node adapter describing how to get the
query text, which Qdrant collection (if any) to search, and how to turn the
subagent's structured output into a flat list of "claims" for the RAGAS-style
checks in rag_evaluators.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.retrieval import RetrievedDocument, semantic_search  # noqa: E402
from schemas.inputs import (  # noqa: E402
    GeneMapperInput,
    LiteratureSupportInput,
    PathwaysInput,
    ProteinDataInput,
)
from subagents.gene_mapper import gene_mapper_agent  # noqa: E402
from subagents.literature_support import literature_support_agent  # noqa: E402
from subagents.pathways.mock import mock_pathways_agent  # noqa: E402
from subagents.protein_data.mock import mock_protein_data_agent  # noqa: E402


@dataclass
class NodeCapture:
    """Everything Step 4 asks us to capture, for one gene at one node."""

    node: str
    gene: str
    collection: Optional[str]          # None for literature_support (no Qdrant surface)
    query_text: str
    context_chunks: list[RetrievedDocument] = field(default_factory=list)
    answer_claims: list[str] = field(default_factory=list)
    raw_output: Any = None
    error: Optional[str] = None


async def _capture_context(collection: Optional[str], query_text: str, top_k: int) -> list[RetrievedDocument]:
    if collection is None:
        return []
    return await semantic_search(collection, query_text, top_k=top_k)


async def capture_gene_mapper(gene: str, trait_name: str, species_name: str, top_k: int = 5) -> NodeCapture:
    query_text = f"{trait_name} {gene}"
    cap = NodeCapture(node="gene_mapper", gene=gene, collection="go_annotations", query_text=query_text)
    try:
        cap.context_chunks = await _capture_context("go_annotations", query_text, top_k)
        out = await gene_mapper_agent(GeneMapperInput(
            trait_name=trait_name, gene_list=[gene], species_name=species_name,
            instruction=f"Map GO annotation for {gene}", context={},
        ))
        cap.raw_output = out
        cap.answer_claims = [a.go_name for a in out.go_annotations if a.gene_symbol == gene]
    except Exception as exc:  # pragma: no cover - live-dependency failures surface as eval findings
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_pathways(gene: str, trait_name: str, top_k: int = 5) -> NodeCapture:
    query_text = f"{trait_name} {gene} pathway"
    cap = NodeCapture(node="pathways", gene=gene, collection="kegg_pathways", query_text=query_text)
    try:
        cap.context_chunks = await _capture_context("kegg_pathways", query_text, top_k)
        # Real pathways_agent (subagents/pathways/__init__.py) hits the live KEGG API +
        # LLM; the mock is used here so this stays runnable offline/in CI. Swap in the
        # real subagent for a live-KB run of this eval.
        out = await mock_pathways_agent(PathwaysInput(
            gene_list=[gene], trait_name=trait_name,
            instruction=f"Find pathway for {gene}", context={},
        ))
        cap.raw_output = out
        cap.answer_claims = [p.pathway_name for p in out.pathways]
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_protein_data(gene: str, trait_name: str, top_k: int = 5) -> NodeCapture:
    query_text = f"{trait_name} {gene} protein function"
    cap = NodeCapture(node="protein_data", gene=gene, collection="uniprot_proteins", query_text=query_text)
    try:
        cap.context_chunks = await _capture_context("uniprot_proteins", query_text, top_k)
        out = await mock_protein_data_agent(ProteinDataInput(
            gene_list=[gene], trait_name=trait_name,
            instruction=f"Find protein function for {gene}", context={},
        ))
        cap.raw_output = out
        cap.answer_claims = [p.function_summary for p in out.proteins]
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


async def capture_literature_support(gene: str, trait_name: str) -> NodeCapture:
    # No Qdrant collection for this node (see module docstring) -- "context" is
    # whatever the Literature Agent cross-agent call returned this run.
    cap = NodeCapture(node="literature_support", gene=gene, collection=None, query_text=trait_name)
    try:
        out = await literature_support_agent(LiteratureSupportInput(
            trait_name=trait_name, gene_list=[gene],
            instruction=f"Find literature support for {gene}", context={},
        ))
        cap.raw_output = out
        cap.context_chunks = [
            RetrievedDocument(id=r.pmid, score=1.0, payload={
                "pmid": r.pmid, "title": r.title, "year": r.year, "short_summary": r.short_summary,
            })
            for r in out.evidence
        ]
        cap.answer_claims = [r.short_summary for r in out.evidence]
    except Exception as exc:  # pragma: no cover
        cap.error = f"{type(exc).__name__}: {exc}"
    return cap


CAPTURE_FNS: dict[str, Callable[..., Awaitable[NodeCapture]]] = {
    "gene_mapper": capture_gene_mapper,
    "pathways": capture_pathways,
    "protein_data": capture_protein_data,
    "literature_support": capture_literature_support,
}
