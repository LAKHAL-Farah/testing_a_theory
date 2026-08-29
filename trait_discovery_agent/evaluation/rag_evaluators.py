"""
Step 5 — the four RAGAS-style checks, implemented as plain comparison/overlap
functions (guide: "enough for a first pass" -- swap in the real `ragas`
library metric functions later if needed, keeping one function per metric so
scores stay reportable separately per node rather than one blended number).

Step 6 — payload schema contract check, wired in here via
`payload_contract()` so it shows up in the same scorecard rather than as a
separate ad-hoc test.

All four metrics operate on plain strings/lists so they work the same way
regardless of which node produced them -- capture.py is what's responsible
for turning each node's structured output into that flat shape.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from kb.retrieval import RetrievedDocument, ValidationResult, validate_document

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "via", "with", "gene", "protein",
}

# Below this relevance score a claim is flagged as off-topic in the detail
# string (the score itself is still the raw model output, not this threshold
# -- this only controls what gets called out for a human to look at).
_RELEVANCY_OFF_TOPIC_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# answer_relevancy scoring backend
#
# A first pass here used cosine similarity between a full question and each
# bare claim, embedded with the same all-MiniLM-L6-v2 model kb/retrieval.py
# uses for search. That flattened every collection into a 0.16-0.40 band
# even though faithfulness and context_precision both sat at ~1.00 -- i.e.
# the answers *were* correct, the metric just couldn't say so. Reformulating
# the query into several shorter, same-register anchors and max-pooling
# barely moved the number (confirmed against a live run), which rules out
# "wrong phrasing" as the cause: a bi-encoder trained for general sentence
# *similarity* just doesn't produce high absolute cosine scores for two
# genuinely-related but differently-worded short phrases -- there's no
# amount of query reformulation that fixes a ceiling built into the model.
#
# A cross-encoder trained specifically for query/passage *relevance*
# ranking (as opposed to paraphrase similarity) is the right tool for this
# job: it takes the (query, claim) pair together and outputs a single
# relevance judgment, which is what "is this claim relevant to the
# question" actually is. This only affects how the eval harness scores
# relevancy -- it doesn't touch kb/embeddings.py or require re-indexing
# Qdrant, since retrieval itself never used this model.
# ---------------------------------------------------------------------------

_cross_encoder = None


def _get_relevance_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _score_relevance_pairs(query: str, claims: list[str]) -> list[float]:
    """Raw (query, claim) relevance scores, sigmoid-mapped to [0, 1].

    Isolated into its own function (rather than inlined in answer_relevancy)
    so tests can monkeypatch this one call instead of needing the real
    ~90MB MS MARCO cross-encoder model.
    """
    encoder = _get_relevance_cross_encoder()
    raw_scores = encoder.predict([(query, claim) for claim in claims])
    return [max(0.0, min(1.0 / (1.0 + math.exp(-float(s))), 1.0)) for s in raw_scores]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _chunk_text(chunk: RetrievedDocument) -> str:
    """Flatten a retrieved document's payload into one searchable string."""
    payload = chunk.payload or {}
    parts = [
        str(payload.get(k, ""))
        for k in ("go_name", "pathway_name", "function_summary", "title", "short_summary", "reasoning")
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    score: float
    detail: str = ""


def faithfulness(answer_claims: list[str], context_chunks: list[RetrievedDocument]) -> MetricResult:
    """Does every claim in the generated answer trace back to something
    actually present in the retrieved context?

    First-pass implementation: for each claim, require meaningful token
    overlap with at least one retrieved chunk. A claim with no supporting
    chunk at all is the unfaithful (hallucinated) case this is meant to
    catch -- e.g. protein_data stating a function that isn't in any
    retrieved UniProt chunk.
    """
    if not answer_claims:
        return MetricResult(score=1.0, detail="no claims made, nothing to be unfaithful about")

    context_token_sets = [_tokenize(_chunk_text(c)) for c in context_chunks]
    unsupported: list[str] = []

    for claim in answer_claims:
        claim_tokens = _tokenize(claim)
        if not claim_tokens:
            continue
        supported = any(
            len(claim_tokens & ctx_tokens) / len(claim_tokens) >= 0.5
            for ctx_tokens in context_token_sets
        )
        if not supported:
            unsupported.append(claim)

    score = 1.0 - (len(unsupported) / len(answer_claims))
    detail = f"unsupported claims: {unsupported}" if unsupported else "all claims grounded in retrieved context"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 2 — Answer relevancy
# ---------------------------------------------------------------------------

async def answer_relevancy(answer_claims: list[str], gene: str, trait_name: str) -> MetricResult:
    """Does the generated answer actually address the gene/trait asked about,
    rather than being generic or off-topic?

    Cross-encoder implementation: scores each claim against a
    "What is the role of {gene} in {trait_name}?" query as a direct
    relevance judgment (see _score_relevance_pairs), not paraphrase
    similarity. Replaces two earlier attempts that both undersold correct
    answers -- pure token overlap scored a claim as *completely* off-topic
    (0.0) whenever it shared zero literal words with the query (e.g. "p53
    signaling pathway" vs trait_name "tumor suppression": a correct answer,
    near-zero literal overlap); a bi-encoder cosine-similarity version fixed
    that false-negative but flattened every collection into a 0.16-0.40
    band regardless of answer quality, because a model trained for sentence
    *similarity* doesn't produce high absolute scores for two genuinely
    related but differently-worded short phrases -- confirmed in practice
    across several query reformulations, so it's the scoring approach, not
    the phrasing, that was capping every collection.
    """
    if not answer_claims:
        return MetricResult(score=0.0, detail="no answer produced")

    query = f"What is the role of {gene} in {trait_name}?"
    # CrossEncoder.predict is a synchronous, CPU-bound call -- offload it so
    # this coroutine doesn't block the event loop the other capture_* calls
    # in run_eval.py share.
    scores = await asyncio.to_thread(_score_relevance_pairs, query, answer_claims)

    off_topic = [claim for claim, sim in zip(answer_claims, scores) if sim < _RELEVANCY_OFF_TOPIC_THRESHOLD]
    score = sum(scores) / len(scores)
    detail = f"off-topic claims: {off_topic}" if off_topic else "answer stays on-topic for gene/trait"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 3 — Context precision
# ---------------------------------------------------------------------------

def context_precision(context_chunks: list[RetrievedDocument], gene: str) -> MetricResult:
    """Of the chunks retrieved, how many were actually about the right gene?

    First-pass implementation: a chunk is "relevant" if its payload's
    gene_symbol matches the queried gene (case-insensitive exact match --
    this is the check that catches the ambiguous/synonym-symbol cases like
    HR vs HRAS or MARCH1 vs the calendar month).
    """
    if not context_chunks:
        return MetricResult(score=0.0, detail="nothing retrieved")

    relevant = [
        c for c in context_chunks
        if str(c.payload.get("gene_symbol", "")).upper() == gene.upper()
    ]
    score = len(relevant) / len(context_chunks)
    irrelevant_ids = [c.id for c in context_chunks if c not in relevant]
    detail = (
        f"{len(relevant)}/{len(context_chunks)} chunks match gene_symbol={gene!r}"
        + (f"; off-gene hits: {irrelevant_ids}" if irrelevant_ids else "")
    )
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Metric 4 — Context recall
# ---------------------------------------------------------------------------

def _term_surfaced(term: str, context_blob: str, context_tokens: set[str]) -> bool:
    """A term counts as recalled if it's a literal substring, OR if every
    content word in it shows up somewhere in the retrieved context.

    Live QuickGO/UniProt text legitimately varies in ways an exact substring
    check punishes for no good reason -- hyphenation ("G-protein coupled"
    vs "G protein-coupled"), word order, or one field spelling out what
    another abbreviates. The token fallback still requires every meaningful
    word in the expected term to be present (not just any one of them), so
    it doesn't turn into a trivially lenient check -- it only forgives
    surface formatting, not missing content.
    """
    normalized_term = term.lower().replace("-", " ")
    normalized_blob = context_blob.replace("-", " ")
    if normalized_term in normalized_blob:
        return True
    term_tokens = _tokenize(term)
    return bool(term_tokens) and term_tokens.issubset(context_tokens)


def context_recall(expected_terms: list[str], context_chunks: list[RetrievedDocument]) -> MetricResult:
    """Of everything relevant that exists in the KB for this gene, how much
    did retrieval actually surface? Compares Step 3's expected_* list
    against the retrieved chunks' text (case-insensitive substring, with a
    token-subset fallback for surface-level wording differences -- see
    _term_surfaced).
    """
    if not expected_terms:
        return MetricResult(score=1.0, detail="no expected terms recorded for this gene")

    context_blob = " ".join(_chunk_text(c) for c in context_chunks).lower()
    context_tokens = _tokenize(context_blob)
    found, missing = [], []
    for term in expected_terms:
        if _term_surfaced(term, context_blob, context_tokens):
            found.append(term)
        else:
            missing.append(term)

    score = len(found) / len(expected_terms)
    detail = f"missing: {missing}" if missing else "all expected terms surfaced"
    return MetricResult(score=score, detail=detail)


# ---------------------------------------------------------------------------
# Step 6 — payload schema contract check (architecture-specific, not a
# generic RAGAS metric, but reported in the same scorecard).
# ---------------------------------------------------------------------------

@dataclass
class ContractCheck:
    collection: str
    total_chunks: int
    valid_chunks: int
    failures: list[str] = field(default_factory=list)


def payload_contract(collection: str, context_chunks: list[RetrievedDocument]) -> ContractCheck:
    """Confirms every retrieved chunk satisfies kb/retrieval.py's
    REQUIRED_FIELDS contract for its collection -- Step 6.1. Any missing
    field here is exactly the "fail loudly on drift" signal the contract is
    designed to surface; Step 6.2 (deliberately malformed payload -> loud
    failure) belongs in a dedicated unit test against validate_document()
    directly (see tests/test_rag_evaluators_unit.py), not this scorecard
    pass, since it needs a synthetic bad payload rather than live data.
    """
    check = ContractCheck(collection=collection, total_chunks=len(context_chunks), valid_chunks=0)
    for chunk in context_chunks:
        result: ValidationResult = validate_document(collection, chunk.payload)
        if result.is_valid:
            check.valid_chunks += 1
        else:
            check.failures.append(f"id={chunk.id} missing={result.missing_fields}")
    return check
