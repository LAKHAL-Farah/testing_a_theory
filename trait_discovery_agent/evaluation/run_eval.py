"""
Step 7 — run the RAG eval over rag_test_cases.yaml and produce a retrieval
scorecard broken down by collection (GO / KEGG / UniProt / literature),
not one blended number.

Usage:
    python evaluation/run_eval.py                 # all genes
    python evaluation/run_eval.py --gene TP53      # single gene, verbose
    python evaluation/run_eval.py --json results/latest.json

Requires QDRANT_URL / QDRANT_API_KEY (and whatever the real gene_mapper
LLM/QuickGO path needs) to be configured for a live-KB run -- pathways and
protein_data use the mock subagents by default so this stays runnable
offline (see capture.py docstring for how to swap in the real ones).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import CAPTURE_FNS, NodeCapture  # noqa: E402
from rag_evaluators import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    payload_contract,
)

TEST_CASES_PATH = Path(__file__).parent / "rag_test_cases.yaml"
NODE_TO_COLLECTION = {
    "gene_mapper": "go_annotations",
    "pathways": "kegg_pathways",
    "protein_data": "uniprot_proteins",
    "literature_support": "literature (no Qdrant collection)",
}
NODE_TO_EXPECTED_KEY = {
    "gene_mapper": "expected_go_terms",
    "pathways": "expected_pathways",
    "protein_data": "expected_protein_summary_contains",
}


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH) as f:
        return yaml.safe_load(f)


async def run_one_node(node: str, case: dict) -> dict[str, Any]:
    gene = case["gene"]
    trait_name = case["trait_name"]
    species_name = case.get("species_name", "")

    capture_fn = CAPTURE_FNS[node]
    if node == "gene_mapper":
        cap: NodeCapture = await capture_fn(gene, trait_name, species_name)
    elif node == "literature_support":
        cap = await capture_fn(gene, trait_name)
    else:
        cap = await capture_fn(gene, trait_name)

    result: dict[str, Any] = {
        "gene": gene, "node": node, "collection": cap.collection,
        "error": cap.error, "num_chunks": len(cap.context_chunks),
    }

    if cap.error:
        return result

    result["faithfulness"] = asdict(faithfulness(cap.answer_claims, cap.context_chunks))
    result["answer_relevancy"] = asdict(answer_relevancy(cap.answer_claims, gene, trait_name))
    if node in NODE_TO_EXPECTED_KEY:
        expected_terms = case.get(NODE_TO_EXPECTED_KEY[node]) or []
        result["context_recall"] = asdict(context_recall(expected_terms, cap.context_chunks))
    if cap.collection:
        result["context_precision"] = asdict(context_precision(cap.context_chunks, gene))
        contract = payload_contract(cap.collection, cap.context_chunks)
        result["payload_contract"] = {
            "total_chunks": contract.total_chunks,
            "valid_chunks": contract.valid_chunks,
            "failures": contract.failures,
        }
    return result


async def run_eval(genes: list[str] | None = None) -> list[dict[str, Any]]:
    cases = load_test_cases()
    if genes:
        wanted = {g.upper() for g in genes}
        cases = [c for c in cases if c["gene"].upper() in wanted]

    results = []
    for case in cases:
        for node in ("gene_mapper", "pathways", "protein_data", "literature_support"):
            results.append(await run_one_node(node, case))
    return results


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_scorecard(results: list[dict[str, Any]]) -> None:
    by_collection: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_collection[r["collection"] or "literature (no Qdrant collection)"].append(r)

    print("\n=== Retrieval scorecard, by collection ===\n")
    for collection, rows in sorted(by_collection.items(), key=lambda kv: str(kv[0])):
        errored = [r for r in rows if r["error"]]
        ok = [r for r in rows if not r["error"]]
        print(f"[{collection}] — {len(rows)} gene(s) evaluated, {len(errored)} errored")
        if ok:
            for metric in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
                scores = [r[metric]["score"] for r in ok if metric in r]
                if scores:
                    print(f"  {metric:<18} avg={_avg(scores):.2f}  (n={len(scores)})")
            contract_rows = [r for r in ok if "payload_contract" in r]
            if contract_rows:
                total = sum(r["payload_contract"]["total_chunks"] for r in contract_rows)
                valid = sum(r["payload_contract"]["valid_chunks"] for r in contract_rows)
                print(f"  {'payload_contract':<18} {valid}/{total} chunks valid")
        for r in errored:
            print(f"  ERROR gene={r['gene']} node={r['node']}: {r['error']}")
        print()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trait_discovery_agent RAG eval (Sprint 4 guide).")
    parser.add_argument("--gene", action="append", help="Restrict to one gene (repeatable).")
    parser.add_argument("--json", type=Path, default=None, help="Write raw results to this path.")
    args = parser.parse_args()

    results = await run_eval(genes=args.gene)
    print_scorecard(results)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"raw results written to {args.json}")

    return 1 if any(r["error"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
