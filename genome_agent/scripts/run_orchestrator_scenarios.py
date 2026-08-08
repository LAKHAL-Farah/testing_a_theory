"""
Scenario runner for the Genome Agent LangGraph orchestrator.

Unlike the pytest suite, this is a plain executable script: it builds the
*real* compiled graph (`build_genome_graph`) via `GenomeAgentLangGraphOrchestrator`
and runs it end to end against a handful of scenarios, printing a readable
trace of what happened at each step. Nothing is mocked at the node/routing
level — the only thing that varies run to run is whether NVIDIA_API_KEY is
set:

    - If it IS set, query_router / capability_resolver / explanation_writer
      call the real LLM.
    - If it is NOT set, each of those three modules catches the failure
      internally and falls back to its deterministic keyword-based logic
      (route_query_fallback / resolve_capability_fallback / raw findings
      text) — so the workflow still runs end to end without an API key.

Run it directly:
    python -m genome_agent.scripts.run_orchestrator_scenarios

or, from inside the container:
    python scripts/run_orchestrator_scenarios.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Node-level INFO logs are useful in isolation but drown out the trace below.
logging.getLogger().setLevel(logging.WARNING)

_WIDTH = 64


def _print_case(title: str, input_lines: list[str], workflow_lines: list[str], result) -> None:
    print("\n" + "=" * _WIDTH)
    print(f"CASE: {title}")
    print("=" * _WIDTH)
    print("Input:")
    for line in input_lines:
        print(f"  {line}")
    print("Workflow:")
    for line in workflow_lines:
        print(f"  {line}")
    print("Result:")
    print(f"  Species      : {result.species}")
    print(f"  Metadata     : {result.metadata}")
    print(f"  Annotation   : {result.annotation}")
    print(f"  Visualization: {result.visualization}")
    print(f"  Explanation  : {result.explanation}")
    print(f"  Errors       : {result.errors}")
    print("=" * _WIDTH)


async def run_all() -> None:
    # Imported here so the module can also be run as a script from either
    # the repo root (`python -m genome_agent.scripts.run_orchestrator_scenarios`)
    # or from inside the package directory (`python scripts/run_orchestrator_scenarios.py`).
    try:
        from ..orchestrator import GenomeAgentLangGraphOrchestrator
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from genome_agent.orchestrator import GenomeAgentLangGraphOrchestrator

    llm_mode = "LIVE (NVIDIA_API_KEY set)" if os.getenv("NVIDIA_API_KEY") else "FALLBACK (no NVIDIA_API_KEY — keyword-based routing/resolution)"
    print("\n" + "#" * _WIDTH)
    print(f"# Genome Agent — Orchestrator Scenario Run")
    print(f"# LLM mode: {llm_mode}")
    print("#" * _WIDTH)

    orch = GenomeAgentLangGraphOrchestrator()

    # ------------------------------------------------------------------
    # Case 1 — Happy path: sequential -> parallel -> sequential
    # ------------------------------------------------------------------
    result = await orch.run(
        user_question="Show me the genome size and genes of the tiger.",
        species_name="tiger",
        visualization_scope="chromosome_map",
    )
    _print_case(
        "Happy path — tiger, chromosome_map",
        ["Species: tiger", "Visualization scope: chromosome_map"],
        [
            "query_router -> species_resolver",
            "species_resolver -> parallel_kickoff (assembly_id resolved)",
            "parallel_kickoff -> get_genome_metadata + get_gene_annotation (parallel)",
            "join_parallel -> generate_visualization -> explanation_writer -> END",
        ],
        result,
    )

    # ------------------------------------------------------------------
    # Case 2 — Unknown species: hard stop via error_end
    # ------------------------------------------------------------------
    result = await orch.run(
        user_question="Show me the genome of the dragon.",
        species_name="dragon",
    )
    _print_case(
        "Unknown species — dragon (error_end routing)",
        ["Species: dragon"],
        [
            "query_router -> species_resolver",
            "species_resolver -> error_end (assembly_id is None, no downstream calls)",
            "error_end -> END",
        ],
        result,
    )

    # ------------------------------------------------------------------
    # Case 3 — protein_structure: NEEDS_AGENT -> capability_resolver
    # ------------------------------------------------------------------
    result = await orch.run(
        user_question="Predict the 3D protein structure for the house mouse.",
        species_name="house mouse",
        visualization_scope="protein_structure",
    )
    _print_case(
        "Escalation — house mouse, protein_structure (NEEDS_AGENT)",
        ["Species: house mouse", "Visualization scope: protein_structure"],
        [
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> generate_visualization (returns NEEDS_AGENT)",
            "generate_visualization -> capability_resolver -> explanation_writer -> END",
        ],
        result,
    )

    # ------------------------------------------------------------------
    # Case 4 — unknown visualization scope: FAILED, degrades gracefully
    # ------------------------------------------------------------------
    result = await orch.run(
        user_question="Show me the genome size and genes of the tiger.",
        species_name="tiger",
        visualization_scope="unknown_scope",
    )
    _print_case(
        "Partial failure — tiger, unknown_scope (visualization FAILED)",
        ["Species: tiger", "Visualization scope: unknown_scope"],
        [
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata + get_gene_annotation -> join_parallel",
            "join_parallel -> generate_visualization (returns FAILED, not escalated)",
            "generate_visualization -> explanation_writer -> END",
        ],
        result,
    )

    # ------------------------------------------------------------------
    # Case 5 — partial results: metadata present, annotation empty
    # ------------------------------------------------------------------
    result = await orch.run(
        user_question="Show me the genome size and genes of the asian elephant.",
        species_name="asian elephant",
        visualization_scope="chromosome_map",
    )
    _print_case(
        "Partial results — asian elephant (empty gene_list, metadata OK)",
        ["Species: asian elephant", "Visualization scope: chromosome_map"],
        [
            "query_router -> species_resolver -> parallel_kickoff",
            "get_genome_metadata (OK) + get_gene_annotation (empty gene_list, non-fatal) -> join_parallel",
            "join_parallel -> generate_visualization -> explanation_writer -> END",
        ],
        result,
    )

    print("\n" + "#" * _WIDTH)
    print("# All scenarios executed.")
    print("#" * _WIDTH)


if __name__ == "__main__":
    asyncio.run(run_all())