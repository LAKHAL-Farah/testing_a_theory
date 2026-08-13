import httpx
from schemas.outputs import GOAnnotation

QUICKGO_SEARCH_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
QUICKGO_TERMS_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"


async def list_go_candidates(gene_symbol: str, uniprot_accession: str) -> list[dict]:
    """Return every biological_process GO annotation QuickGO has for this accession."""
    params = {
        "geneProductId": f"UniProtKB:{uniprot_accession}",
        "aspect": "biological_process",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(QUICKGO_SEARCH_URL, params=params)
        resp.raise_for_status()
        raw = resp.json().get("results", [])

        # Deduplicate by goId while preserving QuickGO order
        seen = set()
        candidates = []
        for r in raw:
            go_id = r.get("goId")
            if go_id and go_id not in seen:
                seen.add(go_id)
                candidates.append({"go_id": go_id, "go_name": None})
        return candidates


async def resolve_go_term_name(go_id: str) -> str | None:
    """Resolve a GO id to its human-readable name via the QuickGO ontology endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        term_resp = await client.get(
            QUICKGO_TERMS_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        term_resp.raise_for_status()
        term_results = term_resp.json().get("results", [])
        if not term_results or not term_results[0].get("name"):
            return None
        return term_results[0]["name"]


async def fetch_go_annotation(gene_symbol: str, uniprot_accession: str) -> GOAnnotation | None:
    """Deterministic fallback: take the first candidate, no ranking (§9)."""
    params = {
        "geneProductId": f"UniProtKB:{uniprot_accession}",
        "aspect": "biological_process",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(QUICKGO_SEARCH_URL, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        go_id = results[0]["goId"]

        term_resp = await client.get(
            QUICKGO_TERMS_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        term_resp.raise_for_status()
        term_results = term_resp.json().get("results", [])

        if not term_results or not term_results[0].get("name"):
            return None

        return GOAnnotation(
            gene_symbol=gene_symbol,
            go_id=go_id,
            go_name=term_results[0]["name"],
        )