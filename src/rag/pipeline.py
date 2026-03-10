from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import chromadb

from src.models import FailureSignal

CHROMA_PATH = Path("chroma_data")
PRIMARY_COLLECTION = "lens_case_studies_sections"    # section-aware: Problem/Solution/Result chunks
FALLBACK_COLLECTION = "lens_case_studies_sentences"  # fallback if primary returns low scores


def build_analyst_query(signals: List[FailureSignal], industry: Optional[str] = None) -> str:
    """Synthesize a natural-language paragraph from failure signals for semantic retrieval.

    A descriptive paragraph outperforms raw signal names because ChromaDB's embedding
    model matches against narrative case study text, not keyword lists.
    """
    signal_summary = ". ".join(
        f"{s.signal_name} ({s.evidence_value})" for s in signals
    )
    top_names = ", ".join(s.signal_name for s in signals[:3])
    industry_str = industry or "unknown industry"
    return (
        f"Business in {industry_str} facing these marketing failures: {signal_summary}. "
        f"Looking for case studies about improving {top_names}."
    )


def retrieve_case_studies(query_text: str, n_results: int = 5) -> List[dict]:
    """Query ChromaDB and return ranked case study chunks.

    Uses the sections collection as primary (best semantic precision for case study
    problem/solution matching). Falls back to sentences collection if primary
    returns uniformly low similarity scores (all < 0.3).

    Returns list of dicts: {id, text, metadata, similarity} sorted descending.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    results = _query_collection(client, PRIMARY_COLLECTION, query_text, n_results)

    # Fallback: if no strong matches in sections, try sentences
    if results and max(r["similarity"] for r in results) < 0.3:
        fallback = _query_collection(client, FALLBACK_COLLECTION, query_text, n_results)
        if fallback and max(r["similarity"] for r in fallback) > max(r["similarity"] for r in results):
            results = fallback

    return results


def _query_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    query_text: str,
    n_results: int,
) -> List[dict]:
    """Run a query against a single ChromaDB collection and normalize distances to similarity."""
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    raw = col.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        # ChromaDB default uses L2 distance for its built-in embedding function.
        # Convert to a 0–1 similarity: similarity = 1 / (1 + distance)
        # (safe formula that works for both L2 and cosine collections)
        similarity = 1.0 / (1.0 + dist)
        results.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "similarity": round(similarity, 4),
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)
