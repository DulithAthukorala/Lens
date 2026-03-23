from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from sentence_transformers import CrossEncoder

from src.models import FailureSignal

log = logging.getLogger(__name__)

CHROMA_PATH = Path("chroma_data")
PRIMARY_COLLECTION = "lens_case_studies_sections"    # section-aware: Problem/Solution/Result chunks
FALLBACK_COLLECTION = "lens_case_studies_sentences"  # fallback if primary returns low scores

# Cross-encoder reranker — loaded lazily to avoid cold-start on import.
# ms-marco-MiniLM-L-6-v2 is a lightweight cross-encoder (~23M params) trained on
# MS MARCO passage ranking. It scores (query, document) pairs directly, giving
# much higher precision than bi-encoder cosine similarity alone.
_reranker: Optional[CrossEncoder] = None
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_CANDIDATES = 10  # over-fetch from ChromaDB, then rerank down to top-k


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Loading cross-encoder reranker: %s", _RERANKER_MODEL)
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


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


def _rerank(query_text: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
    """Re-score candidates with a cross-encoder and return top-k.

    Two-stage retrieval:
      Stage 1 (bi-encoder): ChromaDB's all-MiniLM-L6-v2 does fast ANN retrieval
                             over the full collection. Cheap but noisy.
      Stage 2 (cross-encoder): ms-marco-MiniLM-L-6-v2 jointly encodes (query, doc)
                                for fine-grained relevance scoring. Expensive but precise.

    This is the standard retrieve-then-rerank pattern used in production search systems.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query_text, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    log.debug("Rerank scores: %s", [(r["id"], r["rerank_score"]) for r in reranked[:top_k]])
    return reranked[:top_k]


def retrieve_case_studies(query_text: str, n_results: int = 5) -> List[dict]:
    """Route to Pinecone (production) or ChromaDB (local) based on USE_PINECONE env flag."""
    if os.environ.get("USE_PINECONE", "false").lower() == "true":
        from src.rag.pinecone_pipeline import retrieve_case_studies as _pinecone_retrieve
        return _pinecone_retrieve(query_text, n_results)
    return _retrieve_chromadb(query_text, n_results)


def _retrieve_chromadb(query_text: str, n_results: int = 5) -> List[dict]:
    """Two-stage retrieval: ChromaDB bi-encoder recall → cross-encoder rerank.

    Stage 1: Over-fetch candidates from ChromaDB (bi-encoder ANN, fast but noisy).
    Stage 2: Rerank with cross-encoder for precise relevance scoring.

    Uses the sections collection as primary (best semantic precision for case study
    problem/solution matching). Falls back to sentences collection if primary
    returns uniformly low similarity scores (all < 0.3).

    Returns list of dicts: {id, text, metadata, similarity, rerank_score} sorted
    by rerank_score descending.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Stage 1: over-fetch from bi-encoder
    candidates = _query_collection(client, PRIMARY_COLLECTION, query_text, _RERANK_CANDIDATES)

    # Fallback: if no strong matches in sections, try sentences
    if candidates and max(r["similarity"] for r in candidates) < 0.3:
        fallback = _query_collection(client, FALLBACK_COLLECTION, query_text, _RERANK_CANDIDATES)
        if fallback and max(r["similarity"] for r in fallback) > max(r["similarity"] for r in candidates):
            candidates = fallback

    # Stage 2: cross-encoder rerank
    results = _rerank(query_text, candidates, top_k=n_results)

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
