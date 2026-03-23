from __future__ import annotations

"""Pinecone retrieval pipeline — production vector store.

Drop-in replacement for the ChromaDB pipeline. Set USE_PINECONE=true in .env
to route retrieve_case_studies() through this module instead of ChromaDB.

Ingestion: run src/ingestion/pinecone_ingest.py once to push case study
chunks into the Pinecone index before querying.
"""

import logging
import os
from typing import List

from pinecone import Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer

log = logging.getLogger(__name__)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_CANDIDATES = 10

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_index = None  # Pinecone Index object


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


def _get_index():
    global _index
    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_host = os.environ.get("PINECONE_INDEX_HOST")
        index_name = os.environ.get("PINECONE_INDEX_NAME", "lens-case-studies")

        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set in environment.")

        pc = Pinecone(api_key=api_key)
        _index = pc.Index(host=index_host) if index_host else pc.Index(index_name)
        log.info("Connected to Pinecone index: %s", index_name)
    return _index


def retrieve_case_studies(query_text: str, n_results: int = 5) -> List[dict]:
    """Two-stage retrieval: Pinecone bi-encoder recall → cross-encoder rerank.

    Mirrors the ChromaDB pipeline interface exactly — returns the same
    list-of-dicts format so the Analyst agent needs no changes.
    """
    index = _get_index()
    embedder = _get_embedder()

    # Stage 1: embed query and fetch top candidates from Pinecone
    query_vector = embedder.encode(query_text).tolist()
    response = index.query(
        vector=query_vector,
        top_k=_RERANK_CANDIDATES,
        include_metadata=True,
    )

    candidates = []
    for match in response.matches:
        candidates.append({
            "id": match.id,
            "text": match.metadata.get("text", ""),
            "metadata": match.metadata,
            "similarity": round(float(match.score), 4),
        })

    if not candidates:
        return []

    # Stage 2: cross-encoder rerank
    reranker = _get_reranker()
    pairs = [(query_text, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:n_results]
