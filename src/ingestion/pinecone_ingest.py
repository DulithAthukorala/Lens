from __future__ import annotations

"""Pinecone ingestion script — run once before going to production.

Reads all case study .md files, chunks them (section-aware strategy),
embeds each chunk with all-MiniLM-L6-v2, and upserts into Pinecone.

Usage:
    python -m src.ingestion.pinecone_ingest

Requirements in .env:
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=lens-case-studies   (must already exist in Pinecone console)
    PINECONE_INDEX_HOST=...                 (optional, use host URL for faster init)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from src.ingestion.chunking_strategies import chunk_by_sections
from src.ingestion.kb_builder import load_case_studies

load_dotenv()

_EMBED_MODEL = "all-MiniLM-L6-v2"
_BATCH_SIZE = 100


def ingest_to_pinecone() -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    index_host = os.environ.get("PINECONE_INDEX_HOST")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "lens-case-studies")

    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=index_host) if index_host else pc.Index(index_name)

    embedder = SentenceTransformer(_EMBED_MODEL)
    docs = load_case_studies()

    vectors = []
    for doc in docs:
        chunks = chunk_by_sections(doc["text"])
        for ch in chunks:
            chunk_id = f"{doc['file_name']}::chunk{ch.chunk_index}"
            embedding = embedder.encode(ch.text).tolist()
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": ch.text,
                    "file_name": doc["file_name"],
                    "chunk_index": ch.chunk_index,
                    "strategy": "sections",
                },
            })

    # Upsert in batches
    for i in range(0, len(vectors), _BATCH_SIZE):
        batch = vectors[i: i + _BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // _BATCH_SIZE + 1} ({len(batch)} vectors)")

    print(f"\nDone. {len(vectors)} chunks ingested into Pinecone index '{index_name}'.")


if __name__ == "__main__":
    ingest_to_pinecone()
