from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import chromadb

from src.ingestion.chunking_strategies import (
    chunk_fixed_tokens,
    chunk_by_sentences,
    chunk_by_sections,
)

DATA_DIR = Path("src/ingestion/data")
CHROMA_PATH = Path("chroma_data")


def load_case_studies() -> List[Dict]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {DATA_DIR.resolve()}")

    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {DATA_DIR.resolve()}")

    docs: List[Dict] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8").strip()
        docs.append(
            {
                "file_name": fp.name,
                "source_path": str(fp.as_posix()),
                "text": text,
            }
        )
    return docs


def ingest(strategy: str) -> None:
    """
    Build a Chroma collection for a given chunking strategy.

    Strategies:
      - fixed512: fixed token chunks (512 tokens, 64 overlap)
      - sentences: sentence-based chunking (target/max token bounds)
      - sections: header/section-aware chunking (Summary/Problem/Solution/Result...)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_name = f"lens_case_studies_{strategy}"
    col = client.get_or_create_collection(collection_name)

    docs = load_case_studies()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    for doc in docs:
        raw = doc["text"]
        file_name = doc["file_name"]

        if strategy == "fixed512":
            chunks = chunk_fixed_tokens(raw, chunk_tokens=512, overlap_tokens=64)
        elif strategy == "sentences":
            chunks = chunk_by_sentences(raw, target_tokens=220, max_tokens=320)
        elif strategy == "sections":
            chunks = chunk_by_sections(raw)
        else:
            raise ValueError("Unknown strategy. Use: fixed512 | sentences | sections")

        for ch in chunks:
            chunk_id = f"{file_name}::chunk{ch.chunk_index}"
            ids.append(chunk_id)
            documents.append(ch.text)
            metadatas.append(
                {
                    "file_name": file_name,
                    "chunk_index": ch.chunk_index,
                    "strategy": strategy,
                }
            )

    # Upsert-like behavior: delete existing ids first (simple & safe)
    try:
        col.delete(ids=ids)
    except Exception:
        pass

    col.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingested {len(docs)} files into '{collection_name}'")
    print(f"Total chunks stored: {len(ids)}")


if __name__ == "__main__":
    ingest("fixed512")
    ingest("sentences")
    ingest("sections")