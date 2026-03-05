from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import tiktoken



_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+") # splits on .!? followed by whitespace


def _count_tokens(text: str) -> int:
    """
    to count token for a given text
    """
    enc = tiktoken.get_encoding("cl100k_base") # used by OpenAI models
    return len(enc.encode(text))



#
@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int


def chunk_fixed_tokens(text: str, chunk_tokens: int = 512, overlap_tokens: int = 64) -> List[Chunk]:
    """
    Strategy 1: fixed-size chunks with overlap(to keep some context between chunks).
    """
    text = text.strip()
    if not text:
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)

    chunks: List[Chunk] = [] # [Chunk(text="part1", chunk_index=0), ...]
    start = 0 # where the chunk starts
    idx = 0 # chunk index

    while start < len(toks):
        end = min(len(toks), start + chunk_tokens) # where the chunk ends
        chunk_text = enc.decode(toks[start:end])
        chunks.append(Chunk(text=chunk_text, chunk_index=idx))
        idx += 1
        start = end - overlap_tokens # move back by overlap for next chunk
    return chunks


def  chunk_by_sentences(text,target_tokens: int = 220,max_tokens: int = 320,
) -> List[Chunk]:
    """
    Strategy 2: sentence based chunking.
    - Split into sentences
    - Accumulate sentences until target_tokens (soft) and never exceed max_tokens (hard)
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()] # split into sentences and clean up whitespace
    chunks: List[Chunk] = []

    buf: List[str] = []
    buf_tokens = 0
    idx = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        # If a single sentence is huge, force it into its own chunk.
        if sent_tokens >= max_tokens:
            if buf:
                chunks.append(Chunk(text=" ".join(buf).strip(), chunk_index=idx))
                idx += 1
                buf, buf_tokens = [], 0
            chunks.append(Chunk(text=sent, chunk_index=idx))
            idx += 1
            continue

        # If adding this sentence would exceed hard limit, flush buffer first.
        if buf and (buf_tokens + sent_tokens) > max_tokens:
            chunks.append(Chunk(text=" ".join(buf).strip(), chunk_index=idx))
            idx += 1
            buf, buf_tokens = [], 0

        buf.append(sent)
        buf_tokens += sent_tokens

        # If we hit target, flush (soft boundary).
        if buf_tokens >= target_tokens:
            chunks.append(Chunk(text=" ".join(buf).strip(), chunk_index=idx))
            idx += 1
            buf, buf_tokens = [], 0

    if buf:
        chunks.append(Chunk(text=" ".join(buf).strip(), chunk_index=idx))

    return chunks



def chunk_by_sections(text: str) -> List[Chunk]:
    """
    Strategy 3: section-aware chunking.

    Splits the document based on known headers such as:
    Summary, Client, Industry, Problem, Solution, Result.

    Each section becomes its own chunk.
    """

    text = text.strip()
    if not text:
        return []

    # headers in the case studies
    headers = ["Summary:", "Client:", "Industry:", "Problem:", "Solution:", "Result:"]

    # split text while keeping headers
    pattern = r"(?=(Summary:|Client:|Industry:|Problem:|Solution:|Result:))"
    parts = re.split(pattern, text)

    chunks: List[Chunk] = []
    idx = 0

    # parts will look like: ['', 'Summary:', '...', 'Client:', '...', ...]
    i = 1
    while i < len(parts):
        header = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        chunk_text = f"{header}{content}".strip()

        if chunk_text:
            chunks.append(Chunk(text=chunk_text, chunk_index=idx))
            idx += 1

        i += 2

    return chunks