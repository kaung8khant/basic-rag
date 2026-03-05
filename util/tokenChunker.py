from __future__ import annotations

import re
import tiktoken
from typing import Any


def count_tokens(
    text: str,
) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 300,
    overlap_tokens: int = 0,
) -> list[str]:
    return [
        chunk["text"]
        for chunk in chunk_text_by_tokens_with_counts(
            text=text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )
    ]


def chunk_text_by_tokens_with_counts(
    text: str,
    max_tokens: int = 300,
    overlap_tokens: int = 0,
) -> list[dict[str, int | str]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[dict[str, int | str]] = []
    token_ids = encoding.encode(cleaned)
    if not token_ids:
        return []

    step = max_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        window_ids = token_ids[start : start + max_tokens]
        if not window_ids:
            break

        raw_text = encoding.decode(window_ids).strip()
        if not raw_text:
            continue

        # Prefer ending at sentence punctuation if it stays within max_tokens.
        punct_matches = list(re.finditer(r"[.!?](?:\s|$)", raw_text))
        chunk_text = raw_text
        for match in reversed(punct_matches):
            candidate = raw_text[: match.end()].strip()
            if candidate and len(encoding.encode(candidate)) <= max_tokens:
                chunk_text = candidate
                break

        token_count = len(encoding.encode(chunk_text))
        chunks.append({"text": chunk_text, "token_count": token_count})

        if start + max_tokens >= len(token_ids):
            break

    return chunks


def build_chunks_from_segments(
    segments: list[dict[str, Any]],
    *,
    chunk_size: int = 300,
    token_overlap: int = 0,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if token_overlap < 0:
        raise ValueError("token_overlap must be >= 0")

    all_chunks: list[dict[str, Any]] = []
    chunk_index = 1

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        doc_id = str(segment.get("doc_id", "unknown"))
        page = segment.get("page")
        segment_chunks = chunk_text_by_tokens_with_counts(
            text=text,
            max_tokens=chunk_size,
            overlap_tokens=token_overlap,
        )

        for chunk in segment_chunks:
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_size": chunk_size,
                    "chunk_isze": chunk_size,
                    "token_overlap": token_overlap,
                    "page_start": page,
                    "page_end": page,
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "chunk_id": f"{doc_id}:{chunk_index}",
                }
            )
            chunk_index += 1

    return all_chunks
