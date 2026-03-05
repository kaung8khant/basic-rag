from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from util.chunk_json_store import (
    attach_faiss_ids,
    extract_vectors,
    save_vectorized_chunks_json,
)
from util.faiss_store import search_vectors_in_faiss, store_vectors_in_faiss


def store_vectors_and_attach_faiss_ids(
    vectorized_chunks: list[dict[str, Any]],
    *,
    vector_store_dir: Path,
) -> list[dict[str, Any]]:
    vectors = extract_vectors(vectorized_chunks)
    faiss_result = store_vectors_in_faiss(vectors, vector_store_dir=vector_store_dir)
    vector_ids = faiss_result.get("vector_ids", [])
    chunks_with_faiss_ids = attach_faiss_ids(vectorized_chunks, vector_ids)
    print(
        f"Stored vectors in FAISS: stored={faiss_result.get('stored_count', 0)} "
        f"total={faiss_result.get('total_count', 0)} index={faiss_result.get('index_path')}"
    )
    return chunks_with_faiss_ids


def store_chunks_json(
    chunks_with_faiss_ids: list[dict[str, Any]],
    *,
    doc_id: str,
    chunk_store_dir: Path,
) -> str:
    output_path = chunk_store_dir / f"{doc_id}.json"
    return save_vectorized_chunks_json(chunks_with_faiss_ids, path=output_path)


def load_chunk_map_by_faiss_id(chunk_store_dir: Path) -> dict[int, dict[str, Any]]:
    chunk_map: dict[int, dict[str, Any]] = {}
    if not chunk_store_dir.exists():
        return chunk_map

    for path in chunk_store_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            faiss_id = item.get("faiss_id")
            if not isinstance(faiss_id, int):
                continue
            chunk_map[faiss_id] = {k: v for k, v in item.items() if k != "vector"}
    return chunk_map


def retrieve_ranked_matches(
    query_vector: list[float],
    *,
    top_k: int,
    min_score: float,
    vector_store_dir: Path,
    chunk_store_dir: Path,
) -> list[dict[str, Any]]:
    hits = search_vectors_in_faiss(
        query_vector, top_k=top_k, vector_store_dir=vector_store_dir
    )
    chunk_map = load_chunk_map_by_faiss_id(chunk_store_dir)

    matches: list[dict[str, Any]] = []
    for hit in hits:
        faiss_id = int(hit["faiss_id"])
        score = float(hit["score"])
        if score < min_score:
            continue
        chunk = chunk_map.get(faiss_id)
        if not chunk:
            continue
        matches.append({"faiss_id": faiss_id, "score": score, "chunk": chunk})

    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    ranked: list[dict[str, Any]] = []
    for idx, item in enumerate(matches[:top_k], start=1):
        ranked.append({"k": idx, **item})
    return ranked
