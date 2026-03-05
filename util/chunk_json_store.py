from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_vectorized_chunks_json(
    chunks: list[dict[str, Any]],
    *,
    path: str | Path,
) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def extract_vectors(chunks: list[dict[str, Any]]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for chunk in chunks:
        vector = chunk.get("vector")
        if vector:
            vectors.append(vector)
    return vectors


def attach_faiss_ids(
    chunks: list[dict[str, Any]],
    vector_ids: list[int],
) -> list[dict[str, Any]]:
    if len(chunks) != len(vector_ids):
        raise ValueError("vector_ids length must match chunks length")

    output: list[dict[str, Any]] = []
    for chunk, faiss_id in zip(chunks, vector_ids):
        output.append({**chunk, "faiss_id": faiss_id})
    return output
