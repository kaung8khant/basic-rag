from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np


def store_vectors_in_faiss(
    vectors: list[list[float]],
    *,
    vector_store_dir: str | Path = "vector_store",
) -> dict[str, Any]:
    if not vectors:
        return {"stored_count": 0, "total_count": 0}

    store_dir = Path(vector_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "index.faiss"

    matrix = np.asarray(vectors, dtype="float32")
    dim = int(matrix.shape[1])

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != dim:
            raise RuntimeError(
                f"FAISS dimension mismatch. existing={index.d}, incoming={dim}"
            )
    else:
        index = faiss.IndexFlatIP(dim)

    start_id = int(index.ntotal)
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    vector_ids = list(range(start_id, start_id + len(vectors)))

    return {
        "stored_count": len(vectors),
        "total_count": int(index.ntotal),
        "index_path": str(index_path),
        "vector_ids": vector_ids,
    }


def search_vectors_in_faiss(
    query_vector: list[float],
    *,
    top_k: int = 5,
    vector_store_dir: str | Path = "vector_store",
) -> list[dict[str, float | int]]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not query_vector:
        return []

    index_path = Path(vector_store_dir) / "index.faiss"
    if not index_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    query = np.asarray([query_vector], dtype="float32")
    scores, ids = index.search(query, top_k)

    hits: list[dict[str, float | int]] = []
    for score, vector_id in zip(scores[0], ids[0]):
        if int(vector_id) < 0:
            continue
        hits.append({"faiss_id": int(vector_id), "score": float(score)})
    return hits
