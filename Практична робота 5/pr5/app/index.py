import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .documents import Chunk


INDEX_DIR = Path(__file__).parent.parent / "index"

DEFAULT_TOP_K = int(
    os.getenv("SEARCH_TOP_K", "5")
)

_threshold_raw = os.getenv(
    "SIMILARITY_THRESHOLD",
    "",
).strip()

SIMILARITY_THRESHOLD = (
    float(_threshold_raw)
    if _threshold_raw
    else None
)

ALLOWED_FILTERS = {
    "category",
    "product",
    "audience",
    "status",
}


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict

    def __len__(self) -> int:
        return len(self.chunks)


def build(
    chunks: list[Chunk],
    vectors: np.ndarray,
    model_name: str,
    extra: dict | None = None,
) -> SearchIndex:
    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Вектори індексу повинні бути двовимірним масивом."
        )

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            "Кількість chunks не збігається "
            "з кількістю векторів."
        )

    if len(chunks) == 0:
        raise ValueError(
            "Неможливо побудувати порожній індекс."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Індекс містить нульовий вектор."
        )

    vectors = vectors / norms

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=model_name,
        extra=extra or {},
    )


def save(index: SearchIndex) -> None:
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        INDEX_DIR / "vectors.npy",
        index.vectors,
    )

    chunks_data = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "metadata": chunk.metadata,
        }
        for chunk in index.chunks
    ]

    data = {
        "model_name": index.model_name,
        "dimension": int(
            index.vectors.shape[1]
        ),
        "count": len(index.chunks),
        "chunks": chunks_data,
        "extra": index.extra,
    }

    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load() -> SearchIndex:
    vectors_path = INDEX_DIR / "vectors.npy"
    chunks_path = INDEX_DIR / "chunks.json"

    if (
        not vectors_path.exists()
        or not chunks_path.exists()
    ):
        raise FileNotFoundError(
            "Індекс не знайдено. "
            "Спочатку запустіть: python ingest.py"
        )

    vectors = np.load(
        vectors_path,
        allow_pickle=False,
    )

    data = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    chunks = [
        Chunk(
            text=item["text"],
            source=item["source"],
            metadata=item.get(
                "metadata",
                {},
            ),
        )
        for item in data["chunks"]
    ]

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            "Збережені chunks і вектори "
            "мають різну кількість елементів."
        )

    return SearchIndex(
        chunks=chunks,
        vectors=np.asarray(
            vectors,
            dtype=np.float32,
        ),
        model_name=data["model_name"],
        extra=data.get("extra", {}),
    )


def _matches_filters(
    chunk: Chunk,
    filters: dict | None,
) -> bool:
    if not filters:
        return True

    unknown = set(filters) - ALLOWED_FILTERS

    if unknown:
        names = ", ".join(
            sorted(unknown)
        )
        raise ValueError(
            f"Невідомі фільтри: {names}"
        )

    return all(
        chunk.metadata.get(field) == value
        for field, value in filters.items()
    )


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:

    if top_k <= 0:
        raise ValueError(
            "top_k повинен бути більшим за 0."
        )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    ).reshape(-1)

    if (
        query_vector.shape[0]
        != index.vectors.shape[1]
    ):
        raise ValueError(
            "Розмірність query-вектора "
            "не збігається з індексом."
        )

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:
        raise ValueError(
            "Query-вектор не може бути нульовим."
        )

    query_vector = (
        query_vector / query_norm
    )

    candidate_indices = np.array(
        [
            i
            for i, chunk in enumerate(index.chunks)
            if _matches_filters(
                chunk,
                filters,
            )
        ],
        dtype=np.int64,
    )

    if len(candidate_indices) == 0:
        return []

    candidate_vectors = (
        index.vectors[candidate_indices]
    )

    scores = (
        candidate_vectors @ query_vector
    )

    if threshold is not None:
        mask = scores >= threshold
        candidate_indices = (
            candidate_indices[mask]
        )
        scores = scores[mask]

    if len(candidate_indices) == 0:
        return []

    order = np.argsort(
        -scores
    )[:top_k]

    return [
        Hit(
            chunk=index.chunks[
                candidate_indices[i]
            ],
            score=float(
                scores[i]
            ),
        )
        for i in order
    ]