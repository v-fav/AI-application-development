import re

import numpy as np
from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import Hit


ALLOWED_FILTERS = {
    "category",
    "product",
    "audience",
    "status",
}


class KeywordIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Okapi,
    ):
        self.chunks = chunks
        self.bm25 = bm25


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+",
        text.lower(),
    )


def build(
    chunks: list[Chunk],
) -> KeywordIndex:
    if not chunks:
        raise ValueError(
            "Неможливо побудувати "
            "keyword-індекс без chunks."
        )

    tokenized = [
        tokenize(chunk.text)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized)

    return KeywordIndex(
        chunks=chunks,
        bm25=bm25,
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
    index: KeywordIndex,
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[Hit]:

    if not query.strip():
        raise ValueError(
            "Пошуковий запит не може бути порожнім."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k повинен бути більшим за 0."
        )

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = np.asarray(
        index.bm25.get_scores(query_tokens),
        dtype=np.float32,
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

    candidate_scores = scores[
        candidate_indices
    ]

    positive = candidate_scores > 0

    candidate_indices = (
        candidate_indices[positive]
    )
    candidate_scores = (
        candidate_scores[positive]
    )

    if len(candidate_indices) == 0:
        return []

    order = np.argsort(
        -candidate_scores
    )[:top_k]

    return [
        Hit(
            chunk=index.chunks[
                candidate_indices[i]
            ],
            score=float(
                candidate_scores[i]
            ),
        )
        for i in order
    ]