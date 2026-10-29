
"""Відбір фрагментів для моделі та збирання контексту."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from . import embeddings
from . import index as semantic
from . import keyword as lexical
from .index import Hit, SearchIndex
from .keyword import KeywordIndex

load_dotenv()

CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
RRF_K = 60


@dataclass
class Source:
    """Фрагмент, який показано мовній моделі."""

    ref: int
    chunk: Chunk
    score: float


def _allowed(chunk: Chunk, product: str | None = None) -> bool:
    """Залишити лише чинні документи для клієнтів."""
    metadata = chunk.metadata or {}

    if metadata.get("audience") != "клієнти":
        return False

    if metadata.get("status") != "чинний":
        return False

    if product and metadata.get("product") != product:
        return False

    return True


def _chunk_key(chunk: Chunk) -> tuple:
    """Стабільний ключ фрагмента документа."""
    metadata = chunk.metadata or {}

    chunk_index = metadata.get("chunk_index")

    if chunk_index is not None:
        return chunk.source, chunk_index

    return chunk.source, chunk.text


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Об'єднати семантичний пошук і BM25."""
    query = query.strip()

    if not query:
        return []

    filters = filters or {}
    unknown = set(filters) - {"product"}

    if unknown:
        raise ValueError(
            f"Недозволені фільтри: {', '.join(sorted(unknown))}"
        )

    product = filters.get("product")

    search_filters = {
        "audience": "клієнти",
        "status": "чинний",
    }

    if product:
        search_filters["product"] = product

    limit = max(CONTEXT_CHUNKS * 5, 20)
    combined = {}

    def add_hits(hits: list[Hit]) -> None:
        for rank, hit in enumerate(hits, start=1):
            if not _allowed(hit.chunk, product):
                continue

            key = _chunk_key(hit.chunk)

            if key not in combined:
                combined[key] = {
                    "hit": hit,
                    "score": 0.0,
                }

            combined[key]["score"] += 1.0 / (RRF_K + rank)

    # Семантичний пошук працює з embedding-вектором.
    query_vector = embeddings.embed_query(query)

    semantic_hits = semantic.search(
        index,
        query_vector,
        top_k=limit,
        filters=search_filters,
    )

    add_hits(semantic_hits)

    # BM25 знаходить збіги за словами.
    if keyword_index is not None:
        keyword_hits = lexical.search(
            keyword_index,
            query,
            top_k=limit,
            filters=search_filters,
        )

        add_hits(keyword_hits)

    ranked = [
        Hit(
            chunk=item["hit"].chunk,
            score=item["score"],
        )
        for item in combined.values()
    ]

    ranked.sort(key=lambda hit: hit.score, reverse=True)

    # Спочатку беремо найкращі фрагменти з різних документів.
    selected = []
    selected_keys = set()
    selected_documents = set()

    for hit in ranked:
        source = hit.chunk.source

        if source in selected_documents:
            continue

        selected.append(hit)
        selected_keys.add(_chunk_key(hit.chunk))
        selected_documents.add(source)

        if len(selected) >= CONTEXT_CHUNKS:
            return selected

    # Якщо документів менше за потрібну кількість,
    # заповнюємо контекст наступними найкращими фрагментами.
    for hit in ranked:
        key = _chunk_key(hit.chunk)

        if key in selected_keys:
            continue

        selected.append(hit)
        selected_keys.add(key)

        if len(selected) >= CONTEXT_CHUNKS:
            break

    return selected


def build_context(
    hits: list[Hit],
    budget: int = CONTEXT_BUDGET,
) -> tuple[str, list[Source]]:
    """Зібрати контекст у межах приблизного бюджету токенів."""
    if budget <= 0:
        return "", []

    blocks = []
    sources = []
    used_tokens = 0

    for hit in hits:
        if len(sources) >= CONTEXT_CHUNKS:
            break

        metadata = hit.chunk.metadata or {}

        title = metadata.get("title") or hit.chunk.source
        section = metadata.get("section", "")
        updated = metadata.get("updated", "")

        header = f"Документ: {title}"

        if section:
            header += f"\nРозділ: {section}"

        if updated:
            header += f"\nРедакція: {updated}"

        block = (
            f"[{len(sources) + 1}]\n"
            f"{header}\n"
            f"Текст фрагмента:\n{hit.chunk.text}"
        )

        estimated_tokens = max(1, len(block) // 4)

        if used_tokens + estimated_tokens > budget:
            remaining_chars = max(
                0,
                (budget - used_tokens) * 4,
            )

            if remaining_chars < 100:
                break

            block = block[:remaining_chars]
            estimated_tokens = max(1, len(block) // 4)

        if not block.strip():
            continue

        sources.append(
            Source(
                ref=len(sources) + 1,
                chunk=hit.chunk,
                score=float(hit.score),
            )
        )

        blocks.append(block)
        used_tokens += estimated_tokens

        if used_tokens >= budget:
            break

    return "\n\n---\n\n".join(blocks), sources
