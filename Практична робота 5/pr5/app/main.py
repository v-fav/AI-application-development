"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти."""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import embeddings, index, keyword


app = FastAPI(
    title="Пошук у базі знань — ПР5"
)

INDEX_PAGE = (
    Path(__file__).parent
    / "templates"
    / "index.html"
)


class SearchRequest(BaseModel):
    query: str
    mode: str = "both"
    top_k: int = index.DEFAULT_TOP_K
    filters: dict = Field(
        default_factory=dict
    )
    threshold: float | None = None


def hit_to_dict(
    hit: index.Hit,
) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.on_event("startup")
def load_indexes() -> None:
    app.state.index = None
    app.state.keyword_index = None

    try:
        app.state.index = index.load()
    except Exception as exc:
        print(
            f"Індекс не завантажено: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    app.state.keyword_index = keyword.build(
        app.state.index.chunks
    )


@app.get(
    "/",
    response_class=HTMLResponse,
)
def page() -> str:
    return INDEX_PAGE.read_text(
        encoding="utf-8"
    )


@app.get("/api/status")
def api_status() -> dict:
    idx = app.state.index

    if idx is None:
        return {
            "ready": False,
            "hint": (
                "індекс не збудовано — "
                "виконайте python ingest.py"
            ),
        }

    sources = {
        chunk.source
        for chunk in idx.chunks
    }

    return {
        "ready": True,
        "chunks": len(idx.chunks),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/search")
def api_search(
    payload: SearchRequest,
) -> dict:

    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail=(
                "Пошуковий запит "
                "не може бути порожнім."
            ),
        )

    if payload.mode not in (
        "semantic",
        "keyword",
        "both",
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Невідомий режим пошуку. "
                "Дозволено: semantic, "
                "keyword, both."
            ),
        )

    if payload.top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "top_k повинен бути "
                "більшим за 0."
            ),
        )

    idx = app.state.index

    if idx is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Індекс не збудовано. "
                "Спочатку виконайте "
                "python ingest.py."
            ),
        )

    filters = dict(
        payload.filters or {}
    )

    result = {
        "query": query,
        "semantic": None,
        "keyword": None,
        "elapsed": {},
    }

    if payload.mode in (
        "semantic",
        "both",
    ):
        started = time.perf_counter()

        try:
            vector = embeddings.embed_query(
                query
            )

            hits = index.search(
                idx,
                vector,
                top_k=payload.top_k,
                filters=filters or None,
                threshold=(
                    payload.threshold
                    if payload.threshold
                    is not None
                    else index.SIMILARITY_THRESHOLD
                ),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        result["elapsed"]["semantic"] = (
            time.perf_counter()
            - started
        )

        result["semantic"] = [
            hit_to_dict(hit)
            for hit in hits
        ]

    if payload.mode in (
        "keyword",
        "both",
    ):
        started = time.perf_counter()

        try:
            hits = keyword.search(
                app.state.keyword_index,
                query,
                top_k=payload.top_k,
                filters=filters or None,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        result["elapsed"]["keyword"] = (
            time.perf_counter()
            - started
        )

        result["keyword"] = [
            hit_to_dict(hit)
            for hit in hits
        ]

    return result