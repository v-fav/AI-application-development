"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані як шукаються фрагменти, ані як збирається контекст,
ані якою моделлю й за якою інструкцією отримано відповідь — усе це
лишається в `app/retrieval.py`, `app/llm.py`, `app/schema.py` і
поєднується в `app/rag.py`. Тут вирішується інше: що застосунок приймає
від сторінки, що віддає їй і з яким HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr6),
а тут лише читається при старті — як у ПР5.

Запуск із папки pr6:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""


from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import index, keyword, rag, retrieval, llm

app = FastAPI(title="Помічник за базою знань — ПР6")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Запит від сторінки."""

    question: str
    filters: dict = Field(default_factory=dict)


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    """Перетворити результат RAG на JSON."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {
                "ref": source.ref,
                "score": source.score,
                "source": source.chunk.source,
                "metadata": source.chunk.metadata,
                "text": source.chunk.text,
            }
            for source in result.sources
        ],
        "retrieved": [hit_to_dict(hit) for hit in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Завантажити індекси під час запуску."""
    app.state.index = None
    app.state.keyword_index = None

    try:
        app.state.index = index.load()
        app.state.keyword_index = keyword.build(app.state.index.chunks)
    except Exception as exc:
        app.state.index = None
        app.state.keyword_index = None
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Повернути HTML-сторінку."""
    if not INDEX_PAGE.exists():
        raise HTTPException(
            status_code=500,
            detail="Не знайдено файл app/templates/index.html",
        )

    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Повернути стан індексу."""
    idx = app.state.index

    if idx is None:
        return {
            "ready": False,
            "hint": "Індекс не збудовано. Виконайте команду python ingest.py",
        }

    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len({chunk.source for chunk in idx.chunks}),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Обробити питання клієнта."""
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Введіть питання.",
        )

    if len(question) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Питання занадто довге. Максимум — 2000 символів.",
        )

    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не готовий. Спочатку виконайте python ingest.py.",
        )

    filters = payload.filters or {}

    # Клієнту дозволено фільтрувати лише за товаром.
    if not isinstance(filters, dict) or set(filters) - {"product"}:
        raise HTTPException(
            status_code=400,
            detail="Дозволений лише фільтр за товаром.",
        )

    try:
        result = rag.answer(
            question,
            app.state.index,
            app.state.keyword_index,
            filters=filters or None,
        )
        return answer_to_dict(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except llm.LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(f"Помилка обробки питання: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Не вдалося обробити питання. Спробуйте пізніше.",
        ) from exc