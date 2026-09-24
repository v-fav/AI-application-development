"""Веб-рівень застосунку: сторінка та JSON-ендпоінт."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Question(BaseModel):
    """Звернення користувача."""

    question: str


def load_context() -> str:
    """Прочитати актуальні правила магазину."""

    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""

    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON."""

    try:
        return llm.ask(payload.question, load_context())

    except llm.LLMError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        )