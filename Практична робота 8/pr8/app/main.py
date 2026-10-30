"""Веб-рівень застосунку."""
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from shop import service
from . import assistant, tools, llm

app = FastAPI(title="Помічник клієнта — ПР8")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=4000)


def answer_to_dict(result: assistant.Answer) -> dict:
    return {
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    known = {c["customer_id"] for c in service.list_customers()}
    if payload.customer_id not in known:
        raise HTTPException(status_code=400, detail="Невідомий клієнт сеансу.")
    try:
        return answer_to_dict(assistant.answer(payload.question, payload.customer_id))
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
