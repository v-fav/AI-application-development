"""Веб-рівень: HTTP-статуси, єдиний формат помилок і SSE-події."""

import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples

_LOG = logging.getLogger("pr10.api")
app = FastAPI(title="Помічник оператора — ПР10")
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    request_id: str
    client: str = "anonymous"


class ApiProblem(Exception):
    def __init__(self, status_code: int, kind: str, message: str, *,
                 retryable: bool = False, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after


def _error_body(kind: str, message: str, retryable: bool, request_id: str) -> dict:
    return {"error": {
        "kind": kind,
        "message": message,
        "retryable": retryable,
        "request_id": request_id,
    }}


@app.exception_handler(ApiProblem)
async def api_problem_handler(request: Request, exc: ApiProblem) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    headers = {"X-Request-ID": error_id}
    if exc.retry_after is not None:
        retry_seconds = max(1, int(exc.retry_after + 0.999))
        headers["Retry-After"] = str(retry_seconds)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.kind, exc.message, exc.retryable, error_id),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    return JSONResponse(
        status_code=400,
        content=_error_body("bad_request", "Перевірте дані запиту й повторіть спробу.", False, error_id),
        headers={"X-Request-ID": error_id},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    _LOG.exception("Unhandled API error; request_id=%s", error_id)
    return JSONResponse(
        status_code=500,
        content=_error_body("unknown", "Сталася внутрішня помилка. Спробуйте ще раз.", True, error_id),
        headers={"X-Request-ID": error_id},
    )


def draft_to_dict(draft: jobs.Draft) -> dict:
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    return [samples.public(sample) for sample in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    sample = samples.get(payload.request_id)
    if sample is None:
        raise ApiProblem(404, "not_found", "Звернення з таким номером не знайдено.")
    try:
        limits.acquire(payload.client)
    except limits.RateLimited as exc:
        raise ApiProblem(
            429, "rate_limit", "Ви створили забагато чернеток. Зачекайте й повторіть.",
            retryable=True, retry_after=exc.retry_after,
        ) from exc
    try:
        draft = jobs.start(sample, payload.client)
    except Exception as exc:
        _LOG.exception("Could not start draft for sample %s", payload.request_id)
        raise ApiProblem(503, "unavailable", "Не вдалося запустити генерацію. Спробуйте ще раз.", retryable=True) from exc
    return draft_to_dict(draft)


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    return [draft_to_dict(draft) for draft in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    draft = jobs.get(draft_id)
    if draft is None:
        raise ApiProblem(404, "not_found", "Чернетку не знайдено.")
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    if jobs.get(draft_id) is None:
        raise ApiProblem(404, "not_found", "Чернетку не знайдено.")
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    try:
        draft = jobs.cancel(draft_id)
    except KeyError as exc:
        raise ApiProblem(404, "not_found", "Чернетку не знайдено.") from exc
    except ValueError as exc:
        raise ApiProblem(409, "conflict", "Цю чернетку вже завершено — скасувати її неможливо.") from exc
    return draft_to_dict(draft)
