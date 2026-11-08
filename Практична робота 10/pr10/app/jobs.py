"""Фонові задачі, стан чернеток, журнал подій і кооперативне скасування."""

import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway
from .providers.base import ProviderError

TERMINAL = {"done", "failed", "cancelled"}
_LOG = logging.getLogger("pr10.jobs")


@dataclass
class Draft:
    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancel: dict[str, threading.Event] = {}
_lock = threading.Lock()
_changed = threading.Condition(_lock)
_MAX_DRAFTS = 200


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _append_event_locked(draft: Draft, event: str, data: dict | None = None) -> None:
    draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
    draft.updated_at = _now()
    _changed.notify_all()


def new_draft(request_id: str, client: str) -> Draft:
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id, client=client,
                  created_at=_now(), updated_at=_now())
    with _changed:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        # Реєстр обмежений, але активні роботи не видаляємо.
        while len(_order) > _MAX_DRAFTS:
            old_id = _order[-1]
            old = _drafts.get(old_id)
            if old is not None and old.status not in TERMINAL:
                break
            _order.pop()
            _drafts.pop(old_id, None)
            _cancel.pop(old_id, None)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    with _lock:
        return [_drafts[i] for i in _order[:limit] if i in _drafts]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    with _changed:
        _append_event_locked(draft, event, data)


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def _set_cancelled_locked(draft: Draft) -> None:
    draft.status = "cancelled"
    draft.updated_at = _now()
    _append_event_locked(draft, "cancelled", {"text": draft.text})


def _run(draft: Draft, sample: dict, cancel_event: threading.Event) -> None:
    started = time.monotonic()
    first_token_at: float | None = None

    def on_status(info: dict) -> None:
        with _changed:
            if draft.status in TERMINAL or cancel_event.is_set():
                return
            draft.status = info.get("status", "running")
            if "attempt" in info:
                draft.attempts = max(draft.attempts, int(info["attempt"]))
            if info.get("provider"):
                draft.provider = str(info["provider"])
            if "fallback" in info:
                draft.fallback = bool(info["fallback"])
            _append_event_locked(draft, "status", info)

    def on_delta(value: str) -> None:
        nonlocal first_token_at
        with _changed:
            if draft.status in TERMINAL or cancel_event.is_set():
                return
            if first_token_at is None:
                first_token_at = time.monotonic()
            draft.text += value
            _append_event_locked(draft, "delta", {"text": value})

    try:
        with _changed:
            if draft.status in TERMINAL or cancel_event.is_set():
                return
            draft.status = "running"
            _append_event_locked(draft, "status", {
                "status": "running", "attempt": 0, "provider": gateway.PROVIDER,
                "fallback": False, "detail": "фонову генерацію розпочато",
            })
        request = drafting.build_request(sample)
        result = gateway.generate(request, on_delta, on_status, cancel_event.is_set)
        with _changed:
            # cancel() і завершення генерації змагаються за тим самим lock:
            # якщо оператор скасував першим, статус done вже не встановиться.
            if draft.status in TERMINAL or cancel_event.is_set():
                if draft.status != "cancelled":
                    _set_cancelled_locked(draft)
                return
            draft.text = result.text
            draft.provider = result.provider
            draft.model = result.model
            draft.usage = result.usage
            draft.elapsed = {
                "first_token": max(0.0, first_token_at - started) if first_token_at else 0.0,
                "total": float(result.elapsed or (time.monotonic() - started)),
            }
            draft.warnings = drafting.check(result.text, sample)
            draft.status = "done"
            _append_event_locked(draft, "done", {
                "text": draft.text,
                "provider": draft.provider,
                "model": draft.model,
                "usage": draft.usage,
                "elapsed": draft.elapsed,
                "attempts": draft.attempts,
                "fallback": draft.fallback,
                "warnings": draft.warnings,
            })
    except ProviderError as exc:
        if exc.kind == "cancelled" or cancel_event.is_set():
            with _changed:
                if draft.status not in TERMINAL:
                    _set_cancelled_locked(draft)
            return
        _LOG.warning("Draft %s failed: %s", draft.draft_id, exc.detail or exc.kind)
        with _changed:
            if draft.status in TERMINAL:
                return
            draft.status = "failed"
            draft.error = {"kind": exc.kind, "message": exc.message, "retryable": exc.retryable}
            _append_event_locked(draft, "error", {
                "kind": exc.kind,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": draft.draft_id,
            })
    except Exception:
        _LOG.exception("Unexpected error while generating draft %s", draft.draft_id)
        with _changed:
            if draft.status in TERMINAL:
                return
            draft.status = "failed"
            draft.error = {
                "kind": "unknown",
                "message": "Не вдалося створити чернетку через внутрішню помилку. Спробуйте ще раз.",
                "retryable": True,
            }
            _append_event_locked(draft, "error", {
                **draft.error, "request_id": draft.draft_id,
            })
    finally:
        with _lock:
            _cancel.pop(draft.draft_id, None)


def start(sample: dict, client: str) -> Draft:
    if not sample or not sample.get("id"):
        raise ValueError("Невідоме звернення")
    draft = new_draft(str(sample["id"]), str(client or "anonymous"))
    cancel_event = threading.Event()
    with _changed:
        _cancel[draft.draft_id] = cancel_event
        _append_event_locked(draft, "status", {
            "status": "queued", "attempt": 0, "provider": gateway.PROVIDER,
            "fallback": False, "detail": "чернетку додано до черги",
        })
    thread = threading.Thread(
        target=_run, args=(draft, sample, cancel_event),
        name=f"draft-{draft.draft_id}", daemon=True,
    )
    try:
        thread.start()
    except Exception:
        with _changed:
            draft.status = "failed"
            draft.error = {"kind": "unavailable", "message": "Не вдалося запустити фонову задачу.", "retryable": True}
            _append_event_locked(draft, "error", {**draft.error, "request_id": draft.draft_id})
            _cancel.pop(draft.draft_id, None)
        raise
    return draft


def cancel(draft_id: str) -> Draft:
    with _changed:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise KeyError(draft_id)
        if draft.status in TERMINAL:
            raise ValueError(f"Чернетку вже завершено зі статусом {draft.status}")
        event = _cancel.get(draft_id)
        if event is not None:
            event.set()
        _set_cancelled_locked(draft)
        return draft
