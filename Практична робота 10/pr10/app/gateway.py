"""Шлюз: таймаути, повтори, обмеження одночасності та fallback."""

import logging
import os
import queue
import random
import threading
import time
from dataclasses import replace
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, ProviderError, Request, Result

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
FALLBACK = os.getenv("LLM_FALLBACK", "none").strip()
BASE_URL = os.getenv("LLM_BASE_URL", "")
API_KEY = os.getenv("LLM_API_KEY", "")

TIMEOUT = max(0.1, float(os.getenv("LLM_TIMEOUT", "20")))
IDLE_TIMEOUT = max(0.1, float(os.getenv("LLM_IDLE_TIMEOUT", "8")))
TOTAL_TIMEOUT = max(0.1, float(os.getenv("LLM_TOTAL_TIMEOUT", "45")))
MAX_ATTEMPTS = max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "3")))
BACKOFF_BASE = max(0.0, float(os.getenv("LLM_BACKOFF_BASE", "0.5")))
BACKOFF_MAX = max(BACKOFF_BASE, float(os.getenv("LLM_BACKOFF_MAX", "8")))
MAX_CONCURRENT = max(1, int(os.getenv("LLM_MAX_CONCURRENT", "2")))
MODEL = os.getenv("LLM_MODEL", "")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = max(1, int(os.getenv("LLM_MAX_TOKENS", "600")))

# Глобальний семафор: спільний для всіх операторів і фонових задач процесу.
_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT)
_LOG = logging.getLogger("pr10.gateway")


def settings() -> dict:
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Створити провайдера; довільна назва означає іншу модель API."""
    selected = (name or PROVIDER).strip()
    if selected.lower() == "mock":
        from .providers.mock import MockProvider
        return MockProvider()

    from .providers.openai_compat import OpenAICompatProvider
    model = MODEL if selected.lower() == "openai" else selected
    return OpenAICompatProvider(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=model,
        timeout=TIMEOUT,
    )


def _attempt(
    provider: Provider,
    request: Request,
    *,
    deadline: float,
    is_cancelled: Callable[[], bool],
    on_delta: Callable[[str], None],
    attempt_number: int,
    fallback: bool,
    on_status: Callable[[dict], None],
) -> Result:
    """Виконати один streaming-виклик у worker-потоці.

    Черга дозволяє шлюзу окремо контролювати загальний таймаут спроби та
    таймаут бездіяльності між фрагментами. Семафор звільняє саме worker,
    коли виклик провайдера фактично закінчився.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderError("timeout", "Вичерпано загальний час очікування.", retryable=True)
    if is_cancelled():
        raise ProviderError("cancelled", "Генерацію скасовано.")

    on_status({
        "status": "running", "attempt": attempt_number, "provider": provider.name,
        "fallback": fallback, "detail": "очікування вільного каналу",
    })
    try:
        acquired = _SEMAPHORE.acquire(timeout=max(0.01, min(remaining, TIMEOUT)))
    except Exception:
        acquired = False
    if not acquired:
        raise ProviderError(
            "unavailable", "Усі канали AI-сервісу зайняті. Спробуйте трохи пізніше.",
            retryable=True,
        )
    on_status({
        "status": "running", "attempt": attempt_number, "provider": provider.name,
        "fallback": fallback, "detail": "запит надіслано провайдеру; очікуємо фрагменти",
    })

    messages: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    metadata = dict(request.metadata)
    metadata["_stop_event"] = stop_event
    attempt_request = replace(request, metadata=metadata)
    attempt_started = time.monotonic()

    def stop_provider() -> None:
        stop_event.set()
        close_stream = metadata.get("_close_stream")
        if callable(close_stream):
            try:
                close_stream()
            except Exception:
                pass

    def worker() -> None:
        stream = None
        try:
            stream = provider.stream(attempt_request)
            for chunk in stream:
                if stop_event.is_set():
                    break
                messages.put(("chunk", chunk))
            messages.put(("end", None))
        except ProviderError as exc:
            messages.put(("error", exc))
        except Exception as exc:  # захист від помилок реалізації провайдера
            _LOG.exception("Непередбачений збій у провайдері %s", provider.name)
            messages.put(("error", ProviderError(
                "unknown", "Сталася непередбачена помилка AI-сервісу.",
                retryable=False, detail=f"{type(exc).__name__}: {exc}",
            )))
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            _SEMAPHORE.release()

    thread = threading.Thread(target=worker, name="pr10-provider", daemon=True)
    try:
        thread.start()
    except Exception:
        _SEMAPHORE.release()
        raise

    pieces: list[str] = []
    usage = None
    finish_reason = None
    first_fragment_at: float | None = None

    try:
        while True:
            if is_cancelled():
                stop_provider()
                raise ProviderError("cancelled", "Генерацію скасовано.")

            now = time.monotonic()
            total_left = deadline - now
            attempt_left = TIMEOUT - (now - attempt_started)
            if total_left <= 0:
                stop_provider()
                raise ProviderError("timeout", "Перевищено загальний час генерації.", retryable=True)
            if attempt_left <= 0:
                stop_provider()
                raise ProviderError("timeout", "AI-сервіс не надіслав відповідь у межах таймауту спроби.", retryable=True)

            wait_for = min(total_left, attempt_left)
            if first_fragment_at is not None:
                idle_left = IDLE_TIMEOUT - (now - last_activity)
                if idle_left <= 0:
                    stop_provider()
                    raise ProviderError("timeout", "Потік відповіді надто довго не надсилав нових фрагментів.", retryable=True)
                wait_for = min(wait_for, idle_left)

            try:
                kind, value = messages.get(timeout=max(0.01, wait_for))
            except queue.Empty:
                stop_provider()
                now = time.monotonic()
                if deadline - now <= 0:
                    raise ProviderError("timeout", "Перевищено загальний час генерації.", retryable=True)
                if TIMEOUT - (now - attempt_started) <= 0:
                    raise ProviderError("timeout", "Перевищено час однієї спроби.", retryable=True)
                raise ProviderError("timeout", "Потік відповіді не надсилав фрагментів вчасно.", retryable=True)

            if kind == "error":
                raise value
            if kind == "end":
                result_text = "".join(pieces)
                if not result_text.strip():
                    raise ProviderError(
                        "unavailable", "AI-сервіс завершив відповідь без тексту.", retryable=True,
                    )
                return Result(
                    text=result_text, provider=provider.name,
                    model=getattr(provider, "model", request.model or "unknown"),
                    usage=usage, finish_reason=finish_reason,
                    elapsed=time.monotonic() - attempt_started,
                )
            if kind == "chunk":
                chunk = value
                if chunk.usage:
                    usage = chunk.usage
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.text:
                    if first_fragment_at is None:
                        first_fragment_at = time.monotonic()
                        on_status({
                            "status": "running", "attempt": attempt_number,
                            "provider": provider.name, "fallback": fallback,
                            "first_token": first_fragment_at - attempt_started,
                            "detail": f"перший фрагмент за {first_fragment_at - attempt_started:.2f} с",
                        })
                    last_activity = time.monotonic()
                    pieces.append(chunk.text)
                    on_delta(chunk.text)
    finally:
        # Worker звільнить семафор після фактичного завершення виклику.
        # Під час таймауту/cancel stop_event вже виставлено вище.
        pass


def _sleep_with_cancel(seconds: float, deadline: float, is_cancelled: Callable[[], bool]) -> bool:
    """Почекати перед повтором; False означає скасування/вичерпання бюджету."""
    end = min(time.monotonic() + max(0.0, seconds), deadline)
    while time.monotonic() < end:
        if is_cancelled():
            return False
        time.sleep(min(0.05, end - time.monotonic()))
    return not is_cancelled() and time.monotonic() < deadline


def generate(
    request: Request,
    on_delta: Callable[[str], None],
    on_status: Callable[[dict], None],
    is_cancelled: Callable[[], bool],
) -> Result:
    """Згенерувати відповідь із таймаутами, обмеженими повторами й fallback."""
    started = time.monotonic()
    deadline = started + TOTAL_TIMEOUT
    primary = make_provider(PROVIDER)
    last_error = ProviderError("unknown", "Не вдалося отримати відповідь від AI-сервісу.")
    emitted_any = False
    attempts_made = 0

    def send_delta(value: str) -> None:
        nonlocal emitted_any
        emitted_any = True
        on_delta(value)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_made = attempt
        if is_cancelled():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        try:
            result = _attempt(
                primary, request, deadline=deadline, is_cancelled=is_cancelled,
                on_delta=send_delta, attempt_number=attempt, fallback=False,
                on_status=on_status,
            )
            return replace(result, elapsed=time.monotonic() - started)
        except ProviderError as exc:
            last_error = exc
            if exc.kind == "cancelled" or is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            # Після першого показаного фрагмента повтор почав би текст знову
            # і заплутав би оператора. У такому разі залишаємо частковий текст.
            if emitted_any or not exc.retryable or attempt >= MAX_ATTEMPTS:
                break
            delay = exc.retry_after
            if delay is None:
                delay = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1)))
                delay *= random.uniform(0.5, 1.5)
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                break
            on_status({
                "status": "running", "attempt": attempt, "provider": primary.name,
                "fallback": False, "detail": f"тимчасовий збій; повтор через {delay:.1f} с",
            })
            if not _sleep_with_cancel(delay, deadline, is_cancelled):
                if is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.")
                break

    # Запасний варіант запускається лише якщо основний не показав оператору
    # жодного фрагмента: ми не змішуємо два різні тексти в одному полі.
    if not emitted_any and FALLBACK.lower() not in ("", "none", PROVIDER.lower()) and time.monotonic() < deadline:
        try:
            backup = make_provider(FALLBACK)
            on_status({
                "status": "running", "attempt": attempts_made, "provider": backup.name,
                "fallback": True, "detail": "основний провайдер не впорався; використовується запасний варіант",
            })
            result = _attempt(
                backup, request, deadline=deadline, is_cancelled=is_cancelled,
                on_delta=send_delta, attempt_number=attempts_made, fallback=True,
                on_status=on_status,
            )
            return replace(result, elapsed=time.monotonic() - started)
        except ProviderError as fallback_error:
            if fallback_error.kind == "cancelled" or is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            _LOG.warning("Fallback provider failed: %s", fallback_error.detail or fallback_error.kind)
            last_error = fallback_error

    if time.monotonic() >= deadline and last_error.kind != "timeout":
        last_error = ProviderError("timeout", "Вичерпано загальний час очікування.", retryable=True)
    _LOG.warning("Generation failed: kind=%s retryable=%s detail=%s", last_error.kind,
                 last_error.retryable, last_error.detail)
    raise last_error
