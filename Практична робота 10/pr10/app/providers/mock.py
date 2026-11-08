"""Підставний провайдер для розробки й відтворення збоїв без API-ключа."""

import os
import random
import threading
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = max(0, int(os.getenv("MOCK_FIRST_DELAY_MS", "600")))
DELAY_MS = max(0, int(os.getenv("MOCK_DELAY_MS", "40")))
FAILURE_RATE = min(1.0, max(0.0, float(os.getenv("MOCK_FAILURE_RATE", "0"))))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable").strip().lower()
RETRY_AFTER = max(0.1, float(os.getenv("MOCK_RETRY_AFTER", "1.5")))
FAIL_AFTER_CHUNKS = max(1, int(os.getenv("MOCK_FAIL_AFTER_CHUNKS", "3")))
ATTEMPT_TIMEOUT = max(0.1, float(os.getenv("LLM_TIMEOUT", "20")))


def _sleep(seconds: float, stop: threading.Event | None) -> None:
    """Затримка, яку можна перервати під час скасування або таймауту."""
    end = time.monotonic() + max(0.0, seconds)
    while True:
        if stop is not None and stop.is_set():
            raise ProviderError("cancelled", "Генерацію скасовано.")
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.03))


def _chunks(text: str, width: int = 34) -> list[str]:
    """Розбити текст на невеликі фрагменти, не розрізаючи слова."""
    words = text.split()
    result: list[str] = []
    current = ""
    for word in words:
        addition = word if not current else " " + word
        if current and len(current) + len(addition) > width:
            result.append(current)
            current = word
        else:
            current += addition
    if current:
        result.append(current)
    return result or [""]


class MockProvider:
    """Провайдер, який віддає reply_example зі зразка та імітує збої."""

    name = "mock"
    model = "mock-1"

    def _reply(self, request: Request) -> str:
        request_id = str(request.metadata.get("request_id", ""))
        # Імпорт усередині методу не створює циклічної залежності під час
        # завантаження модулів.
        from .. import samples

        sample = samples.get(request_id)
        if sample and sample.get("reply_example"):
            return str(sample["reply_example"])
        messages = request.messages or []
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return "Добрий день! Дякуємо за звернення. " + str(message["content"])
        return "Добрий день! Дякуємо за звернення до служби підтримки «Сузірʼя»."

    def _failure(self, request: Request) -> str | None:
        forced = request.metadata.get("mock_failure_kind")
        kind = str(forced or FAILURE_KIND).lower()
        rate = request.metadata.get("mock_failure_rate", FAILURE_RATE)
        try:
            rate = min(1.0, max(0.0, float(rate)))
        except (TypeError, ValueError):
            rate = FAILURE_RATE
        if rate <= 0 or random.random() >= rate:
            return None
        return kind

    def stream(self, request: Request) -> Iterator[Chunk]:
        stop = request.metadata.get("_stop_event")
        if not isinstance(stop, threading.Event):
            stop = None

        failure = self._failure(request)
        if failure == "timeout":
            # Саме затримка, а не миттєвий виняток: так перевіряється
            # таймаут шлюзу на очікування першого фрагмента.
            delay = max(FIRST_DELAY_MS / 1000, ATTEMPT_TIMEOUT + 1.0)
        else:
            delay = FIRST_DELAY_MS / 1000
        _sleep(delay, stop)

        if failure == "rate_limit":
            raise ProviderError(
                "rate_limit", "Сервіс тимчасово перевантажений. Спробуйте ще раз пізніше.",
                retryable=True, retry_after=RETRY_AFTER, detail="mock: simulated HTTP 429",
            )
        if failure == "auth":
            raise ProviderError(
                "auth", "Не вдалося авторизуватися в AI-сервісі. Перевірте налаштування доступу.",
                retryable=False, detail="mock: simulated authentication failure",
            )
        if failure == "unavailable":
            raise ProviderError(
                "unavailable", "AI-сервіс тимчасово недоступний.", retryable=True,
                detail="mock: simulated provider outage",
            )
        if failure not in (None, "timeout", "mid_stream", "rate_limit", "auth", "unavailable"):
            raise ProviderError(
                "unknown", "Сталася непередбачена помилка AI-сервісу.",
                retryable=False, detail=f"mock: unknown failure kind {failure!r}",
            )

        text = self._reply(request)
        pieces = _chunks(text)
        for index, piece in enumerate(pieces, start=1):
            if index > 1:
                _sleep(DELAY_MS / 1000, stop)
            if failure == "mid_stream" and index > FAIL_AFTER_CHUNKS:
                raise ProviderError(
                    "unavailable", "Потік відповіді перервався. Частину тексту вже отримано.",
                    retryable=True, detail=f"mock: simulated stream interruption after {index - 1} chunks",
                )
            yield Chunk(text=piece + (" " if index < len(pieces) else ""))

        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in request.messages) // 4)
        completion_tokens = max(1, len(text) // 4)
        yield Chunk(
            text="",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
        )

    def complete(self, request: Request) -> Result:
        started = time.monotonic()
        text_parts: list[str] = []
        usage = None
        finish_reason = None
        for chunk in self.stream(request):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        return Result(
            text="".join(text_parts), provider=self.name, model=self.model,
            usage=usage, finish_reason=finish_reason, elapsed=time.monotonic() - started,
        )
