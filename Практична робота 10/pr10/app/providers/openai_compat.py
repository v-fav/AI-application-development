"""Реалізація контракту для OpenAI-сумісного API (зокрема Gemini)."""

from datetime import datetime, timezone
import threading
from email.utils import parsedate_to_datetime
from typing import Iterator

from .base import Chunk, ProviderError, Request, Result


def _retry_after(headers) -> float | None:
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                moment = parsedate_to_datetime(value)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    value_ms = headers.get("retry-after-ms")
    if value_ms:
        try:
            return max(0.0, float(value_ms) / 1000)
        except (TypeError, ValueError):
            return None
    return None


class OpenAICompatProvider:
    """Провайдер поверх SDK openai; SDK не використовується поза цим файлом."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = max(0.1, float(timeout))
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise ProviderError(
                "auth", "Не вказано API-ключ. Додайте LLM_API_KEY у файл .env або виберіть mock.",
                retryable=False, detail="LLM_API_KEY is empty",
            )
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url or None,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=0,
                )
            except ImportError as exc:
                raise ProviderError(
                    "unavailable", "Не встановлено пакет openai. Виконайте pip install -r requirements.txt.",
                    retryable=False, detail=repr(exc),
                ) from exc
        return self._client

    @staticmethod
    def _convert_error(exc: Exception) -> ProviderError:
        try:
            import openai
        except ImportError:
            return ProviderError("unknown", "Сталася помилка AI-сервісу.", detail=repr(exc))

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = _retry_after(headers)
        detail = f"{type(exc).__name__}: {exc}"

        if isinstance(exc, openai.APITimeoutError):
            return ProviderError("timeout", "AI-сервіс не відповів вчасно.", retryable=True, detail=detail)
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(
                "rate_limit", "AI-сервіс тимчасово обмежив кількість запитів.",
                retryable=True, retry_after=retry_after, detail=detail,
            )
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError("auth", "Не вдалося авторизуватися в AI-сервісі. Перевірте API-ключ.", detail=detail)
        if isinstance(exc, openai.BadRequestError):
            return ProviderError("bad_request", "AI-сервіс відхилив запит. Перевірте його параметри.", detail=detail)
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError("unavailable", "Не вдалося зʼєднатися з AI-сервісом.", retryable=True, detail=detail)
        if isinstance(exc, openai.APIStatusError):
            code = int(getattr(exc, "status_code", 0) or 0)
            if code == 429:
                return ProviderError(
                    "rate_limit", "AI-сервіс тимчасово обмежив кількість запитів.",
                    retryable=True, retry_after=retry_after, detail=detail,
                )
            if code in (401, 403):
                return ProviderError("auth", "AI-сервіс не прийняв облікові дані.", detail=detail)
            if code >= 500:
                return ProviderError("unavailable", "AI-сервіс тимчасово недоступний.", retryable=True, detail=detail)
            return ProviderError("bad_request", "AI-сервіс відхилив запит.", detail=detail)
        return ProviderError("unknown", "Сталася непередбачена помилка AI-сервісу.", detail=detail)

    @staticmethod
    def _usage(value) -> dict | None:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        if isinstance(value, dict):
            return value
        return None

    def complete(self, request: Request) -> Result:
        client = self._get_client()
        model = request.model or self.model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
            choice = response.choices[0] if response.choices else None
            text = (choice.message.content or "") if choice else ""
            return Result(
                text=text, provider=self.name, model=model,
                usage=self._usage(getattr(response, "usage", None)),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._convert_error(exc) from exc

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Повернути ітератор; close callback дозволяє шлюзу зупинити HTTP-потік."""
        client = self._get_client()
        model = request.model or self.model
        try:
            raw_stream = client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._convert_error(exc) from exc

        close_lock = threading.Lock()
        closed = False

        def close_stream() -> None:
            nonlocal closed
            with close_lock:
                if closed:
                    return
                closed = True
                close = getattr(raw_stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        # Цей callback — внутрішня деталь контракту; він не передається API.
        request.metadata["_close_stream"] = close_stream
        stop_event = request.metadata.get("_stop_event")

        def iterate() -> Iterator[Chunk]:
            try:
                if stop_event is not None and stop_event.is_set():
                    return
                for part in raw_stream:
                    if stop_event is not None and stop_event.is_set():
                        break
                    choices = getattr(part, "choices", None) or []
                    if choices:
                        choice = choices[0]
                        delta = getattr(choice, "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        yield Chunk(
                            text=content or "",
                            usage=self._usage(getattr(part, "usage", None)),
                            finish_reason=getattr(choice, "finish_reason", None),
                        )
                    else:
                        usage = self._usage(getattr(part, "usage", None))
                        if usage:
                            yield Chunk(text="", usage=usage)
            except ProviderError:
                raise
            except Exception as exc:
                raise self._convert_error(exc) from exc
            finally:
                close_stream()

        return iterate()
