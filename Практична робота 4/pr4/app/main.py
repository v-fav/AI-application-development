from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import llm


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
CONTEXT_FILE = BASE_DIR.parent / "context.md"


app = FastAPI(
    title="Сузір'я — Помічник служби підтримки",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.get("/")
def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def api_chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Повідомлення не може бути порожнім.",
        )

    history = [
        {
            "role": item.role,
            "content": item.content,
        }
        for item in request.history
    ]

    try:
        if not CONTEXT_FILE.exists():
            raise FileNotFoundError(
                f"Файл context.md не знайдено: {CONTEXT_FILE}"
            )

        context = CONTEXT_FILE.read_text(encoding="utf-8")

        result = llm.ask(
            message,
            history,
            context,
        )

        return result

    except llm.LLMError as exc:
        print(f"LLM ERROR: {exc}")
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:
        print(f"SERVER ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутрішня помилка сервера: {exc}",
        )