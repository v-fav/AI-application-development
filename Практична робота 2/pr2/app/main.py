"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не має знати про `ultralytics`, ваги моделі й формат її «сирого»
виводу — усе це лишається в `app/detector.py`. Тут вирішується інше: що
застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import detector

app = FastAPI(title="Детекція обʼєктів — ПР2")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Query(
        detector.DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0
    ),
):
    """Прийняти зображення та повернути результат детекції."""

    content = await image.read()

    try:
        result = detector.detect(content, confidence)
        return result

    except detector.DetectionError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )
    """Повернути знайдені на зображенні обʼєкти у форматі JSON.

    Зараз виняток із модуля inference не обробляється — застосунок просто
    впаде з помилкою 500. Спроєктуйте обробку самі: які збої можливі
    (не зображення, порожній файл, помилка моделі), який HTTP-статус
    відповідає кожному з них і що в такому разі отримає клієнт.

    Поріг упевненості поки що жорстко зашитий у модулі. Вирішіть, чи має
    користувач змогу його змінювати, і якщо так — як передати це сюди.
    """
