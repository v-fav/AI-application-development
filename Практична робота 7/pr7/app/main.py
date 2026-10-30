"""FastAPI веб-рівень ПР7."""
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from . import extraction, images, llm, schema

app = FastAPI(title="Розбір рахунків — ПР7")
ROOT = Path(__file__).resolve().parent.parent
INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = ROOT / "samples"
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

def result_to_dict(result):
    return {
        "decision": result.decision, "reasons": result.reasons,
        "document": result.document, "issues": [asdict(x) for x in result.issues],
        "image": result.image, "model": result.model,
        "elapsed": result.elapsed, "usage": result.usage,
        "raw": result.raw,
    }

@app.get("/", response_class=HTMLResponse)
def page():
    return INDEX_PAGE.read_text(encoding="utf-8")

@app.get("/api/samples")
def api_samples():
    groups = {}
    for folder in ("clean", "degraded", "own"):
        path = SAMPLES_DIR / folder
        if path.exists():
            groups[folder] = sorted(p.name for p in path.iterdir()
                                     if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    return groups

@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)):
    content = await image.read()
    try:
        result = extraction.process(content)
        return result_to_dict(result)
    except images.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except schema.SchemaError as exc:
        raise HTTPException(status_code=502, detail=f"Некоректна відповідь моделі: {exc}")
    except Exception as exc:
        # Не показуємо traceback/секрети користувачу.
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка обробки: {type(exc).__name__}. Перевірте консоль сервера.")
