"""Атомарне JSON-сховище станів звернень."""
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from .state import Run, Step

load_dotenv()
RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


def _path(run_id: str) -> Path:
    # Не дозволяємо використовувати шлях замість ідентифікатора.
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in run_id):
        raise ValueError("Некоректний ідентифікатор звернення.")
    return RUNS_DIR / f"{run_id}.json"


def save(run: Run) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = _path(run.run_id)
    payload = json.dumps(asdict(run), ensure_ascii=False, indent=2, default=str)
    fd, temp_name = tempfile.mkstemp(prefix=f".{run.run_id}-", suffix=".tmp", dir=RUNS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load(run_id: str) -> Run | None:
    path = _path(run_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["steps"] = [Step(**item) for item in data.get("steps", [])]
    return Run(**data)


def list_runs(status: str | None = None) -> list[Run]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            run = load(path.stem)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if run and (status is None or run.status == status):
            result.append(run)
    return sorted(result, key=lambda item: item.created_at or "", reverse=True)
