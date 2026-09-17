"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* де саме завантажувати ваги, щоб це сталося **один раз**, а не на кожен запит;
* яким узяти поріг упевненості й чи дозволяти змінювати його ззовні;
* у якому вигляді віддавати результат: які поля, які одиниці координат;
* як виміряти час inference і що саме до нього зараховувати;
* як повестися, коли надійшов не той файл — не зображення або порожній.

Довідка про модель: https://docs.ultralytics.com/
"""

import time
from functools import lru_cache
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25


class DetectionError(Exception):
    """Помилка детекції, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — некоректний файл, збій моделі, — і що з цього має
    побачити користувач.
    """

@lru_cache(maxsize=1)

def load_model():
    """Повернути готову до роботи модель.

    Завантаження ваг коштує дорого. Подумайте, як зробити так, щоб воно
    відбулося один раз за час життя застосунку.
    """
    try:
        return YOLO(WEIGHTS)
    except Exception as exc:
        raise NotImplementedError("load_model ще не реалізовано")


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE):
    """Знайти обʼєкти на зображенні.

    Приймає байти завантаженого файлу, повертає структурований результат:
    для кожного знайденого обʼєкта — клас, рамку й упевненість, а також
    їхню кількість і час виконання. Точний склад полів — ваше рішення.
    """
    if not image_bytes:
        raise DetectionError("Файл порожній.")

    if not 0 <= confidence <= 1:
        raise DetectionError("Поріг confidence має бути від 0 до 1.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DetectionError("Файл не є коректним зображенням.") from exc

    model = load_model()

    try:
        started = time.perf_counter()

        results = model(
            image,
            conf=confidence,
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

    except Exception as exc:
        raise DetectionError("Помилка під час виконання детекції.") from exc

    detections = []

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            box_confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(box_confidence, 4),
                    "bbox": {
                        "x1": round(coordinates[0], 2),
                        "y1": round(coordinates[1], 2),
                        "x2": round(coordinates[2], 2),
                        "y2": round(coordinates[3], 2),
                    },
                }
            )

    return {
        "count": len(detections),
        "inference_time_ms": round(elapsed_ms, 2),
        "confidence_threshold": confidence,
        "detections": detections,
    }
