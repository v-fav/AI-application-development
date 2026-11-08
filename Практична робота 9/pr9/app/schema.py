"""Схема структурованого розбору звернення клієнта."""
import json
import re

from jsonschema import Draft202012Validator


_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other", "unclear", "multiple"],
        },
        "order_id": {"type": ["string", "null"], "maxLength": 32},
        "item": {"type": ["string", "null"], "maxLength": 120},
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", None],
        },
        "quantity": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
    },
    "required": ["kind", "order_id", "item", "reason", "quantity"],
    "additionalProperties": False,
}


_VALIDATOR = Draft202012Validator(_SCHEMA)


def request_schema() -> dict:
    return _SCHEMA


def validate(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Модель повернула порожню відповідь замість JSON.")

    text = raw.strip()
    # Прибираємо типову Markdown-обгортку, якщо модель її додала.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Модель повернула некоректний JSON (позиція {exc.pos})."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Результат розбору повинен бути JSON-об'єктом.")

    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise ValueError(f"Результат розбору не пройшов перевірку схеми: {details}")

    if data["order_id"] is not None:
        data["order_id"] = data["order_id"].strip() or None
    if data["item"] is not None:
        data["item"] = data["item"].strip() or None

    return data
