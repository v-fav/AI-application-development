"""Схема структурованої відповіді помічника."""

import json

from jsonschema import ValidationError, validate as jsonschema_validate


def output_schema() -> dict:
    """Повернути JSON Schema відповіді помічника."""
    return {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "Текст відповіді клієнту українською мовою.",
            },
            "topic": {
                "type": "string",
                "enum": [
                    "замовлення",
                    "доставка",
                    "оплата",
                    "повернення",
                    "гарантія",
                    "підтримка",
                    "інше",
                ],
                "description": "Тема звернення.",
            },
            "rules_based": {
                "type": "boolean",
                "description": (
                    "true, якщо відповідь безпосередньо ґрунтується "
                    "на правилах із context.md."
                ),
            },
            "needs_clarification": {
                "type": "boolean",
                "description": "Чи потрібно отримати від клієнта уточнення.",
            },
            "escalate_to_operator": {
                "type": "boolean",
                "description": "Чи потрібно передати звернення оператору.",
            },
            "order_number": {
                "type": ["string", "null"],
                "pattern": "^[0-9]{6}$",
                "description": "Шестизначний номер замовлення або null.",
            },
        },
        "required": [
            "reply",
            "topic",
            "rules_based",
            "needs_clarification",
            "escalate_to_operator",
            "order_number",
        ],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити відповідь моделі та повернути структурований об'єкт."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Відповідь моделі не є коректним JSON: {exc}"
        ) from exc

    try:
        jsonschema_validate(instance=data, schema=output_schema())
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        location = f"поле '{path}'" if path else "об'єкт"
        raise ValueError(
            f"Відповідь не відповідає JSON Schema ({location}): {exc.message}"
        ) from exc

    return data