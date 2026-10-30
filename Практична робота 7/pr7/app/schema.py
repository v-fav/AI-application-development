"""Структурний контракт JSON-відповіді моделі."""
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation

class SchemaError(ValueError):
    pass

def output_schema() -> dict:
    nullable_string = {"type": ["string", "null"]}
    nullable_number = {"type": ["string", "number", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {"type": "string", "enum": ["invoice", "other", "unknown"]},
            "invoice_number": nullable_string,
            "date": nullable_string,
            "valid_until": nullable_string,
            "supplier": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": nullable_string, "code": nullable_string, "iban": nullable_string},
                "required": ["name", "code", "iban"],
            },
            "buyer": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": nullable_string, "code": nullable_string},
                "required": ["name", "code"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "name": nullable_string, "unit": nullable_string,
                        "quantity": nullable_number, "price": nullable_number,
                        "amount": nullable_number,
                    },
                    "required": ["name", "unit", "quantity", "price", "amount"],
                },
            },
            "subtotal": nullable_number,
            "vat": nullable_number,
            "total": nullable_number,
        },
        "required": ["document_type", "invoice_number", "date", "valid_until",
                     "supplier", "buyer", "items", "subtotal", "vat", "total"],
    }

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    return raw.strip()

def _valid_date(value):
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        raise SchemaError(f"Некоректна дата: {value!r}. Очікується РРРР-ММ-ДД.")

def _valid_number(value, field):
    if value is None:
        return
    try:
        Decimal(str(value).replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        raise SchemaError(f"Поле {field} має бути числом або null.")

def validate(raw: str) -> dict:
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Модель повернула невалідний JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("Корінь відповіді має бути JSON-об'єктом.")

    required = output_schema()["required"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SchemaError("У відповіді відсутні поля: " + ", ".join(missing))

    for key in ("supplier", "buyer"):
        if not isinstance(data[key], dict):
            raise SchemaError(f"{key} має бути об'єктом.")
        for field in (["name", "code", "iban"] if key == "supplier" else ["name", "code"]):
            if field not in data[key]:
                raise SchemaError(f"В об'єкті {key} відсутнє поле {field}.")

    if not isinstance(data["items"], list):
        raise SchemaError("items має бути списком.")
    for i, item in enumerate(data["items"]):
        if not isinstance(item, dict):
            raise SchemaError(f"Позиція {i+1} має бути об'єктом.")
        for field in ("name", "unit", "quantity", "price", "amount"):
            if field not in item:
                raise SchemaError(f"У позиції {i+1} відсутнє поле {field}.")
            if field in ("quantity", "price", "amount"):
                _valid_number(item[field], f"items.{i}.{field}")

    for field in ("subtotal", "vat", "total"):
        _valid_number(data[field], field)
    _valid_date(data["date"])
    _valid_date(data["valid_until"])
    if data["document_type"] not in ("invoice", "other", "unknown"):
        raise SchemaError("document_type має бути invoice, other або unknown.")
    return data
