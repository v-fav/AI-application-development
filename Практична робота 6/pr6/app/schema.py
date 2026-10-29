"""Контракт відповіді моделі: що саме вона має повернути і як це
перевіряється.

Окремий модуль, як у ПР4, і з тієї самої причини: схема — домовленість
між моделлю та рештою застосунку, і вона не залежить від провайдера.
`app/rag.py` працює лише з тим, що пройшло перевірку тут.

Що потрібно решті застосунку від відповіді (розділ 2 практичної роботи):

* текст відповіді для клієнта;
* на які фрагменти контексту відповідь спирається — номери з позначок,
  які ви поставили в `retrieval.build_context`;
* чи знайдено відповідь у наданих фрагментах, чи їх для відповіді не
  вистачило.

Як назвати поля, які з них обовʼязкові, чи потрібні ще якісь — ваше
рішення. Як і те, чим описати схему: JSON Schema вручну або
pydantic-модель. Реалізацію з ПР4 можна взяти за основу.

Сторінка каркаса бере текст відповіді з поля `answer` результату
`rag.answer`, а не з відповіді моделі напряму — тож назви полів у схемі
ні до чого не привʼязані, крім вашого коду.
"""


import json
import re


def output_schema() -> dict:
    """JSON Schema відповіді моделі."""
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Текст відповіді клієнту",
            },
            "source_refs": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "uniqueItems": True,
            },
            "found": {"type": "boolean"},
        },
        "required": ["answer", "source_refs", "found"],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити JSON та типи полів відповіді."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Модель повернула порожню відповідь")

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є коректним JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Відповідь має бути JSON-об'єктом")

    required = {"answer", "source_refs", "found"}
    if set(data) != required:
        raise ValueError(
            f"Очікуються лише поля {sorted(required)}"
        )

    if not isinstance(data["answer"], str) or not data["answer"].strip():
        raise ValueError("Поле answer має містити непорожній текст")

    if not isinstance(data["found"], bool):
        raise ValueError("Поле found має бути логічним значенням")

    refs = data["source_refs"]
    if not isinstance(refs, list):
        raise ValueError("Поле source_refs має бути списком")

    if any(type(ref) is not int or ref < 1 for ref in refs):
        raise ValueError("Номери джерел мають бути додатними цілими числами")

    if len(refs) != len(set(refs)):
        raise ValueError("Номери джерел не повинні повторюватися")

    if data["found"] and not refs:
        raise ValueError("Якщо відповідь знайдено, потрібно вказати джерела")

    if not data["found"] and refs:
        raise ValueError("Якщо відповідь не знайдено, список джерел має бути порожнім")

    return data