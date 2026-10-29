"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Це єдине місце, яке знає всі кроки послідовно: знайти фрагменти
(`retrieval.retrieve`), зібрати з них контекст (`retrieval.build_context`),
спитати модель (`llm.ask`), перевірити те, що вона повернула, і скласти
результат для веб-рівня. Модулі-учасники одне про одного не знають:
пошук не знає про модель, модель — про індекс, веб-рівень — ні про що з
цього.

Функція `answer` — заготовка. Порядок кроків очевидний; рішення — ні:

* що робити, коли пошук не повернув нічого придатного: не викликати
  модель і відповісти наперед заданим текстом, чи викликати все одно —
  і чому;
* чи довіряти полю «відповідь знайдено», яке заповнила сама модель, і
  чим його перевірити: наприклад, чи існують названі нею номери джерел
  серед показаних, чи не порожній їх перелік;
* що показувати як джерела: лише ті фрагменти, на які модель послалася,
  чи всі, які їй показали;
* що потрапляє в `retrieved` — те, що бачила модель, — і чи віддавати
  це користувачеві чи лише вам для налагодження;
* що рахувати окремо: час пошуку і час генерації — це різні витрати, і
  на сторінці вони показуються окремо.
"""


import time
from dataclasses import dataclass, field

from . import llm, retrieval
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source


@dataclass
class Answer:
    """Результат конвеєра для веб-рівня."""

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Знайти контекст, отримати відповідь і перевірити посилання."""
    question = question.strip()
    if not question:
        raise ValueError("Питання не може бути порожнім")

    started = time.perf_counter()
    hits = retrieval.retrieve(
        question,
        index,
        keyword_index,
        filters=filters,
    )
    retrieval_time = round(time.perf_counter() - started, 3)

    context, sources = retrieval.build_context(hits)

    if not sources or not context.strip():
        return Answer(
            text="У наявній базі знань немає достатньої інформації для відповіді.",
            found=False,
            sources=[],
            retrieved=[],
            elapsed={"retrieval": retrieval_time, "generation": 0.0},
            usage=None,
        )

    generation_started = time.perf_counter()

    try:
        result = llm.ask(question, context)
    except llm.LLMError:
        raise
    except Exception as exc:
        raise llm.LLMError(
            f"Не вдалося отримати відповідь: {type(exc).__name__}"
        ) from exc

    generation_time = round(time.perf_counter() - generation_started, 3)

    # Модель може послатися лише на номери, які реально були в контексті.
    sources_by_ref = {source.ref: source for source in sources}
    refs = result["source_refs"]

    invalid_refs = [ref for ref in refs if ref not in sources_by_ref]
    if invalid_refs:
        raise llm.LLMError(
            "Модель послалася на джерела, яких не було в контексті."
        )

    found = bool(result["found"])
    if found and not refs:
        raise llm.LLMError("Модель не вказала джерела для знайденої відповіді.")

    cited_sources = [sources_by_ref[ref] for ref in refs] if found else []
    text = result["answer"].strip()

    # Перевіряємо, що номери з JSON також присутні в тексті відповіді.
    if found:
        for ref in refs:
            citation = f"[{ref}]"
            if citation not in text:
                text += f" {citation}"

    return Answer(
        text=text,
        found=found,
        sources=cited_sources,
        retrieved=hits,
        model=result.get("model"),
        elapsed={
            "retrieval": retrieval_time,
            "generation": generation_time,
        },
        usage=result.get("usage"),
    )