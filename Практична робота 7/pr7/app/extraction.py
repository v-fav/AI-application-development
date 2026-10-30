"""Послідовний конвеєр: підготовка >> модель >> правила >> рішення."""
from dataclasses import dataclass, field
from time import perf_counter
from . import images, llm, rules

@dataclass
class Result:
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[rules.Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None
    raw: str | None = None

def decide(document, issues):
    if not document or document.get("document_type") == "other":
        return "reject", ["Документ не є рахунком на оплату."]
    reasons = []
    critical = {"supplier.iban", "supplier.code", "buyer.code", "total", "document_type"}
    blocking = [x for x in issues if x.severity == "error"]
    if any(x.field in critical or x.rule in {"iban_checksum", "iban_registry", "buyer", "supplier_registry"} for x in blocking):
        reasons.extend(x.message for x in blocking)
        return "review", reasons or ["Є критичні проблеми."]
    if blocking:
        reasons.extend(x.message for x in blocking)
        return "review", reasons
    if any(document.get(k) is None for k in ("supplier", "buyer", "subtotal", "vat", "total")):
        return "review", ["Не всі обов'язкові дані прочитано."]
    return "auto", ["Критичні поля присутні, перевірки пройдено."]

def process(content: bytes) -> Result:
    elapsed = {}
    t = perf_counter()
    prepared = images.prepare(content)
    elapsed["prepare"] = perf_counter() - t
    image_info = {"original": prepared.original, "sent": prepared.sent}

    t = perf_counter()
    extracted = llm.extract(prepared)
    elapsed["extraction"] = perf_counter() - t

    t = perf_counter()
    issues = rules.check(extracted["document"])
    elapsed["checks"] = perf_counter() - t
    decision, reasons = decide(extracted["document"], issues)
    return Result(
        decision=decision, reasons=reasons, document=extracted["document"],
        issues=issues, image=image_info, model=extracted["model"],
        elapsed=elapsed, usage=extracted["usage"], raw=extracted.get("raw"),
    )
