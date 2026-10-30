"""Детерміновані перевірки змісту рахунку."""
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import date
import json
import re

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
CENT = Decimal("0.01")

@dataclass
class Issue:
    field: str
    rule: str
    message: str
    severity: str = "error"

def load_reference():
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    if isinstance(suppliers, dict):
        suppliers = suppliers.get("suppliers", [])
    return company, suppliers

def money(value):
    if value is None:
        return None
    return Decimal(str(value).replace(" ", "").replace(",", ".")).quantize(CENT, rounding=ROUND_HALF_UP)

def digits(value):
    return re.sub(r"\D", "", str(value or ""))

def iban_normalize(value):
    return re.sub(r"\s+", "", str(value or "")).upper()

def iban_valid(value):
    iban = iban_normalize(value)
    if not re.fullmatch(r"UA\d{27}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    remainder = 0
    for char in numeric:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1

def code_valid(value):
    code = digits(value)
    if len(code) == 8:
        nums = list(map(int, code))
        # Український ЄДРПОУ: стандартні ваги; для залишку 10 застосовується
        # альтернативний набір ваг.
        weights = [1, 2, 3, 4, 5, 6, 7]
        check = sum(a*b for a, b in zip(nums[:7], weights)) % 11
        if check == 10:
            weights = [3, 4, 5, 6, 7, 8, 9]
            check = sum(a*b for a, b in zip(nums[:7], weights)) % 11
        return check == nums[7]
    if len(code) == 10:
        nums = list(map(int, code))
        check = sum(nums[i] * (i + 1) for i in range(9)) % 11
        return check % 10 == nums[9]
    return False

def _first(d, *keys):
    for key in keys:
        if key in d:
            return d[key]
    return None

def _norm_name(value):
    return re.sub(r"\s+", " ", str(value or "").replace("’", "'").replace("ʼ", "'").replace("`", "'")).strip().casefold()

def check(document: dict) -> list[Issue]:
    issues = []
    def add(field, rule, msg, severity="error"):
        issues.append(Issue(field, rule, msg, severity))

    if document.get("document_type") == "other":
        add("document_type", "document_type", "Це не рахунок на оплату.", "error")
        return issues
    if document.get("document_type") == "unknown":
        add("document_type", "document_type", "Не вдалося визначити тип документа.")
    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}

    required = [
        ("invoice_number", document.get("invoice_number")),
        ("date", document.get("date")),
        ("supplier.name", supplier.get("name")),
        ("supplier.code", supplier.get("code")),
        ("supplier.iban", supplier.get("iban")),
        ("buyer.name", buyer.get("name")),
        ("buyer.code", buyer.get("code")),
        ("subtotal", document.get("subtotal")),
        ("vat", document.get("vat")),
        ("total", document.get("total")),
    ]
    for field, value in required:
        if value is None or value == "":
            add(field, "missing_required", f"Обов'язкове поле {field} не прочитано.")

    iban = supplier.get("iban")
    if iban:
        if not iban_valid(iban):
            add("supplier.iban", "iban_checksum", "IBAN має неправильний формат або контрольну суму.")
    for path, value in (("supplier.code", supplier.get("code")), ("buyer.code", buyer.get("code"))):
        if value and not code_valid(value):
            add(path, "code_checksum", "Код ЄДРПОУ/РНОКПП не пройшов перевірку контрольного розряду.")

    try:
        today = date.today()
        doc_date = date.fromisoformat(document["date"]) if document.get("date") else None
        valid_until = date.fromisoformat(document["valid_until"]) if document.get("valid_until") else None
        if doc_date and doc_date > today:
            add("date", "date_future", "Дата рахунку в майбутньому.")
        if doc_date and valid_until and valid_until < doc_date:
            add("valid_until", "validity_date", "Строк дії раніше дати рахунку.")
        if valid_until and valid_until < today:
            add("valid_until", "expired", "Строк дії рахунку минув.", "warning")
    except ValueError:
        add("date", "date_format", "Дата має некоректний формат.")

    try:
        company, suppliers = load_reference()
        company_name = _first(company, "name", "company_name", "legal_name")
        company_code = _first(company, "code", "edrpou", "tax_id")
        if buyer.get("name") and company_name and _norm_name(buyer["name"]) != _norm_name(company_name):
            add("buyer.name", "buyer", "Рахунок виставлено не на нашу компанію.")
        if buyer.get("code") and company_code and digits(buyer["code"]) != digits(company_code):
            add("buyer.code", "buyer", "Код покупця не збігається з реквізитами компанії.")
        matched = None
        for item in suppliers:
            nm = _first(item, "name", "supplier_name")
            cd = _first(item, "code", "edrpou", "tax_id")
            if (supplier.get("code") and cd and digits(supplier["code"]) == digits(cd)) or (
                supplier.get("name") and nm and _norm_name(supplier["name"]) == _norm_name(nm)
            ):
                matched = item
                break
        if supplier.get("name") or supplier.get("code"):
            if not matched:
                add("supplier.name", "supplier_registry", "Постачальника немає в довіднику.")
            else:
                known_iban = _first(matched, "iban", "account")
                if supplier.get("iban") and known_iban and iban_normalize(supplier["iban"]) != iban_normalize(known_iban):
                    add("supplier.iban", "iban_registry", "IBAN не збігається з довідником постачальників.")
    except (OSError, ValueError, TypeError):
        add("supplier", "reference_unavailable", "Не вдалося прочитати довідники reference/.", "warning")

    items = document.get("items") or []
    for i, item in enumerate(items):
        try:
            q, p, a = money(item.get("quantity")), money(item.get("price")), money(item.get("amount"))
            if q is not None and p is not None and a is not None:
                if abs(q * p - a) > CENT:
                    add(f"items.{i}.amount", "arithmetic_line", "Кількість × ціна не дорівнює сумі рядка.")
        except (InvalidOperation, TypeError):
            add(f"items.{i}", "arithmetic_line", "Не вдалося перевірити арифметику позиції.")

    try:
        amounts = [money(x.get("amount")) for x in items if x.get("amount") is not None]
        subtotal = money(document.get("subtotal"))
        vat = money(document.get("vat"))
        total = money(document.get("total"))
        if amounts and subtotal is not None and abs(sum(amounts, Decimal("0")) - subtotal) > CENT:
            add("subtotal", "arithmetic_subtotal", "Сума позицій не дорівнює підсумку без ПДВ.")
        if subtotal is not None and vat is not None and total is not None:
            expected_vat = (subtotal * Decimal("0.20")).quantize(CENT, rounding=ROUND_HALF_UP)
            if abs(vat - expected_vat) > CENT and vat != Decimal("0.00"):
                add("vat", "vat_rate", "ПДВ не дорівнює 20% від суми без ПДВ і не є нульовим.")
            if abs(subtotal + vat - total) > CENT:
                add("total", "arithmetic_total", "Підсумок без ПДВ + ПДВ не дорівнює сумі до сплати.")
    except (InvalidOperation, TypeError):
        add("subtotal", "arithmetic", "Не вдалося перевірити підсумкову арифметику.")
    return issues
