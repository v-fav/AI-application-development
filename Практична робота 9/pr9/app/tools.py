"""Безпечний шлюз між workflow/моделлю та сервісом магазину."""
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from jsonschema import Draft202012Validator
from shop import service


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


class PolicyError(Exception):
    """Запит зрозумілий, але правила магазину забороняють дію."""


class ServiceUnavailableError(Exception):
    """Внутрішній сервіс магазину тимчасово недоступний."""


def _schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


STR_ID = {"type": "string", "minLength": 1, "maxLength": 32}
SKU = {"type": "string", "minLength": 2, "maxLength": 24}
TOOLS = [
    ("list_orders", "Показати всі замовлення поточного клієнта.", _schema({})),
    ("get_order", "Отримати безпечні відомості про конкретне замовлення поточного клієнта.", _schema({"order_id": STR_ID}, ["order_id"])),
    ("search_products", "Шукати товари в каталозі.", _schema({"query": {"type": "string", "maxLength": 120}, "category": {"type": ["string", "null"], "maxLength": 60}, "max_price": {"type": ["number", "null"], "minimum": 0, "maximum": 1000000}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"])),
    ("get_product", "Отримати характеристики товару за SKU.", _schema({"sku": SKU}, ["sku"])),
    ("get_stock", "Перевірити залишок товару за SKU.", _schema({"sku": SKU}, ["sku"])),
    ("delivery_quote", "Розрахувати доставку.", _schema({"city": {"type": "string", "minLength": 1, "maxLength": 80}, "method": {"type": "string", "enum": ["branch", "courier", "pickup"]}, "items": {"type": "array", "minItems": 1, "maxItems": 10, "items": _schema({"sku": SKU, "quantity": {"type": "integer", "minimum": 1, "maximum": 50}}, ["sku", "quantity"])}}, ["city", "method", "items"])),
    ("create_return", "Створити заявку на повернення товару з отриманого замовлення клієнта. Використовуй лише коли клієнт прямо просить оформити повернення. reason: not_suitable — якісний товар не підійшов; defect — дефект; wrong_item — надіслали не той товар; damaged — пошкоджено доставкою. Сервіс перевіряє правила повернення. Це заявка, а не миттєве повернення коштів.", _schema({"order_id": STR_ID, "sku": SKU, "reason": {"type": "string", "enum": ["not_suitable", "defect", "wrong_item", "damaged"]}, "quantity": {"type": "integer", "minimum": 1, "maximum": 50}, "comment": {"type": "string", "maxLength": 300}}, ["order_id", "sku", "reason"])),
    ("list_returns", "Перевірити заявки на повернення конкретного замовлення.", _schema({"order_id": STR_ID}, ["order_id"])),
    ("cancel_order", "Скасувати замовлення у статусі «Готується». Для клієнта доступно лише після підтвердження оператора у workflow.", _schema({"order_id": STR_ID}, ["order_id"])),
]
_SPEC = [{"type": "function", "function": {"name": n, "description": d, "parameters": sc}} for n, d, sc in TOOLS]
_SCHEMAS = {x["function"]["name"]: x["function"]["parameters"] for x in _SPEC}
# Агент отримує набір інструментів ПР8. cancel_order доступний лише коду workflow.
_AGENT_SPEC = [x for x in _SPEC if x["function"]["name"] != "cancel_order"]


def specs() -> list[dict]:
    return _AGENT_SPEC


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _safe_order(order):
    delivery = order.get("delivery") or {}
    return {"order_id": order["order_id"], "created_at": order["created_at"], "status": order["status"],
            "status_label": order.get("status_label", order["status"]),
            "items": [{"sku": i["sku"], "name": i["name"], "quantity": i["quantity"], "price": i["price"]} for i in order.get("items", [])],
            "total": order["total"], "delivery": {k: delivery.get(k) for k in ("method", "carrier", "city", "point", "tracking", "delivered_at", "cost")}}


def _owned_order(order_id, ctx):
    normalized = re.sub(r"\s+", "", str(order_id))
    order = service.get_order(normalized)
    if order.get("customer_id") != ctx.customer_id:
        raise PermissionError("Це замовлення не належить поточному клієнтові.")
    return order


def customer_exists(customer_id: str) -> bool:
    """Перевірити ідентифікатор клієнта, отриманий веб-рівнем із сеансу."""
    return any(c.get("customer_id") == customer_id for c in service.list_customers())


def check_return_eligibility(order_id: str, sku: str, reason: str, quantity: int, ctx: Context) -> dict:
    """Перевірка правил без створення заявки; повторюється перед виконанням."""
    try:
        order = _owned_order(order_id, ctx)
        line = next((x for x in order.get("items", []) if x.get("sku") == sku), None)
        if line is None:
            raise service.InvalidRequest(f"Товар {sku} відсутній у замовленні {order_id}.")
        if quantity < 1 or quantity > int(line["quantity"]):
            raise service.InvalidRequest(f"У замовленні доступно {line['quantity']} шт. цього товару.")
        if order["status"] != "delivered":
            raise service.PolicyViolation("Повернення можливе лише після отримання замовлення.")
        product = service.get_product(sku)
        delivered = date.fromisoformat(order["delivery"]["delivered_at"])
        days = (service.today() - delivered).days
        if reason == "not_suitable":
            if not product.get("returnable"):
                raise service.PolicyViolation("Цей товар належної якості не підлягає поверненню.")
            if days > service.RETURN_WINDOW_DAYS:
                raise service.PolicyViolation(f"Строк повернення минув: від отримання минуло {days} дн.")
        elif reason == "defect":
            if service.today() > delivered + timedelta(days=30 * int(product.get("warranty_months", 0))):
                raise service.PolicyViolation("Гарантійний строк минув.")
        return {"allowed": True, "days_from_delivery": days, "status": order["status"],
                "product": product.get("name", sku)}
    except service.Unavailable as exc:
        raise ServiceUnavailableError("Сервіс магазину тимчасово недоступний. Спробуйте пізніше.") from exc
    except service.PolicyViolation as exc:
        raise PolicyError(str(exc)) from exc
    except service.ShopError as exc:
        raise ValueError(str(exc)) from exc


def call(name: str, raw_arguments: str, ctx: Context, allowed_tools=None) -> ToolResult:
    args = None
    try:
        if allowed_tools is not None and name not in set(allowed_tools):
            return ToolResult("rejected", {"error": "tool_not_allowed"},
                              f"Інструмент {name} недоступний для цього режиму.")
        if name not in _SCHEMAS:
            return ToolResult("rejected", {"error": "unknown_tool"}, f"Невідомий інструмент: {name}")
        if not isinstance(raw_arguments, str) or len(raw_arguments) > 12000:
            return ToolResult("rejected", {"error": "invalid_arguments"}, "Аргументи мають бути коротким JSON-об'єктом.")
        try:
            args = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return ToolResult("rejected", {"error": "invalid_json"}, "Аргументи не є коректним JSON.")
        errors = sorted(Draft202012Validator(_SCHEMAS[name]).iter_errors(args), key=lambda e: list(e.path))
        if errors:
            reason = "; ".join(e.message for e in errors[:3])
            return ToolResult("rejected", {"error": "schema_validation", "details": reason}, f"Аргументи не пройшли перевірку схеми: {reason}", args if isinstance(args, dict) else None)
        if name == "list_orders":
            result = service.list_orders(ctx.customer_id)
        elif name == "get_order":
            result = _safe_order(_owned_order(args["order_id"], ctx))
        elif name == "search_products":
            result = service.search_products(args["query"], args.get("category"), args.get("max_price"), args.get("limit", 8))
        elif name == "get_product":
            p = service.get_product(args["sku"])
            result = {k: p[k] for k in ("sku", "name", "category", "price", "weight_kg", "warranty_months", "returnable", "bulky", "description") if k in p}
        elif name == "get_stock":
            result = service.get_stock(args["sku"])
        elif name == "delivery_quote":
            result = service.delivery_quote(args["city"], args["method"], args["items"])
        elif name == "create_return":
            _owned_order(args["order_id"], ctx)
            result = service.create_return(args["order_id"], args["sku"], args["reason"], args.get("quantity", 1), args.get("comment", ""))
            result = {k: result[k] for k in ("return_id", "order_id", "sku", "quantity", "reason", "created_at", "status", "next_steps")}
        elif name == "list_returns":
            _owned_order(args["order_id"], ctx)
            result = service.list_returns(args["order_id"])
            result = [{k: r[k] for k in ("return_id", "order_id", "sku", "quantity", "reason", "created_at", "status", "next_steps")} for r in result]
        elif name == "cancel_order":
            _owned_order(args["order_id"], ctx)
            result = service.cancel_order(args["order_id"])
            result = {k: result[k] for k in ("order_id", "status", "status_label", "total") if k in result}
        return ToolResult("ok", _json_safe(result), arguments=args)
    except PermissionError as exc:
        return ToolResult("rejected", {"error": "forbidden"}, str(exc), args)
    except service.Unavailable:
        return ToolResult("error", {"error": "unavailable"}, "Сервіс магазину тимчасово недоступний. Спробуйте пізніше.", args)
    except service.ShopError as exc:
        return ToolResult("error", {"error": exc.code, "message": str(exc)}, str(exc), args)
    except Exception:
        return ToolResult("error", {"error": "internal_error"}, "Не вдалося безпечно виконати операцію.", args)


def list_customers() -> list[dict]:
    """Список клієнтів для навчального перемикача сеансів."""
    return service.list_customers()


def reset_shop() -> None:
    """Повернути тестові дані магазину до початкового стану."""
    service.reset()
