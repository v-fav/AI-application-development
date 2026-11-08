"""Цикл виклику моделі та інструментів з ПР8 (агентний режим для порівняння)."""
import json
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from . import llm, tools

load_dotenv()
MAX_ROUNDS = max(1, int(os.getenv("TOOL_MAX_ROUNDS", "3")))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    if not isinstance(customer_id, str) or not customer_id:
        raise llm.LLMError("Не визначено клієнта сеансу.")
    messages = llm.build_messages(question)
    ctx = tools.Context(customer_id=customer_id)
    traces = []
    allowed_tools = {item["function"]["name"] for item in tools.specs()}
    model_time = tool_time = 0.0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    model_name = llm.MODEL
    last_text = ""
    rounds = 0
    stopped = "answer"
    for round_no in range(1, MAX_ROUNDS + 1):
        response = llm.chat(messages, tools.specs(), "auto")
        rounds += 1
        model_time += response["elapsed"]
        model_name = response.get("model") or model_name
        for key in usage:
            usage[key] += (response.get("usage") or {}).get(key, 0)
        message = response["message"]
        tool_calls = message.get("tool_calls") or []
        last_text = message.get("content") or ""
        if not tool_calls:
            return Answer(last_text.strip() or "Не вдалося сформувати відповідь. Спробуйте уточнити запит.", traces,
                          rounds, "answer", model_name,
                          {"model": round(model_time, 3), "tools": round(tool_time, 3)}, usage)
        messages.append(message)
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw = fn.get("arguments", "")
            start = time.perf_counter()
            result = tools.call(name, raw, ctx, allowed_tools=allowed_tools)
            elapsed = time.perf_counter() - start
            tool_time += elapsed
            traces.append(ToolTrace(round=round_no, name=name, arguments=raw, status=result.status,
                                    reason=result.reason, result=result.content, elapsed=round(elapsed, 4)))
            payload = {"status": result.status, "data": result.content, "reason": result.reason}
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": json.dumps(payload, ensure_ascii=False, default=str)})
        if round_no == MAX_ROUNDS:
            stopped = "limit"
            last_text = "Я отримав частину результатів, але не встиг завершити перевірку. Будь ласка, повторіть запит."
            break
    return Answer(last_text, traces, rounds, stopped, model_name,
                  {"model": round(model_time, 3), "tools": round(tool_time, 3)}, usage)
