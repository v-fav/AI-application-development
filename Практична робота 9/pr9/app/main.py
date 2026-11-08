"""FastAPI: веб-рівень не містить логіки workflow."""
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import workflow

app = FastAPI(title="Звернення клієнтів — ПР9")


@app.on_event("startup")
def recover_interrupted_runs():
    # Звернення, підтверджене оператором до аварійного перезапуску, відновлюється.
    workflow.recover_executing()


INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=32)
    text: str = Field(min_length=1, max_length=4000)
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    approve: bool
    comment: str = Field(default="", max_length=500)


def run_to_dict(run):
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result):
    return {"mode": "agent", "answer": result.text, "calls": [asdict(call) for call in result.calls],
            "rounds": result.rounds, "stopped": result.stopped, "model": result.model,
            "elapsed": result.elapsed, "usage": result.usage}


@app.get("/", response_class=HTMLResponse)
def page():
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers():
    return workflow.customers()


@app.post("/api/reset")
def api_reset():
    workflow.reset_shop()
    return {"ok": True}


@app.post("/api/requests")
def api_create(payload: RequestIn):
    if not payload.text.strip():
        raise HTTPException(400, "Текст звернення не може бути порожнім.")
    known = {c["customer_id"] for c in workflow.customers()}
    if payload.customer_id not in known:
        raise HTTPException(404, "Клієнта не знайдено.")
    try:
        if payload.mode == "agent":
            return answer_to_dict(workflow.agent_answer(payload.text, payload.customer_id))
        return run_to_dict(workflow.start(payload.text, payload.customer_id))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/runs")
def api_runs(status: str | None = None):
    return [run_to_dict(run) for run in workflow.list_runs(status)]


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    run = workflow.load_run(run_id)
    if run is None:
        raise HTTPException(404, "Звернення не знайдено.")
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn):
    try:
        return run_to_dict(workflow.resume(run_id, payload.approve, payload.comment))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
