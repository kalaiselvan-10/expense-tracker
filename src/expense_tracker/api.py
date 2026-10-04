"""FastAPI layer: the same results as the dashboard, as JSON.

Run:  uvicorn expense_tracker.api:app --reload      (docs at /docs)
"""

import json
import re

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .analysis import (
    budget_vs_actual,
    category_shares,
    load_budgets,
    monthly_summary,
    recurring_expenses,
)
from .categorizer import add_correction
from .config import setup_logging
from .forecast import backtest, backtest_metrics, forecast_next_month, forecast_total, next_month_label
from .pipeline import build_dataset, load_metrics

setup_logging()
app = FastAPI(title="Expense Tracker API", version=__version__)


class Correction(BaseModel):
    description: str
    category: str


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (handles dates and numpy types)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def check_month(df: pd.DataFrame, month: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=400, detail="Month must look like 2026-09")
    if month not in set(df["date"].dt.to_period("M").astype(str)):
        raise HTTPException(status_code=404, detail=f"No transactions in {month}")


@app.get("/")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/transactions")
def transactions(month: str | None = None):
    df = build_dataset()
    if month:
        check_month(df, month)
        df = df[df["date"].dt.to_period("M").astype(str) == month]
    return records(df)


@app.get("/summary")
def summary():
    return records(monthly_summary(build_dataset()))


@app.get("/categories")
def categories(month: str | None = None):
    df = build_dataset()
    if month:
        check_month(df, month)
    return records(category_shares(df, month))


@app.get("/recurring")
def recurring():
    return records(recurring_expenses(build_dataset()))


@app.get("/budget/{month}")
def budget(month: str):
    df = build_dataset()
    check_month(df, month)
    return records(budget_vs_actual(df, load_budgets(), month))


@app.get("/forecast")
def forecast():
    df = build_dataset()
    return {
        "month": next_month_label(df),
        "predicted_total": round(forecast_total(df), 2),
        "by_category": records(forecast_next_month(df)),
    }


@app.get("/backtest")
def backtest_endpoint():
    result = backtest(build_dataset())
    return {"metrics": backtest_metrics(result), "months": records(result)}


@app.get("/model")
def model_metrics():
    metrics = load_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="Model not trained yet. Run: expense-tracker train")
    return metrics


@app.post("/corrections")
def save_correction(correction: Correction):
    add_correction(correction.description, correction.category)
    return {"saved": correction.model_dump()}
