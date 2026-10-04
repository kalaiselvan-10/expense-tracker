"""Monthly totals, category shares, recurring expenses and budget variance."""

import json
import logging
from pathlib import Path

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)


def load_budgets(path: str | Path | None = None) -> dict[str, float]:
    """Monthly budget per category."""
    path = path or load_config()["paths"]["budgets"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(month=df["date"].dt.to_period("M").astype(str))


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Income, expenses, net savings and savings rate per month."""
    data = add_month(df)
    income = data[data["amount"] > 0].groupby("month")["amount"].sum()
    expenses = data[data["amount"] < 0].groupby("month")["amount"].sum().abs()
    summary = pd.DataFrame({"income": income, "expenses": expenses}).fillna(0)
    summary["net"] = summary["income"] - summary["expenses"]
    summary["savings_rate_pct"] = (
        (summary["net"] / summary["income"].where(summary["income"] > 0) * 100).fillna(0).round(1)
    )
    return summary.sort_index().reset_index()


def spending_by_category(df: pd.DataFrame, month: str | None = None) -> pd.DataFrame:
    """Total spending per category, largest first (expenses only). Optional month filter."""
    data = add_month(df)
    if month:
        data = data[data["month"] == month]
    spent = data[data["amount"] < 0]
    result = spent.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
    return result.rename("spent").reset_index()


def category_shares(df: pd.DataFrame, month: str | None = None) -> pd.DataFrame:
    """Spending per category with each category's percentage share."""
    result = spending_by_category(df, month)
    total = result["spent"].sum()
    result["share_pct"] = (result["spent"] / total * 100).round(1) if total else 0.0
    return result


def recurring_expenses(
    df: pd.DataFrame, min_months: int | None = None, tolerance: float | None = None
) -> pd.DataFrame:
    """Merchants charged in several months with a similar amount (e.g. Netflix, rent)."""
    config = load_config()["recurring"]
    min_months = config["min_months"] if min_months is None else min_months
    tolerance = config["tolerance"] if tolerance is None else tolerance

    spent = add_month(df[df["amount"] < 0])
    grouped = spent.groupby("description").agg(
        months=("month", "nunique"),
        typical_amount=("amount", "mean"),
        lowest=("amount", "min"),
        highest=("amount", "max"),
        last_seen=("date", "max"),
    )
    grouped = grouped[grouped["months"] >= min_months]
    spread = (grouped["highest"] - grouped["lowest"]).abs() / grouped["typical_amount"].abs()
    grouped = grouped[spread <= tolerance].copy()
    grouped["typical_amount"] = grouped["typical_amount"].abs().round(2)
    grouped["last_seen"] = grouped["last_seen"].dt.strftime("%Y-%m-%d")
    return (
        grouped[["months", "typical_amount", "last_seen"]]
        .sort_values("typical_amount", ascending=False)
        .reset_index()
    )


def budget_vs_actual(df: pd.DataFrame, budgets: dict[str, float], month: str) -> pd.DataFrame:
    """Budget variance for one month ('YYYY-MM'). Positive variance = under budget."""
    data = add_month(df)
    spent = data[(data["amount"] < 0) & (data["month"] == month)]
    actual = spent.groupby("category")["amount"].sum().abs().rename("spent")

    result = pd.DataFrame({"budget": pd.Series(budgets, dtype=float)}).join(actual, how="outer")
    result = result.fillna({"budget": 0, "spent": 0})
    result["variance"] = result["budget"] - result["spent"]
    result["used_pct"] = (
        (result["spent"] / result["budget"].where(result["budget"] > 0) * 100).round(1)
    )
    status = pd.Series("OK", index=result.index)
    status[result["variance"] < 0] = "Over"
    status[(result["budget"] == 0) & (result["spent"] > 0)] = "No budget"
    result["status"] = status
    return result.sort_values("variance").reset_index(names="category")
