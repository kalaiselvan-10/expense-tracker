"""Moving-average spending forecast, with a walk-forward backtest to measure its error."""

import logging

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)


def monthly_category_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Expenses per month (rows) and category (columns)."""
    spent = df[df["amount"] < 0].copy()
    spent["month"] = spent["date"].dt.to_period("M")
    table = spent.pivot_table(
        index="month", columns="category", values="amount", aggfunc="sum", fill_value=0
    )
    return table.abs().sort_index()


def monthly_expenses(df: pd.DataFrame) -> pd.Series:
    """Total expenses per month."""
    return monthly_category_spend(df).sum(axis=1)


def _window(window: int | None) -> int:
    return load_config()["forecast"]["window"] if window is None else window


def forecast_next_month(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Predict next month's spend per category: mean of the last `window` months."""
    window = _window(window)
    recent = monthly_category_spend(df).tail(window)
    result = recent.mean().round(2).rename("forecast").sort_values(ascending=False).reset_index()
    result["based_on_months"] = len(recent)
    return result


def forecast_total(df: pd.DataFrame, window: int | None = None) -> float:
    return float(forecast_next_month(df, window)["forecast"].sum())


def next_month_label(df: pd.DataFrame) -> str:
    return str(df["date"].max().to_period("M") + 1)


def backtest(df: pd.DataFrame, window: int | None = None, min_train: int = 3) -> pd.DataFrame:
    """Predict each month's total expenses using only earlier months.

    Compares the moving average with a naive baseline (same as last month).
    """
    window = _window(window)
    totals = monthly_expenses(df)
    rows = []
    for i in range(min_train, len(totals)):
        history = totals.iloc[:i]
        rows.append({
            "month": str(totals.index[i]),
            "actual": float(totals.iloc[i]),
            "moving_avg": float(history.tail(window).mean()),
            "naive": float(history.iloc[-1]),
        })
    return pd.DataFrame(rows)


def backtest_metrics(result: pd.DataFrame) -> dict[str, float]:
    """MAE (average error in rupees) and MAPE (average error in %) for both methods."""
    metrics = {"months_tested": int(len(result))}
    if result.empty:
        return metrics
    for name in ("moving_avg", "naive"):
        error = (result[name] - result["actual"]).abs()
        metrics[f"{name}_mae"] = round(float(error.mean()), 2)
        metrics[f"{name}_mape_pct"] = round(float((error / result["actual"]).mean() * 100), 2)
    return metrics
