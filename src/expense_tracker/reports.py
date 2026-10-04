"""Charts, cleaned-data export and a Markdown report."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import (  # noqa: E402
    budget_vs_actual,
    category_shares,
    load_budgets,
    monthly_summary,
    recurring_expenses,
)
from .forecast import (  # noqa: E402
    backtest,
    backtest_metrics,
    forecast_next_month,
    forecast_total,
    next_month_label,
)

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = ["date", "description", "amount", "category", "account"]


def export_clean(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the cleaned dataset in the standard schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[SCHEMA_COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    logger.info("Exported %d cleaned rows to %s", len(out), path)
    return path


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def chart_monthly(df: pd.DataFrame, out_dir: Path) -> Path:
    summary = monthly_summary(df).set_index("month")[["income", "expenses"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary.plot.bar(ax=ax, color=["#2e7d32", "#c62828"])
    ax.set_title("Income vs expenses by month")
    ax.set_xlabel("")
    ax.set_ylabel("Rupees")
    return _save(fig, out_dir / "monthly_income_expenses.png")


def chart_categories(df: pd.DataFrame, month: str, out_dir: Path) -> Path:
    shares = category_shares(df, month).sort_values("spent")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(shares["category"], shares["spent"], color="#1565c0")
    for bar, pct in zip(bars, shares["share_pct"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {pct}%", va="center")
    ax.set_title(f"Spending by category, {month}")
    ax.set_xlabel("Rupees")
    return _save(fig, out_dir / "category_share.png")


def chart_budget(df: pd.DataFrame, month: str, out_dir: Path) -> Path:
    table = budget_vs_actual(df, load_budgets(), month).set_index("category")[["budget", "spent"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.plot.bar(ax=ax, color=["#9e9e9e", "#ef6c00"])
    ax.set_title(f"Budget vs actual, {month}")
    ax.set_xlabel("")
    ax.set_ylabel("Rupees")
    return _save(fig, out_dir / "budget_variance.png")


def chart_forecast(df: pd.DataFrame, out_dir: Path) -> Path:
    result = backtest(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not result.empty:
        ax.plot(result["month"], result["actual"], marker="o", label="Actual")
        ax.plot(result["month"], result["moving_avg"], marker="s", label="Moving average")
        ax.plot(result["month"], result["naive"], marker="^", linestyle="--", label="Naive (last month)")
    ax.scatter([next_month_label(df)], [forecast_total(df)], color="red", zorder=5, label="Forecast")
    ax.set_title("Forecast backtest: total monthly expenses")
    ax.set_ylabel("Rupees")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45)
    return _save(fig, out_dir / "forecast_backtest.png")


def to_markdown_table(df: pd.DataFrame) -> str:
    """Small Markdown table writer (avoids an extra dependency)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def generate_report(df: pd.DataFrame, month: str | None, out_dir: str | Path) -> dict[str, Path]:
    """Write charts, cleaned CSV and report.md. Returns the created files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    month = month or str(df["date"].max().to_period("M"))

    files = {
        "clean_csv": export_clean(df, out_dir / "clean_transactions.csv"),
        "monthly_chart": chart_monthly(df, out_dir),
        "category_chart": chart_categories(df, month, out_dir),
        "budget_chart": chart_budget(df, month, out_dir),
        "forecast_chart": chart_forecast(df, out_dir),
    }

    metrics = backtest_metrics(backtest(df))
    sections = [
        f"# Expense report, {month}\n",
        "## Monthly summary\n" + to_markdown_table(monthly_summary(df)),
        f"## Spending by category ({month})\n" + to_markdown_table(category_shares(df, month)),
        f"## Budget vs actual ({month})\n"
        + to_markdown_table(budget_vs_actual(df, load_budgets(), month)),
        "## Recurring expenses\n" + to_markdown_table(recurring_expenses(df)),
        f"## Forecast for {next_month_label(df)}\n"
        f"Predicted total: {forecast_total(df):,.0f}\n\n"
        + to_markdown_table(forecast_next_month(df)),
        "## Forecast accuracy (backtest)\n"
        + "\n".join(f"- {k}: {v}" for k, v in metrics.items()),
        "## Charts\n"
        + "\n".join(f"![{p.stem}]({p.name})" for k, p in files.items() if k.endswith("chart")),
    ]
    report = out_dir / "report.md"
    report.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    files["report"] = report
    logger.info("Report written to %s", report)
    return files
