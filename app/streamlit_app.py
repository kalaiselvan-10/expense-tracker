"""Streamlit dashboard.  Run:  streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without pip install

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from expense_tracker.analysis import (  # noqa: E402
    budget_vs_actual, category_shares, load_budgets, monthly_summary, recurring_expenses,
)
from expense_tracker.categorizer import UNCATEGORIZED, add_correction, load_corrections, load_rules  # noqa: E402
from expense_tracker.config import load_config, setup_logging  # noqa: E402
from expense_tracker.forecast import (  # noqa: E402
    backtest, backtest_metrics, forecast_next_month, forecast_total, next_month_label,
)
from expense_tracker.pipeline import build_dataset, load_metrics, train_and_evaluate  # noqa: E402
from expense_tracker.reports import SCHEMA_COLUMNS  # noqa: E402

setup_logging()
st.set_page_config(page_title="Expense Tracker", layout="wide")
st.title("Expense Tracker & Budget Analyzer")

# ---- Sidebar: data source and month ----
uploaded = st.sidebar.file_uploader("Upload a transactions CSV", type="csv")
st.sidebar.caption("Needs columns: date, description, amount (account optional). Expenses negative.")
source = uploaded if uploaded is not None else load_config()["paths"]["transactions"]

try:
    df = build_dataset(source)
except ValueError as error:
    st.error(f"Could not read the file: {error}")
    st.stop()

months = sorted(df["date"].dt.to_period("M").astype(str).unique())
month = st.sidebar.selectbox("Month", months, index=len(months) - 1)

tabs = st.tabs(["Overview", "Budget", "Recurring", "Forecast", "Corrections", "Model", "Data"])

# ---- Overview ----
with tabs[0]:
    summary = monthly_summary(df)
    row = summary[summary["month"] == month].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"₹{row['income']:,.0f}")
    c2.metric("Expenses", f"₹{row['expenses']:,.0f}")
    c3.metric("Net savings", f"₹{row['net']:,.0f}")
    c4.metric("Savings rate", f"{row['savings_rate_pct']:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Income vs expenses by month")
        st.bar_chart(summary.set_index("month")[["income", "expenses"]])
    with right:
        st.subheader(f"Spending by category ({month})")
        shares = category_shares(df, month)
        st.bar_chart(shares.set_index("category")["spent"])
    st.dataframe(shares, hide_index=True, width="stretch")

# ---- Budget ----
with tabs[1]:
    st.subheader(f"Budget vs actual ({month})")
    budget = budget_vs_actual(df, load_budgets(), month)
    over = budget[budget["status"] == "Over"]
    if len(over):
        st.warning("Over budget: " + ", ".join(over["category"]))
    else:
        st.success("Within budget in every category.")
    st.bar_chart(budget.set_index("category")[["budget", "spent"]])
    st.dataframe(budget, hide_index=True, width="stretch")

# ---- Recurring ----
with tabs[2]:
    st.subheader("Recurring expenses")
    st.dataframe(recurring_expenses(df), hide_index=True, width="stretch")

# ---- Forecast ----
with tabs[3]:
    st.subheader(f"Forecast for {next_month_label(df)}")
    st.metric("Predicted total expenses", f"₹{forecast_total(df):,.0f}")
    st.dataframe(forecast_next_month(df), hide_index=True, width="stretch")
    result = backtest(df)
    st.subheader("How accurate is it? (backtest)")
    if result.empty:
        st.info("Need at least 4 months of data to backtest.")
    else:
        st.caption("Each month is predicted using only earlier months, then compared with what happened.")
        st.line_chart(result.set_index("month")[["actual", "moving_avg", "naive"]])
        st.json(backtest_metrics(result))

# ---- Corrections ----
with tabs[4]:
    st.subheader("Fix a category")
    st.caption("Corrections are saved to corrections.csv and always override rules and the model.")
    unknown = sorted(df.loc[df["category"] == UNCATEGORIZED, "description"].unique())
    options = sorted(set(load_rules()) | set(load_budgets()))
    if unknown:
        merchant = st.selectbox("Uncategorized merchant", unknown)
    else:
        st.success("Nothing is uncategorized. You can still correct any merchant:")
        merchant = st.selectbox("Merchant", sorted(df["description"].unique()))
    category = st.selectbox("Category", options)
    if st.button("Save correction"):
        add_correction(merchant, category)
        st.success(f"Saved: {merchant} -> {category}")
        st.rerun()
    saved = load_corrections()
    if saved:
        st.write("Saved corrections")
        st.dataframe(pd.DataFrame(saved.items(), columns=["merchant", "category"]),
                     hide_index=True, width="stretch")

# ---- Model ----
with tabs[5]:
    st.subheader("Categorizer: rules vs trained model")
    st.caption("Trained on data/labeled_transactions.csv (synthetic). The model only fills rows "
               "the rules could not label, and only when confident.")
    if st.button("Train / retrain model"):
        with st.spinner("Training..."):
            train_and_evaluate()
        st.rerun()
    metrics = load_metrics()
    if metrics:
        table = pd.DataFrame(
            {"Known merchants": metrics["known_merchants"], "Unseen merchants": metrics["unseen_merchants"]}
        ).T.rename(columns={
            "n_test": "test rows", "rules_accuracy": "rules", "model_accuracy": "model alone",
            "hybrid_accuracy": "rules + model", "unresolved_pct": "left unresolved %",
        })
        st.dataframe(table, width="stretch")
        st.caption(metrics["note"])
    else:
        st.info("No model yet. Click the button above.")

# ---- Data ----
with tabs[6]:
    st.subheader("Cleaned transactions")
    st.dataframe(df, hide_index=True, width="stretch")
    export = df[SCHEMA_COLUMNS].copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    st.download_button("Download cleaned CSV", export.to_csv(index=False),
                       file_name="clean_transactions.csv", mime="text/csv")
