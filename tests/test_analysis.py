from expense_tracker.analysis import (
    budget_vs_actual, category_shares, monthly_summary, recurring_expenses, spending_by_category,
)
from helpers import sample_df


def test_monthly_summary_september():
    row = monthly_summary(sample_df()).query("month == '2026-09'").iloc[0]
    assert row["income"] == 50000
    assert row["expenses"] == 9347
    assert row["net"] == 40653


def test_category_totals_and_shares():
    spent = spending_by_category(sample_df()).set_index("category")["spent"]
    assert spent["Bills"] == 4148 and spent["Groceries"] == 2300
    shares = category_shares(sample_df())
    assert abs(shares["share_pct"].sum() - 100) < 0.5
    assert shares.iloc[0]["category"] == "Bills"


def test_recurring_expenses_found():
    names = set(recurring_expenses(sample_df())["description"])
    assert names == {"Electricity Bill", "Mobile Recharge", "Netflix Subscription"}


def test_budget_variance_flags_overspend():
    budgets = {"Food": 1500, "Subscriptions": 600, "Shopping": 1000, "Transport": 2000}
    result = budget_vs_actual(sample_df(), budgets, "2026-09").set_index("category")
    assert result.loc["Shopping", "status"] == "Over"
    assert result.loc["Shopping", "variance"] == -299
    assert result.loc["Subscriptions", "variance"] == -49
    assert result.loc["Food", "status"] == "OK"


def test_spending_without_budget_is_flagged():
    result = budget_vs_actual(sample_df(), {"Food": 1500}, "2026-09").set_index("category")
    assert result.loc["Groceries", "status"] == "No budget"
