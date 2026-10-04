from expense_tracker.forecast import (
    backtest, backtest_metrics, forecast_next_month, forecast_total, monthly_expenses, next_month_label,
)
from helpers import labeled_df, sample_df


def test_forecast_is_average_of_available_months():
    assert abs(forecast_total(sample_df()) - 6022.5) < 0.01
    assert next_month_label(sample_df()) == "2026-10"


def test_category_forecast_uses_window():
    result = forecast_next_month(labeled_df(), window=3)
    assert (result["based_on_months"] == 3).all()
    assert result.set_index("category").loc["Rent", "forecast"] == 12000


def test_backtest_uses_only_earlier_months():
    df = labeled_df()
    result = backtest(df, window=3, min_train=3)
    totals = monthly_expenses(df)
    assert len(result) == len(totals) - 3
    first = result.iloc[0]
    assert first["month"] == str(totals.index[3])
    assert abs(first["moving_avg"] - totals.iloc[:3].mean()) < 0.01  # no peeking at the actual month


def test_backtest_metrics_are_sensible():
    metrics = backtest_metrics(backtest(labeled_df()))
    assert metrics["months_tested"] == 9
    assert 0 < metrics["moving_avg_mape_pct"] < 20


def test_backtest_with_too_little_data_is_empty():
    assert backtest(sample_df()).empty
