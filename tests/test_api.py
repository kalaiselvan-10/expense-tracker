import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from expense_tracker.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/").json()["status"] == "ok"


def test_summary_and_categories():
    summary = client.get("/summary").json()
    assert len(summary) == 12 and "expenses" in summary[0]
    assert client.get("/categories", params={"month": "2026-09"}).status_code == 200


def test_budget_and_forecast():
    assert client.get("/budget/2026-09").status_code == 200
    forecast = client.get("/forecast").json()
    assert forecast["month"] == "2026-10" and forecast["predicted_total"] > 0
    assert "metrics" in client.get("/backtest").json()


def test_bad_months_are_rejected():
    assert client.get("/budget/not-a-month").status_code == 400
    assert client.get("/budget/1999-01").status_code == 404
