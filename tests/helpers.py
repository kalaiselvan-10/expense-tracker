"""Shared helpers for tests."""

from pathlib import Path

import pandas as pd

from expense_tracker.categorizer import add_categories
from expense_tracker.loader import load_transactions

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_transactions.csv"
LABELED = ROOT / "data" / "labeled_transactions.csv"


def sample_df() -> pd.DataFrame:
    """Sample data with rules only (no corrections, no model) so results are stable."""
    return add_categories(load_transactions(SAMPLE), corrections={})


def labeled_df() -> pd.DataFrame:
    return add_categories(load_transactions(LABELED), corrections={})
