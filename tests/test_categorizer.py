import tempfile
from pathlib import Path

from expense_tracker.categorizer import (
    UNCATEGORIZED, add_categories, add_correction, categorize_description, load_corrections,
)
from expense_tracker.loader import clean_transactions
from helpers import sample_df
import pandas as pd

RULES = {"Transport": ["uber", "ola"], "Food": ["swiggy", "pizza"]}


def test_keyword_match_and_unknown():
    assert categorize_description("Uber Trip", RULES) == "Transport"
    assert categorize_description("Unknown Shop", RULES) == UNCATEGORIZED


def test_keywords_match_whole_words_only():
    assert categorize_description("Chocolate Shop", RULES) == UNCATEGORIZED  # 'ola' inside a word
    assert categorize_description("Ola Ride", RULES) == "Transport"


def test_sample_has_no_uncategorized():
    assert (sample_df()["category"] != UNCATEGORIZED).all()


def test_corrections_override_rules():
    raw = pd.DataFrame({"date": ["2026-01-01"], "description": ["Uber Trip"], "amount": [-100]})
    df = add_categories(clean_transactions(raw), rules=RULES, corrections={"uber trip": "Shopping"})
    assert df["category"].iloc[0] == "Shopping"
    assert df["category_source"].iloc[0] == "correction"


def test_add_correction_saves_and_updates():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "corrections.csv"
        add_correction("Dominos Pizza", "Food", path)
        add_correction("dominos pizza", "Shopping", path)  # same merchant, replaced
        assert load_corrections(path) == {"dominos pizza": "Shopping"}
