import tempfile
from pathlib import Path

import pandas as pd

from expense_tracker.categorizer import UNCATEGORIZED
from expense_tracker.model import (
    evaluate_random_split, evaluate_unseen_merchants, fill_uncategorized, load_model,
    save_model, train_final,
)
from helpers import LABELED

LABELED_DF = pd.read_csv(LABELED)


def test_model_beats_chance_on_known_merchants():
    result = evaluate_random_split(LABELED_DF)
    assert result["model_accuracy"] > 0.9
    assert result["hybrid_accuracy"] >= result["rules_accuracy"]


def test_unseen_merchant_test_holds_out_merchants():
    result = evaluate_unseen_merchants(LABELED_DF)
    assert len(result["held_out_merchants"]) >= 3
    assert result["n_test"] > 0 and "hybrid_accuracy" in result


def test_model_fills_only_uncategorized_rows():
    model = train_final(LABELED_DF)
    df = pd.DataFrame({
        "description": ["Dominos Pizza", "Uber Trip"],
        "category": [UNCATEGORIZED, "Transport"],
        "category_source": ["none", "rule"],
    })
    out = fill_uncategorized(df, model, min_confidence=0.2)
    assert out.loc[0, "category"] == "Food" and out.loc[0, "category_source"] == "model"
    assert out.loc[1, "category_source"] == "rule"  # rule result left untouched


def test_model_can_be_saved_and_loaded():
    model = train_final(LABELED_DF)
    with tempfile.TemporaryDirectory() as folder:
        path = save_model(model, Path(folder) / "m.joblib")
        assert list(load_model(path).predict(["Swiggy Order"])) == ["Food"]
        assert load_model(Path(folder) / "missing.joblib") is None
