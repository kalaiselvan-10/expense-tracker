"""Train and evaluate a text classifier (TF-IDF + logistic regression) for categories."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .categorizer import UNCATEGORIZED, categorize_description, load_rules
from .config import load_config
from .loader import normalize_merchant

logger = logging.getLogger(__name__)


def build_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        LogisticRegression(max_iter=2000),
    )


def prepare(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Normalized merchant text and true labels from a labeled dataset."""
    return df["description"].map(normalize_merchant), df["true_category"]


def _hybrid_predict(model, rules, X, min_confidence: float) -> np.ndarray:
    """Rules first; the model fills only rows the rules left uncategorized (if confident)."""
    rule_pred = X.map(lambda d: categorize_description(d, rules)).to_numpy()
    probabilities = model.predict_proba(X)
    model_pred = model.classes_[probabilities.argmax(axis=1)]
    fill = (rule_pred == UNCATEGORIZED) & (probabilities.max(axis=1) >= min_confidence)
    return np.where(fill, model_pred, rule_pred)


def _score(model, rules, X_test, y_test) -> dict:
    min_confidence = load_config()["model"]["min_confidence"]
    rule_pred = X_test.map(lambda d: categorize_description(d, rules))
    model_pred = model.predict(X_test)
    hybrid_pred = _hybrid_predict(model, rules, X_test, min_confidence)
    return {
        "n_test": len(X_test),
        "rules_accuracy": accuracy_score(y_test, rule_pred),
        "model_accuracy": accuracy_score(y_test, model_pred),
        "hybrid_accuracy": accuracy_score(y_test, hybrid_pred),
        "unresolved_pct": float((hybrid_pred == UNCATEGORIZED).mean() * 100),
        "report": classification_report(y_test, model_pred, zero_division=0),
    }


def evaluate_random_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> dict:
    """Held-out rows, but merchants may also appear in training (the easy test)."""
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = build_model().fit(X_train, y_train)
    return {"model": model, "n_train": len(X_train),
            **_score(model, load_rules(), X_test, y_test)}


def evaluate_unseen_merchants(df: pd.DataFrame, seed: int = 42) -> dict:
    """Hold out whole merchants (the hard, honest test).

    From every category with 3+ distinct merchants, about a quarter of the
    merchants are removed from training and used only for testing.
    """
    X, y = prepare(df)
    rng = np.random.default_rng(seed)
    held_out = set()
    for category, merchants in X.groupby(y).unique().items():
        if len(merchants) >= 3:
            count = max(1, round(len(merchants) * 0.25))
            held_out.update(rng.choice(merchants, size=count, replace=False))
    is_test = X.isin(held_out)
    model = build_model().fit(X[~is_test], y[~is_test])
    return {"n_train": int((~is_test).sum()), "held_out_merchants": sorted(held_out),
            **_score(model, load_rules(), X[is_test], y[is_test])}


def train_final(df: pd.DataFrame):
    """Fit on all labeled rows for real use."""
    X, y = prepare(df)
    return build_model().fit(X, y)


def save_model(model, path: str | Path | None = None) -> Path:
    path = Path(path or load_config()["paths"]["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: str | Path | None = None):
    """Return the saved model, or None if it has not been trained yet."""
    path = Path(path or load_config()["paths"]["model"])
    return joblib.load(path) if path.exists() else None


def fill_uncategorized(df: pd.DataFrame, model, min_confidence: float | None = None) -> pd.DataFrame:
    """Use the model only for rows the rules and corrections left uncategorized."""
    if min_confidence is None:
        min_confidence = load_config()["model"]["min_confidence"]
    result = df.copy()
    mask = result["category"] == UNCATEGORIZED
    if model is None or not mask.any():
        return result
    probabilities = model.predict_proba(result.loc[mask, "description"])
    best = probabilities.argmax(axis=1)
    confident = probabilities.max(axis=1) >= min_confidence
    predicted = model.classes_[best]
    index = result.index[mask]
    result.loc[index[confident], "category"] = predicted[confident]
    result.loc[index[confident], "category_source"] = "model"
    logger.info("Model filled %d of %d uncategorized rows", int(confident.sum()), int(mask.sum()))
    return result
