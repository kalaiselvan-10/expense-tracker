"""One place that wires the steps together (used by the CLI, dashboard and API)."""

import json
import logging
from pathlib import Path

import pandas as pd

from .categorizer import add_categories
from .config import load_config
from .loader import load_transactions
from .model import (
    evaluate_random_split,
    evaluate_unseen_merchants,
    fill_uncategorized,
    load_model,
    save_model,
    train_final,
)

logger = logging.getLogger(__name__)


def build_dataset(source=None, use_model: bool = True) -> pd.DataFrame:
    """Load -> clean -> corrections and rules -> model fills what is still unknown."""
    source = source or load_config()["paths"]["transactions"]
    df = add_categories(load_transactions(source))
    if use_model:
        df = fill_uncategorized(df, load_model())
    return df


def metrics_path() -> Path:
    return Path(load_config()["paths"]["output_dir"]) / "model_metrics.json"


def load_metrics() -> dict | None:
    path = metrics_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def train_and_evaluate(labeled_path: str | Path | None = None) -> dict:
    """Evaluate rules vs model, train on all labeled rows, save the model and metrics."""
    labeled_path = labeled_path or load_config()["paths"]["transactions"]
    labeled = pd.read_csv(labeled_path)
    if "true_category" not in labeled.columns:
        raise ValueError("Training needs a 'true_category' column (see scripts/generate_data.py)")

    easy = evaluate_random_split(labeled)
    hard = evaluate_unseen_merchants(labeled)
    save_model(train_final(labeled))

    pick = ("n_test", "rules_accuracy", "model_accuracy", "hybrid_accuracy", "unresolved_pct")
    metrics = {
        "labeled_rows": len(labeled),
        "note": "Evaluated on synthetic data; treat as a pipeline check, not real-world accuracy.",
        "known_merchants": {k: round(easy[k], 4) for k in pick},
        "unseen_merchants": {k: round(hard[k], 4) for k in pick},
    }
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", path)
    return metrics
