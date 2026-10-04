"""Rule-based categorization with manual corrections."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)

UNCATEGORIZED = "Uncategorized"


def load_rules(path: str | Path | None = None) -> dict[str, list[str]]:
    """Keyword rules: {category: [keywords]}."""
    path = path or load_config()["paths"]["categories"]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_corrections(path: str | Path | None = None) -> dict[str, str]:
    """Manual corrections: {merchant (lowercase): category}. Missing file means none."""
    path = Path(path or load_config()["paths"]["corrections"])
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {
        str(row.description).strip().lower(): str(row.category).strip()
        for row in table.itertuples()
    }


def add_correction(description: str, category: str, path: str | Path | None = None) -> Path:
    """Save (or update) a manual correction so future runs use it."""
    path = Path(path or load_config()["paths"]["corrections"])
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["description", "category"])
    key = description.strip().lower()
    existing = existing[existing["description"].str.strip().str.lower() != key]
    new_row = pd.DataFrame([{"description": description.strip().title(), "category": category.strip()}])
    pd.concat([existing, new_row], ignore_index=True).to_csv(path, index=False)
    logger.info("Saved correction: %s -> %s", description, category)
    return path


def categorize_description(description: str, rules: dict[str, list[str]]) -> str:
    """First category (in file order) with a whole-word keyword match."""
    text = description.lower()
    words = set(re.findall(r"[a-z]+", text))
    for category, keywords in rules.items():
        for keyword in keywords:
            if (" " in keyword and keyword in text) or keyword in words:
                return category
    return UNCATEGORIZED


def add_categories(
    df: pd.DataFrame,
    rules: dict[str, list[str]] | None = None,
    corrections: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add `category` and `category_source` (correction / rule / none). Corrections win."""
    rules = load_rules() if rules is None else rules
    corrections = load_corrections() if corrections is None else corrections
    result = df.copy()

    categories, sources = [], []
    for description in result["description"]:
        key = description.strip().lower()
        if key in corrections:
            categories.append(corrections[key])
            sources.append("correction")
            continue
        category = categorize_description(description, rules)
        categories.append(category)
        sources.append("none" if category == UNCATEGORIZED else "rule")

    result["category"] = categories
    result["category_source"] = sources
    logger.info(
        "Categorized %d rows (%d uncategorized)",
        len(result), (result["category"] == UNCATEGORIZED).sum(),
    )
    return result
