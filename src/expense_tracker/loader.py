"""Load raw transaction CSVs and normalize them to the standard schema."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "amount"]
NOISE_TOKENS = {"upi", "pos", "imps", "neft", "txn", "ref"}


def normalize_merchant(text) -> str:
    """'UPI/482913/swiggy ORDER #12' -> 'Swiggy Order'."""
    if pd.isna(text):
        return ""
    cleaned = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = [t for t in cleaned.split() if t not in NOISE_TOKENS]
    return " ".join(tokens).title()


def parse_amount(value) -> float:
    """Parse '1,299', '₹-450', 'Rs. 250' or '(450)' into a float (NaN if impossible)."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^0-9.\-]", "", text.replace("rs.", "").replace("inr", ""))
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return -abs(number) if negative else number


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates (YYYY-MM-DD) strictly, and everything else day-first (14/09/2026)."""
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    rest = parsed.isna()
    if rest.any():
        parsed[rest] = pd.to_datetime(
            text[rest], format="mixed", dayfirst=True, errors="coerce"
        )
    return parsed


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, amounts and merchant names; drop bad rows and duplicates."""
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "account" not in df.columns:
        df["account"] = "Unknown"

    df["date"] = parse_dates(df["date"])
    df["amount"] = df["amount"].map(parse_amount)
    df["description"] = df["description"].map(normalize_merchant)
    df["account"] = df["account"].fillna("Unknown").astype(str).str.strip()

    before = len(df)
    df = df.dropna(subset=["date", "amount"])
    df = df[df["description"] != ""]
    dropped_invalid = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    dropped_duplicates = before - len(df)

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    logger.info(
        "Cleaned %d rows -> %d (invalid dropped: %d, duplicates dropped: %d)",
        len(raw), len(df), dropped_invalid, dropped_duplicates,
    )
    return df


def load_transactions(source) -> pd.DataFrame:
    """Read a CSV (path or file-like object) and return a cleaned DataFrame."""
    if isinstance(source, (str, Path)):
        logger.info("Loading transactions from %s", source)
    return clean_transactions(pd.read_csv(source))
