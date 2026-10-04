"""Load raw transaction CSVs and normalize them to the standard schema."""

import io
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "amount"]
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 accepts any byte, so it is the last resort
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


def read_csv_any_encoding(source) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file, whatever encoding Excel saved it in."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif hasattr(source, "getvalue"):  # Streamlit upload: safe to read again on every rerun
        data = source.getvalue()
    else:
        data = source.read()
    if isinstance(data, str):
        text = data
    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 (Excel "Unicode Text")
        text = data.decode("utf-16")
    else:
        for encoding in ENCODINGS:
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        logger.debug("Decoded CSV as %s", encoding)
    first_line = text.splitlines()[0] if text.strip() else ""
    separator = ";" if ";" in first_line and "," not in first_line else ","
    return pd.read_csv(io.StringIO(text), sep=separator)


def load_transactions(source) -> pd.DataFrame:
    """Read a CSV (path or file-like object) and return a cleaned DataFrame."""
    if isinstance(source, (str, Path)):
        logger.info("Loading transactions from %s", source)
    return clean_transactions(read_csv_any_encoding(source))
