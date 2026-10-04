import io
import tempfile
from pathlib import Path

import pandas as pd

from expense_tracker.loader import (
    clean_transactions, load_transactions, normalize_merchant, parse_amount, parse_dates,
)
from helpers import sample_df


def test_normalize_merchant_removes_bank_noise():
    assert normalize_merchant("UPI/482913/swiggy ORDER #12") == "Swiggy Order"
    assert normalize_merchant("  uber   trip ") == "Uber Trip"
    assert normalize_merchant(None) == ""


def test_parse_amount_handles_symbols_and_brackets():
    assert parse_amount("₹-450") == -450
    assert parse_amount("1,299") == 1299
    assert parse_amount("(450)") == -450
    assert parse_amount(250) == 250
    assert pd.isna(parse_amount("abc"))


def test_parse_dates_never_swaps_iso_day_and_month():
    parsed = parse_dates(pd.Series(["2026-08-01", "14/09/2026"]))
    assert parsed.iloc[0] == pd.Timestamp("2026-08-01")
    assert parsed.iloc[1] == pd.Timestamp("2026-09-14")


def test_sample_cleaning_removes_duplicate_and_fixes_date():
    df = sample_df()
    assert len(df) == 15  # 16 rows minus 1 exact duplicate
    assert df["date"].notna().all()
    assert pd.Timestamp("2026-09-14") in set(df["date"])
    assert df["date"].is_monotonic_increasing


def test_missing_required_column_raises():
    try:
        clean_transactions(pd.DataFrame({"date": ["2026-01-01"], "amount": [1]}))
    except ValueError as error:
        assert "description" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_account_is_optional():
    raw = pd.DataFrame({"date": ["2026-01-01"], "description": ["Uber Trip"], "amount": [-100]})
    assert clean_transactions(raw)["account"].iloc[0] == "Unknown"


CSV_TEXT = "date,description,amount,account\n2026-09-02,Caf\u00e9 Coffee \u00f0,-120,HDFC\n14/09/2026,Uber Trip,-250,HDFC\n"


def test_reads_non_utf8_files_saved_by_excel():
    for encoding in ("cp1252", "latin-1", "utf-8-sig", "utf-16"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.csv"
            path.write_bytes(CSV_TEXT.encode(encoding))
            df = load_transactions(path)
            assert len(df) == 2, encoding
            assert pd.Timestamp("2026-09-14") in set(df["date"]), encoding


def test_reads_uploaded_file_object_more_than_once():
    upload = io.BytesIO(CSV_TEXT.encode("cp1252"))
    assert len(load_transactions(upload)) == 2
    assert len(load_transactions(upload)) == 2  # Streamlit reruns reuse the same upload


def test_reads_semicolon_separated_files():
    text = "date;description;amount\n2026-09-02;Swiggy Order;-450\n"
    assert len(load_transactions(io.BytesIO(text.encode("utf-8")))) == 1
