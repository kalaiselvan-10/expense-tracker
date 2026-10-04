import contextlib
import io
import tempfile
from pathlib import Path

import pandas as pd

from expense_tracker.cli import main
from expense_tracker.reports import SCHEMA_COLUMNS, export_clean, generate_report, to_markdown_table
from helpers import LABELED, SAMPLE, labeled_df, sample_df


def test_export_uses_standard_schema():
    with tempfile.TemporaryDirectory() as folder:
        path = export_clean(sample_df(), Path(folder) / "clean.csv")
        out = pd.read_csv(path)
        assert list(out.columns) == SCHEMA_COLUMNS
        assert len(out) == 15 and out["date"].iloc[0] == "2026-08-01"


def test_generate_report_creates_all_files():
    with tempfile.TemporaryDirectory() as folder:
        files = generate_report(labeled_df(), "2026-09", folder)
        for path in files.values():
            assert Path(path).exists() and Path(path).stat().st_size > 0
        assert "Budget vs actual" in files["report"].read_text(encoding="utf-8")


def test_markdown_table():
    table = to_markdown_table(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert table.splitlines()[0] == "| a | b |" and table.splitlines()[2] == "| 1 | x |"


def test_cli_forecast_runs():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(["forecast", "--input", str(LABELED)])
    assert "Forecast for 2026-10" in buffer.getvalue()
