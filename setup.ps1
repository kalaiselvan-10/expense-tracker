# One-command setup for Windows PowerShell. Run from the project folder:  .\setup.ps1
$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[app,dev]"

Write-Host "`n== Running tests ==" -ForegroundColor Cyan
.\.venv\Scripts\python -m pytest -q

Write-Host "`n== Training the categorizer ==" -ForegroundColor Cyan
.\.venv\Scripts\python -m expense_tracker train

Write-Host "`n== Generating the report ==" -ForegroundColor Cyan
.\.venv\Scripts\python -m expense_tracker report

Write-Host "`nDone. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\activate"
Write-Host "  streamlit run app/streamlit_app.py"
Write-Host "  uvicorn expense_tracker.api:app --reload"
