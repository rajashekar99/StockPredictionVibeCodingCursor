@echo off
cd /d "%~dp0"

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing dependencies...
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo Starting Streamlit...
echo Open http://localhost:8501 in your browser
echo.
venv\Scripts\python.exe -m streamlit run app.py

pause
