@echo off
REM ============================================================
REM setup.bat — Windows 11 Setup Script
REM Run this ONCE before running main.py
REM Double-click or run from terminal: .\setup.bat
REM ============================================================

echo.
echo ============================================================
echo   Customer Churn Prediction Model — Setup
echo   Windows 11 / Python 3.10
echo ============================================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 from python.org
    pause
    exit /b 1
)

echo.
echo [STEP 1] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo [STEP 2] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [STEP 3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [STEP 4] Installing all dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To run the project:
echo   1. Activate venv:     venv\Scripts\activate
echo   2. Run pipeline:      python main.py
echo   3. Run dashboard:     python src\dashboard.py
echo   4. Open notebook:     jupyter notebook notebooks\
echo.
pause
