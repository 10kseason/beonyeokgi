@echo off
setlocal

where python >NUL 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ was not found on PATH.
    echo Install Python and re-run this script.
    exit /b 1
)

if not exist .venv (
    echo [INFO] Creating virtual environment at .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo [INFO] Reusing existing virtual environment.
)

call .\.venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    exit /b 1
)

echo [INFO] Upgrading pip, setuptools, and wheel ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel.
    exit /b 1
)

echo [INFO] Installing project requirements (see requirements.txt) ...
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo.
echo [SUCCESS] Virtual environment is ready.
echo [SUCCESS] Activate later with: call .\.venv\Scripts\activate
echo [SUCCESS] Launch the app with: run.bat

deactivate
endlocal
