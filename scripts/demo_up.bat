@echo off
setlocal
cd /d "%~dp0\.."

if not defined PYTHON set PYTHON=python
if not defined APP_HOST set APP_HOST=127.0.0.1
if not defined APP_PORT set APP_PORT=5000
if not defined APP_DEBUG set APP_DEBUG=0

if not exist ".venv\Scripts\python.exe" (
  echo [demo] creating virtualenv at .venv
  %PYTHON% -m venv .venv
  if errorlevel 1 exit /b 1
)

echo [demo] installing requirements
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
if errorlevel 1 exit /b 1

echo [demo] dashboard: http://127.0.0.1:%APP_PORT%
echo [demo] health:    GET http://127.0.0.1:%APP_PORT%/health
echo [demo] CNN mode requires models\artifact_detector.pt (optional; heuristic mode is the default)
echo [demo] optional CNN path: python scripts\generate_demo_dataset.py ^&^& python training\train.py

if /I "%~1"=="--smoke" (
  echo [demo] Windows --smoke is not implemented; start the server and curl /health instead.
  exit /b 2
)

".venv\Scripts\python.exe" app\server.py
endlocal
