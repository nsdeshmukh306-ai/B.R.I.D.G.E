@echo off
setlocal
REM BRIDGE first-time setup. Repo and venv both live in this folder.
set ROOT=%~dp0
set VENV=%ROOT%.venv

if not exist "%ROOT%pyproject.toml" (
  echo Could not find pyproject.toml next to this script - is setup.bat in the project folder?
  pause & exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv "%VENV%" || (echo Python 3.11+ not found on PATH & pause & exit /b 1)
)

call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip
echo Installing BRIDGE and dependencies (a few minutes the first time)...
pip install -e "%ROOT%[dev]" || (echo Install failed & pause & exit /b 1)
REM keep only the contrib build of OpenCV so the CSRT/KCF trackers are available
pip uninstall -y opencv-python opencv-python-headless >nul 2>&1
pip install opencv-contrib-python

if not exist "%ROOT%.env" (
  copy "%ROOT%.env.example" "%ROOT%.env" >nul
  echo.
  echo Created .env - open it and put your key after GEMINI_API_KEY=
)

echo.
echo Running self-test (clinic simulation: calibrate, locate syringe, start a procedure)...
cd /d "%ROOT%"
bridge --headless-selftest
echo.
echo Setup finished. Use run-bridge.bat (hardware) or run-simulation.bat (no hardware).
pause
