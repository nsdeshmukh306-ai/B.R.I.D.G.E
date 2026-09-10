@echo off
setlocal
set ROOT=%~dp0
call "%ROOT%.venv\Scripts\activate.bat"
echo Reinstalling dependencies after a code update...
pip install -e "%ROOT%[dev]" || (echo Install failed & pause & exit /b 1)
pip uninstall -y opencv-python opencv-python-headless >nul 2>&1
pip install opencv-contrib-python
cd /d "%ROOT%"
if exist profiles\default.json del profiles\default.json
echo.
bridge --headless-selftest
echo.
echo Update finished.
pause
