@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
set "TORCH_HOME=%~dp0.cache\torch"
if not exist ".venv\Scripts\python.exe" (
    echo Missing integrated_analysis\.venv. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -u integrated_app.py %*
set "integrated_exit_code=%ERRORLEVEL%"
if not "%integrated_exit_code%"=="0" pause
exit /b %integrated_exit_code%
