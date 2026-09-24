@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title 2. CMA-ES optimization (budget from config.json)
echo ============================================================
echo   2. CMA-ES optimization (budget from config.json)
echo ============================================================
echo.
if exist "..\mjenv\Scripts\activate.bat" (
  call "..\mjenv\Scripts\activate.bat"
) else if exist "mjenv\Scripts\activate.bat" (
  call "mjenv\Scripts\activate.bat"
) else (
  echo [!] Virtual env not found at ..\mjenv  -- using system python
  echo     If it fails, run the "0_" setup .bat first.
  echo.
)
python -c "import mujoco" 2>nul
if errorlevel 1 (
  echo [X] mujoco not installed. Run the "0_" setup .bat first.
  echo.
  pause
  exit /b 1
)
python optimize.py config.json
echo.
echo ============================================================
echo   Finished. Press any key to close.
echo ============================================================
pause >nul
