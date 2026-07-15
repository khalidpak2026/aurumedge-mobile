@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title AurumEdge Mobile - iPhone Terminal

echo ============================================================
echo AurumEdge Mobile v5 - iPhone / mobile web terminal
echo No MetaTrader. No broker login. No order execution.
echo ============================================================

if not exist ".env" (
  echo Locating your existing API configuration...
  for %%F in (
    "%~dp0..\.env"
    "%~dp0..\gold_ai_web_terminal_pro_v4\.env"
    "%~dp0..\gold_ai_web_terminal_pro_v3\.env"
    "%~dp0..\gold_ai_web_terminal\.env"
    "%USERPROFILE%\Downloads\gold_ai_web_terminal_pro_v4\.env"
    "%USERPROFILE%\Downloads\gold_ai_web_terminal_pro_v3\.env"
    "%USERPROFILE%\Desktop\gold_ai_web_terminal_pro_v4\.env"
    "%USERPROFILE%\Desktop\gold_ai_web_terminal_pro_v3\.env"
  ) do (
    if not exist ".env" if exist "%%~F" copy /Y "%%~F" ".env" >nul
  )
)
if not exist ".env" copy ".env.example" ".env" >nul

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv 2>nul
  if errorlevel 1 py -m venv .venv
  if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
  set IP=%%A
  set IP=!IP: =!
  goto :gotip
)
:gotip
echo.
echo Laptop: http://localhost:8515
echo iPhone on same Wi-Fi: http://!IP!:8515
echo Keep this window and laptop running for local access.
echo.
start "" http://localhost:8515
python -m streamlit run mobile_app.py --server.port 8515 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true
exit /b 0

:error
echo Installation failed. Install Python 3.11 or newer and run again.
pause
exit /b 1
