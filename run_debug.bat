@echo off
title Tool-Edit Review Phim Studio (DEBUG - console + auto-reload)
cd /d "%~dp0"

echo [Tool-Edit DEBUG] Dang khoi dong backend tai http://127.0.0.1:8765 (co console + tu reload khi sua code)...
start "" /b "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload

timeout /t 2 /nobreak >nul

set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
set "CHROME=C:\Program Files\Google\Chrome\Application\chrome.exe"

if exist "%EDGE%" (
    start "" "%EDGE%" --app=http://127.0.0.1:8765 --window-size=1440,900
) else if exist "%CHROME%" (
    start "" "%CHROME%" --app=http://127.0.0.1:8765 --window-size=1440,900
) else (
    start "" http://127.0.0.1:8765
)

echo [Tool-Edit DEBUG] Cua so ung dung da mo. Nhan Ctrl+C hoac dong cua so nay de tat server.
pause >nul
