@echo off
rem Tool-Edit Review Phim Studio - chay an, khong hien man hinh cmd (chi hien giao dien).
rem Can xem log / tu reload khi sua code: dung run_debug.bat
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Chua co moi truong .venv. Xem huong dan cai dat trong README.md
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0launcher.pyw"
exit
