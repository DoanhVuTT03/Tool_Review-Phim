@echo off
rem Tool-Edit Review Phim Studio - cai dat / sua moi truong (chay lai bao nhieu lan cung duoc).
rem  1. Tao .venv (Python 3.11) va cai thu vien
rem  2. Sua xung dot onnxruntime: rapidocr-onnxruntime keo theo goi "onnxruntime" thuong, goi nay DE LEN
rem     onnxruntime-directml -> AI OCR xoa sub mat GPU, chi chay CPU. Go goi thuong, cai lai ban DirectML.
rem  3. Tai ffmpeg/ffprobe (gyan.dev essentials: co NVENC/NVDEC/QSV) va aria2c vao bin\ neu chua co
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Tao moi truong .venv bang Python 3.11...
    py -3.11 -m venv .venv || (echo Khong tim thay Python 3.11. Cai Python 3.11 roi chay lai. & pause & exit /b 1)
)
set "PY=%~dp0.venv\Scripts\python.exe"

echo [2/4] Cai thu vien...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt || (echo Cai thu vien loi. & pause & exit /b 1)

echo [3/4] Sua xung dot onnxruntime / opencv (giu ban DirectML cho GPU)...
"%PY%" -m pip uninstall -y onnxruntime onnxruntime-directml opencv-python >nul 2>&1
"%PY%" -m pip install --force-reinstall --no-deps onnxruntime-directml opencv-python-headless || (echo Loi cai onnxruntime-directml. & pause & exit /b 1)
"%PY%" -c "import onnxruntime as o; p=o.get_available_providers(); print('   ONNX providers:', p); raise SystemExit(0 if 'DmlExecutionProvider' in p else 1)" || echo    CANH BAO: DirectML chua hoat dong, AI OCR se chay CPU.

rem May co card NVIDIA: PyTorch CUDA de xoa sub AI chay tren GPU (OCR + tai tao nen), nhanh ~2.3x.
rem Thu tu quan trong: cai torch ban CUDA truoc, onnx2torch --no-deps sau (neu khong pip se keo torch ban CPU de len).
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [GPU] Phat hien card NVIDIA: cai PyTorch CUDA ^(~2.5 GB, lan dau hoi lau^)...
    "%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        "%PY%" -m pip uninstall -y torch torchvision >nul 2>&1
        "%PY%" -m pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    )
    "%PY%" -m pip install onnx
    "%PY%" -m pip install --no-deps onnx2torch
    "%PY%" -c "import torch; print('   PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
)

echo [4/4] Kiem tra ffmpeg / aria2c trong bin\ ...
if not exist "bin\ffmpeg.exe" (
    echo    Dang tai ffmpeg essentials ^(~120 MB^)...
    curl -L -o "%TEMP%\ff_essentials.zip" https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip && (
        mkdir "%TEMP%\ff_essentials" 2>nul
        tar -xf "%TEMP%\ff_essentials.zip" -C "%TEMP%\ff_essentials"
        for /d %%D in ("%TEMP%\ff_essentials\ffmpeg-*") do (copy /y "%%D\bin\ffmpeg.exe" bin\ >nul & copy /y "%%D\bin\ffprobe.exe" bin\ >nul)
        rmdir /s /q "%TEMP%\ff_essentials"
        del "%TEMP%\ff_essentials.zip"
    )
)
if not exist "bin\aria2c.exe" (
    echo    Dang tai aria2c ^(tai video da luong^)...
    curl -L -o "%TEMP%\aria2.zip" https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip && (
        mkdir "%TEMP%\aria2x" 2>nul
        tar -xf "%TEMP%\aria2.zip" -C "%TEMP%\aria2x"
        for /d %%D in ("%TEMP%\aria2x\aria2-*") do (copy /y "%%D\aria2c.exe" bin\ >nul)
        rmdir /s /q "%TEMP%\aria2x"
        del "%TEMP%\aria2.zip"
    )
)
if exist "bin\ffmpeg.exe" (echo    ffmpeg OK) else (echo    CANH BAO: chua co bin\ffmpeg.exe - xem README.md)

echo.
echo Xong! Nhap dup run.bat de mo tool.
pause
