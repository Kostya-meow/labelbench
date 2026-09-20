@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "LABELBENCH_DEVICE=cuda"
set "HF_HOME=%CD%\data\models\huggingface"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
"%CD%\.venv-gpu\Scripts\python.exe" scripts\prefetch_models.py --providers ppocr6 ppocr6_small ppocr6_tiny
if errorlevel 1 (
  echo Download failed. Run INSTALL_PPOCR6.bat and retry.
  pause
  exit /b 1
)
echo PP-OCRv6 weights ready. Run START.bat.
pause
exit /b 0
