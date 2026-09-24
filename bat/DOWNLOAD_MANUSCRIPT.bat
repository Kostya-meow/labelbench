@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "PYTHONIOENCODING=utf-8"
if not exist ".venv-manuscript\Scripts\python.exe" goto missing
".venv-manuscript\Scripts\python.exe" "scripts\manuscript_worker.py" --prefetch --device cuda
if errorlevel 1 goto fail
".venv-manuscript\Scripts\python.exe" "scripts\optimize_manuscript_onnx.py"
if errorlevel 1 goto fail
".venv-manuscript\Scripts\python.exe" "scripts\verify_manuscript_onnx.py"
if errorlevel 1 goto fail
echo All three release files verified; CUDA model loaded.
pause
exit /b 0
:missing
echo Run INSTALL_MANUSCRIPT.bat first.
:fail
echo FAILED. See the error above.
pause
exit /b 1
