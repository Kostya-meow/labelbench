@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "PYTHONIOENCODING=utf-8"
".venv-gpu\Scripts\python.exe" "scripts\setup_manuscript.py"
if errorlevel 1 goto fail
echo Manuscript source environment and ONNX GPU model are ready.
pause
exit /b 0
:fail
echo FAILED. See the error above.
pause
exit /b 1
