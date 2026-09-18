@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "LABELBENCH_DEVICE=cuda"
set "MAIN_PYTHON=%CD%\.venv-gpu\Scripts\python.exe"
set "LABELBENCH_RTMDET_PYTHON=%CD%\.venv-rtmdet\Scripts\python.exe"
set "LABELBENCH_RTMDET_CHECKPOINT=%CD%\data\models\rtmdet_lines\model.pth"
set "LABELBENCH_RTMDET_CONFIG=%CD%\data\models\rtmdet_lines\config.py"

if not exist "%MAIN_PYTHON%" goto :missing
if not exist "%LABELBENCH_RTMDET_PYTHON%" goto :missing
if not exist "%LABELBENCH_RTMDET_CONFIG%" "%MAIN_PYTHON%" scripts\download_file.py "https://huggingface.co/Riksarkivet/rtmdet_lines/resolve/main/config.py" "%LABELBENCH_RTMDET_CONFIG%" --size 20524 --connections 1 || goto :failed
if not exist "%LABELBENCH_RTMDET_CHECKPOINT%" "%MAIN_PYTHON%" scripts\download_file.py "https://huggingface.co/Riksarkivet/rtmdet_lines/resolve/main/model.pth" "%LABELBENCH_RTMDET_CHECKPOINT%" --size 474957088 --connections 8 || goto :failed
"%MAIN_PYTHON%" scripts\prefetch_models.py --providers rtmdet_lines || goto :failed
echo READY. RTMDet Lines is cached and verified on GPU.
pause
exit /b 0

:missing
echo ERROR: Run INSTALL_GPU.bat or INSTALL_RTMDET.bat first.
pause
exit /b 1

:failed
echo FAILED. Copy the first error above.
pause
exit /b 1
