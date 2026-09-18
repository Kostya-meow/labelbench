@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "UV_LINK_MODE=copy"
set "PYTHON=%CD%\.venv-rtmdet\Scripts\python.exe"

where uv >nul 2>nul || (
  echo ERROR: Install uv first: winget install Astral-sh.uv
  pause
  exit /b 1
)
if not exist "%PYTHON%" uv venv ".venv-rtmdet" --python 3.10 || goto :failed
uv pip install --python "%PYTHON%" "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || goto :failed
uv pip install --python "%PYTHON%" "mmcv-lite==2.0.1" "mmengine>=0.10,<0.11" "mmdet==3.1.0" "htrflow==0.2.6" || goto :failed
"%PYTHON%" -c "import torch; assert torch.cuda.is_available(); print('RTMDet GPU:', torch.cuda.get_device_name(0))" || goto :failed
echo READY. Run DOWNLOAD_RTMDET.bat, then START.bat.
pause
exit /b 0

:failed
echo FAILED. Copy the first error above.
pause
exit /b 1
