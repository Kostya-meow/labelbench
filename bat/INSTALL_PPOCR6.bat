@echo off
setlocal EnableExtensions
set "UV_LINK_MODE=copy"
cd /d "%~dp0.."
set "PYTHON=%CD%\.venv-gpu\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Run INSTALL_GPU.bat first.
  pause
  exit /b 1
)
uv pip install --python "%PYTHON%" "transformers==5.17.0" "safetensors>=0.8" "opencv-python>=4.10" || goto :failed
"%PYTHON%" -c "from transformers import PPOCRV6MediumDetForObjectDetection; import torch; assert torch.cuda.is_available(); print('PP-OCRv6 runtime ready on CUDA')" || goto :failed
echo Run DOWNLOAD_PPOCR6.bat next.
pause
exit /b 0
:failed
echo PP-OCRv6 installation failed. See the error above.
pause
exit /b 1
