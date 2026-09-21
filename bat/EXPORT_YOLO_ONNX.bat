@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: EXPORT_YOLO_ONNX.bat "path-to-image.jpg"
  exit /b 1
)
uv pip install --python ".venv-gpu\Scripts\python.exe" "onnx>=1.17,<2" "onnxruntime-gpu>=1.21,<2"
if errorlevel 1 exit /b 1
set "LABELBENCH_DEVICE=cuda"
set "YOLO_AUTOINSTALL=False"
".venv-gpu\Scripts\python.exe" scripts\export_yolo_onnx.py --image "%~1"
if errorlevel 1 exit /b 1
echo Export and CUDA parity check completed.
