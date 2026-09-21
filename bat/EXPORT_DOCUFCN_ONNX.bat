@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: EXPORT_DOCUFCN_ONNX.bat "path-to-image.jpg"
  exit /b 1
)
uv pip install --python ".venv-dla\Scripts\python.exe" "onnx>=1.17,<2" "onnxruntime-gpu>=1.21,<2"
if errorlevel 1 exit /b 1
".venv-gpu\Scripts\python.exe" scripts\export_docufcn_onnx.py --image "%~1"
exit /b %errorlevel%
