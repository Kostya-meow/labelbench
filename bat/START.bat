@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "LABELBENCH_DEVICE=cuda"
set "HF_HOME=%CD%\data\models\huggingface"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
set "TORCH_HOME=%CD%\data\models\torch"
set "YOLO_CONFIG_DIR=%CD%\data\models\ultralytics"
set "RF_HOME=%CD%\data\models\rfdetr"
set "UVICORN=%CD%\.venv-gpu\Scripts\uvicorn.exe"
set "LABELBENCH_OCR_PYTHON=%CD%\.venv-ocr\Scripts\python.exe"
set "LABELBENCH_SAM2_CHECKPOINT=%CD%\data\models\sam2.1_hiera_large.pt"
set "LABELBENCH_DLA_PYTHON=%CD%\.venv-dla\Scripts\python.exe"
set "LABELBENCH_RFDETR_CHECKPOINT=%CD%\data\models\rfdetr\rfdetr_text_seg_model_202510.pth.complete"
set "LABELBENCH_DOCUFCN_CHECKPOINT=%CD%\data\models\docufcn\generic_historical_line_model.pth"
set "LABELBENCH_EYNOLLAH_PYTHON=%CD%\.venv-eynollah\Scripts\python.exe"
set "LABELBENCH_EYNOLLAH_CHECKPOINT=%CD%\data\models\eynollah\eynollah_textline.onnx"

if not exist "%UVICORN%" (
  echo ERROR: Run INSTALL_GPU.bat first.
  pause
  exit /b 1
)
if not exist "%LABELBENCH_OCR_PYTHON%" (
  echo ERROR: Run INSTALL_GPU.bat first.
  pause
  exit /b 1
)
if not exist "%LABELBENCH_DLA_PYTHON%" (
  echo ERROR: Run INSTALL_GPU.bat first.
  pause
  exit /b 1
)
if not exist "%LABELBENCH_EYNOLLAH_PYTHON%" (
  echo ERROR: Run INSTALL_GPU.bat first.
  pause
  exit /b 1
)

echo Open http://127.0.0.1:8000
echo Put images into data\images. Press Ctrl+C here to stop the server.
"%UVICORN%" labelbench.api:app --host 127.0.0.1 --port 8000
pause
