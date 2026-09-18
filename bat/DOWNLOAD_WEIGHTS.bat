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
set "PYTHON=%CD%\.venv-gpu\Scripts\python.exe"
set "LABELBENCH_OCR_PYTHON=%CD%\.venv-ocr\Scripts\python.exe"
set "LABELBENCH_SAM2_CHECKPOINT=%CD%\data\models\sam2.1_hiera_large.pt"
set "LABELBENCH_DLA_PYTHON=%CD%\.venv-dla\Scripts\python.exe"
set "LABELBENCH_RFDETR_CHECKPOINT=%CD%\data\models\rfdetr\rfdetr_text_seg_model_202510.pth.complete"
set "LABELBENCH_DOCUFCN_CHECKPOINT=%CD%\data\models\docufcn\generic_historical_line_model.pth"
set "LABELBENCH_EYNOLLAH_PYTHON=%CD%\.venv-eynollah\Scripts\python.exe"
set "LABELBENCH_EYNOLLAH_CHECKPOINT=%CD%\data\models\eynollah\eynollah_textline.onnx"
set "EYNOLLAH_KERAS=%CD%\data\models\eynollah\keras"

if not exist "%PYTHON%" (
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

"%PYTHON%" -c "import torch; assert torch.cuda.is_available()" || goto :failed
"%LABELBENCH_OCR_PYTHON%" -c "import torch, paddle; assert paddle.is_compiled_with_cuda()" || goto :failed
if not exist "%LABELBENCH_SAM2_CHECKPOINT%" (
  echo Downloading SAM 2.1 checkpoint with resume support...
  "%PYTHON%" scripts\download_file.py "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" "%LABELBENCH_SAM2_CHECKPOINT%" --size 898083611 --connections 8 || goto :failed
)
if not exist "%LABELBENCH_RFDETR_CHECKPOINT%" (
  echo Downloading RF-DETR historical textline checkpoint...
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/Kansallisarkisto/rfdetr_textline_textregion_detection_model/resolve/main/rfdetr_text_seg_model_202510.pth" "%LABELBENCH_RFDETR_CHECKPOINT%" --size 135572819 --connections 8 || goto :failed
)
if not exist "%LABELBENCH_DOCUFCN_CHECKPOINT%" (
  echo Downloading Doc-UFCN generic historical line checkpoint...
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/Teklia/doc-ufcn-generic-historical-line/resolve/main/model.pth" "%LABELBENCH_DOCUFCN_CHECKPOINT%" --size 49198561 --connections 8 || goto :failed
)
if not exist "%EYNOLLAH_KERAS%\keras_metadata.pb" (
  echo Downloading Eynollah Textline SavedModel...
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/SBB/eynollah-textline/resolve/main/keras_metadata.pb" "%EYNOLLAH_KERAS%\keras_metadata.pb" --size 433715 --connections 1 || goto :failed
)
if not exist "%EYNOLLAH_KERAS%\saved_model.pb" (
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/SBB/eynollah-textline/resolve/main/saved_model.pb" "%EYNOLLAH_KERAS%\saved_model.pb" --size 4421149 --connections 1 || goto :failed
)
if not exist "%EYNOLLAH_KERAS%\variables\variables.index" (
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/SBB/eynollah-textline/resolve/main/variables/variables.index" "%EYNOLLAH_KERAS%\variables\variables.index" --size 22657 --connections 1 || goto :failed
)
if not exist "%EYNOLLAH_KERAS%\variables\variables.data-00000-of-00001" (
  "%PYTHON%" scripts\download_file.py "https://huggingface.co/SBB/eynollah-textline/resolve/main/variables/variables.data-00000-of-00001" "%EYNOLLAH_KERAS%\variables\variables.data-00000-of-00001" --size 152969897 --connections 1 || goto :failed
)
if not exist "%LABELBENCH_EYNOLLAH_CHECKPOINT%" (
  echo Converting Eynollah Textline SavedModel to ONNX...
  "%LABELBENCH_EYNOLLAH_PYTHON%" -m tf2onnx.convert --saved-model "%EYNOLLAH_KERAS%" --output "%LABELBENCH_EYNOLLAH_CHECKPOINT%" --opset 13 || goto :failed
)
echo Downloading PP-OCRv5 Server, Mask2Former, SAM 2.1, YOLO26-seg, RF-DETR, Doc-UFCN and Eynollah weights...
"%PYTHON%" scripts\prefetch_models.py --providers ppocr mask2former sam2 yolo26 rfdetr_historical docufcn eynollah_textline || goto :failed
echo.
echo READY. All available weights are cached. Run START.bat.
pause
exit /b 0

:failed
echo.
echo FAILED. Copy the first error above and send it to me.
pause
exit /b 1
