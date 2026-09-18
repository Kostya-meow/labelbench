@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "UV_LINK_MODE=copy"
set "HF_HOME=%CD%\data\models\huggingface"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
set "TORCH_HOME=%CD%\data\models\torch"
set "PYTHON=%CD%\.venv-gpu\Scripts\python.exe"
set "OCR_PYTHON=%CD%\.venv-ocr\Scripts\python.exe"
set "DLA_PYTHON=%CD%\.venv-dla\Scripts\python.exe"
set "EYNOLLAH_PYTHON=%CD%\.venv-eynollah\Scripts\python.exe"
set "RTMDET_PYTHON=%CD%\.venv-rtmdet\Scripts\python.exe"

where uv >nul 2>nul
if errorlevel 1 (
  echo ERROR: Install uv first: winget install Astral-sh.uv
  pause
  exit /b 1
)

echo Creating isolated web and OCR environments...
if not exist "%PYTHON%" uv venv ".venv-gpu" --python 3.10 || goto :failed
if not exist "%OCR_PYTHON%" uv venv ".venv-ocr" --python 3.10 || goto :failed
if not exist "%DLA_PYTHON%" uv venv ".venv-dla" --python 3.10 || goto :failed
if not exist "%EYNOLLAH_PYTHON%" uv venv ".venv-eynollah" --python 3.10 || goto :failed
if not exist "%RTMDET_PYTHON%" uv venv ".venv-rtmdet" --python 3.10 || goto :failed
uv pip install --python "%PYTHON%" fastapi "uvicorn[standard]" python-multipart pillow numpy pydantic-settings pytest ruff || goto :failed

echo Installing Mask2Former, YOLO26 and official SAM 2 runtime...
uv pip install --python "%PYTHON%" "transformers==5.17.0" accelerate scipy "safetensors>=0.8" "opencv-python>=4.10" || goto :failed
uv pip install --python "%PYTHON%" "SAM-2 @ git+https://github.com/facebookresearch/sam2.git" || goto :failed

echo Installing PyTorch CUDA runtime for RTX 5060 Ti...
uv pip install --python "%PYTHON%" "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || goto :failed
uv pip install --python "%PYTHON%" ultralytics || goto :failed
uv pip install --python "%PYTHON%" rfdetr supervision "pyDeprecate>=0.9,<0.10" || goto :failed
uv pip install --python "%PYTHON%" -e . --no-deps || goto :failed

echo Installing isolated Doc-UFCN environment...
uv pip install --python "%DLA_PYTHON%" "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || goto :failed
uv pip install --python "%DLA_PYTHON%" --no-deps "doc-ufcn==0.1.9" "teklia-toolbox>=0.3" || goto :failed
uv pip install --python "%DLA_PYTHON%" "huggingface-hub>=0.19" "numpy<3" opencv-python-headless pyyaml requests tqdm || goto :failed

echo Installing isolated Eynollah ONNX conversion and GPU environment...
uv pip install --python "%EYNOLLAH_PYTHON%" "tensorflow==2.15.1" "tf2onnx>=1.16" "onnx==1.16.2" "onnxruntime-gpu>=1.18" pillow numpy opencv-python-headless || goto :failed

echo Installing isolated Riksarkivet RTMDet Lines environment...
uv pip install --python "%RTMDET_PYTHON%" "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || goto :failed
uv pip install --python "%RTMDET_PYTHON%" "mmcv-lite==2.0.1" "mmengine>=0.10,<0.11" "mmdet==3.1.0" "htrflow==0.2.6" || goto :failed

echo Installing isolated PP-OCRv5 and Paddle CUDA runtime...
uv pip install --python "%OCR_PYTHON%" fastapi "uvicorn[standard]" python-multipart pydantic-settings pillow numpy pydantic "torch==2.11.0" "paddleocr==3.4.1" "modelscope>=1.28,<1.30" || goto :failed
uv pip install --python "%OCR_PYTHON%" "paddlepaddle-gpu==3.2.0" --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ || goto :failed
uv pip install --python "%OCR_PYTHON%" -e . --no-deps || goto :failed

echo Verifying PyTorch and Paddle in isolated processes...
"%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; print('PyTorch GPU:', torch.cuda.get_device_name(0), torch.__version__)" || goto :failed
"%OCR_PYTHON%" -c "import torch, paddle; assert paddle.is_compiled_with_cuda(), 'Paddle CUDA is unavailable'; print('Paddle GPU:', paddle.device.get_device()); print('OCR helper Torch:', torch.__version__)" || goto :failed
"%RTMDET_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'RTMDet CUDA is unavailable'; print('RTMDet GPU:', torch.cuda.get_device_name(0), torch.__version__)" || goto :failed

echo.
echo READY. Run DOWNLOAD_WEIGHTS.bat, then START.bat.
pause
exit /b 0

:failed
echo.
echo FAILED. Copy the first error above and send it to me. Do not run DOWNLOAD_WEIGHTS.bat yet.
pause
exit /b 1
