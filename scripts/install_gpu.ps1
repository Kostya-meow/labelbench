[CmdletBinding()]
param(
    [ValidateSet('cu126', 'cu128')]
    [string]$TorchCuda = 'cu128',
    [ValidateSet('cu118', 'cu126', 'cu129')]
    [string]$PaddleCuda = 'cu126'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Create the environment first: .\scripts\bootstrap.ps1'
}

& uv pip install --python $python --upgrade --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$TorchCuda"
& uv pip install --python $python --upgrade "paddlepaddle-gpu==3.2.0" --index-url "https://www.paddlepaddle.org.cn/packages/stable/$PaddleCuda/"
& uv pip install --python $python "SAM-2 @ git+https://github.com/facebookresearch/sam2.git"
& $python -c "import paddle, torch; assert paddle.is_compiled_with_cuda(); assert torch.cuda.is_available(); print('GPU frameworks ready:', torch.cuda.get_device_name(0))"
