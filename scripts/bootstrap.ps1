[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$SkipSam2
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv не найден. Установите его: winget install Astral-sh.uv'
}

uv venv --python 3.11
if ($SkipSam2) {
    uv sync --extra ocr --extra vision
} else {
    uv sync --extra all-models
}

if ($Gpu) {
    & "$PSScriptRoot\install_gpu.ps1"
}

Write-Host 'Environment ready. Start: .venv\Scripts\uvicorn.exe labelbench.api:app --host 127.0.0.1 --port 8000'
