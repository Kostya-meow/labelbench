[CmdletBinding()]
param(
    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = 'Stop'
if ($WaitForProcessId -gt 0 -and (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue)) {
    Wait-Process -Id $WaitForProcessId
}

& "$PSScriptRoot\install_gpu.ps1"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
& $python "$PSScriptRoot\prefetch_models.py" --providers ppocr mask2former sam2
