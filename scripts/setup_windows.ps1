$ErrorActionPreference = 'Stop'
$previousPythonUtf8 = $env:PYTHONUTF8
$previousPipProgress = $env:PIP_PROGRESS_BAR
$env:PYTHONUTF8 = '1'
$env:PIP_PROGRESS_BAR = 'off'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

try {
    if (-not (Test-Path $venvPython)) {
        py -3.10 -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
    }

    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Upgrading pip failed.' }
    & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-cu121.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Installing the CUDA dependencies failed.' }
    & $venvPython -m pip install --no-deps -e $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Installing the project failed.' }

    Write-Host 'Environment installed. Activate with: .\.venv\Scripts\Activate.ps1'
    Write-Host 'Check CUDA with: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
}
finally {
    if ($null -eq $previousPythonUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue }
    else { $env:PYTHONUTF8 = $previousPythonUtf8 }
    if ($null -eq $previousPipProgress) { Remove-Item Env:PIP_PROGRESS_BAR -ErrorAction SilentlyContinue }
    else { $env:PIP_PROGRESS_BAR = $previousPipProgress }
}
