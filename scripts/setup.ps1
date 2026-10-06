$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force work\tmp | Out-Null
$env:TEMP = (Join-Path (Get-Location) 'work\tmp')
$env:TMP = $env:TEMP
if (-not (Test-Path work\.venv\Scripts\python.exe)) {
    python -m venv work\.venv
    if ($LASTEXITCODE -ne 0) { throw 'venv failed' }
}
$python = '.\work\.venv\Scripts\python.exe'
& $python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
if (Test-Path requirements-lock.txt) {
    & $python -m pip install --no-cache-dir -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependencies failed' }
    & $python -m pip install --no-cache-dir --no-deps -e .
} else {
    & $python -m pip install --no-cache-dir -e . pytest==8.3.5 httpx==0.28.1 dulwich==0.22.8
}
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
$env:PYTHONPATH = 'src'
& $python scripts\download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Pinned model download failed' }
