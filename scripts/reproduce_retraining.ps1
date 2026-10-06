param([string]$Config = 'configs/retrain_canonical.json', [string]$ArtifactRoot, [ValidateSet('sft','frozen')][string]$Mode = 'sft')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path (Get-Location) 'work/tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'work/cache/matplotlib'
$python = Join-Path (Get-Location) 'work/.venv/Scripts/python.exe'
if (-not $ArtifactRoot) { $ArtifactRoot = 'artifacts/reproductions/' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
# Override only the output root; keep all declared training/data/hyperparameter settings.
$configObject = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
$configObject.artifact_root = $ArtifactRoot
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
$resolved = Join-Path $ArtifactRoot 'input_config.json'
[IO.File]::WriteAllText((Join-Path (Get-Location) $resolved), ($configObject | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
function Run-Step([string[]]$Arguments) {
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + ($Arguments -join ' ')) }
}
Run-Step -Arguments @('-m','decisionos_sre','--config',$resolved,'train','--mode',$Mode)
$artifact = Join-Path $ArtifactRoot $Mode
Run-Step -Arguments @('-m','decisionos_sre','--config',$resolved,'calibrate','--artifact',$artifact,'--device','cpu')
Run-Step -Arguments @('-m','decisionos_sre','--config',$resolved,'select-policy','--artifact',$artifact,'--device','cpu')
Run-Step -Arguments @('-m','decisionos_sre','--config',$resolved,'evaluate','--artifact',$artifact,'--device','cpu')
Run-Step -Arguments @('-m','decisionos_sre','--config',$resolved,'benchmark','--artifact',$artifact)
Run-Step -Arguments @('scripts/verify_integration.py',$artifact)
Write-Output ('Completed artifact: ' + $artifact)
