param([string]$Config='configs/mvp.json')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:MPLCONFIGDIR = (Join-Path (Get-Location) 'work/cache/matplotlib')
$env:TEMP = (Join-Path (Get-Location) 'work/tmp')
$env:TMP = $env:TEMP
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
$python = '.\work\.venv\Scripts\python.exe'
function Run-Step([string[]]$Arguments) {
    & $python -m decisionos_sre --config $Config @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + ($Arguments -join ' ')) }
}
Run-Step -Arguments @('audit-data')
Run-Step -Arguments @('make-splits')
Run-Step -Arguments @('prepare-data')
& $python -m pytest -q --junitxml=outputs/test-results.xml
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
Run-Step -Arguments @('baseline')
$cfg=Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
foreach ($mode in @('frozen','sft')) {
    $artifact=Join-Path $cfg.artifact_root $mode
    Run-Step -Arguments @('train','--mode',$mode)
    Run-Step -Arguments @('calibrate','--artifact',$artifact,'--device',$cfg.train_device)
    Run-Step -Arguments @('select-policy','--artifact',$artifact,'--device',$cfg.train_device)
    Run-Step -Arguments @('evaluate','--artifact',$artifact,'--device','cpu')
    Run-Step -Arguments @('benchmark','--artifact',$artifact)
    & $python scripts/verify_integration.py $artifact
    if ($LASTEXITCODE -ne 0) { throw 'Integration verification failed' }
}

& $python scripts/generate_report.py $Config
if ($LASTEXITCODE -ne 0) { throw "Report generation failed" }
