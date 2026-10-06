param([string]$Config='configs/mvp.json')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
$python = '.\work\.venv\Scripts\python.exe'
function Run-Step([string[]]$Arguments) {
    & $python -m decisionos_sre --config $Config @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + ($Arguments -join ' ')) }
}
Run-Step @('audit-data')
Run-Step @('make-splits')
Run-Step @('prepare-data')
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
Run-Step @('baseline')
$cfg=Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
foreach ($mode in @('frozen','sft')) {
    $artifact=Join-Path $cfg.artifact_root $mode
    Run-Step @('train','--mode',$mode)
    Run-Step @('calibrate','--artifact',$artifact,'--device',$cfg.train_device)
    Run-Step @('select-policy','--artifact',$artifact,'--device',$cfg.train_device)
    Run-Step @('evaluate','--artifact',$artifact,'--device','cpu')
    Run-Step @('benchmark','--artifact',$artifact)
}
