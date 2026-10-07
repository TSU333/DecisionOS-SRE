param([string]$OutputDir, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) 'work/.venv/Scripts/python.exe'
$arguments = @('scripts/reproduce_round13.py')
if ($OutputDir) { $arguments += @('--output-dir', $OutputDir) }
if ($CheckOnly) { $arguments += '--check-only' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Round13 reproduction failed' }
