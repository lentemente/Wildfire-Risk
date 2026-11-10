# Build the wildfire risk map end to end, then serve the viewer.
#   .\run.ps1            run every step, then open the map
#   .\run.ps1 -Serve     just serve the existing map
param([switch]$Serve, [int]$Port = 8765)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    python -m venv .venv
    & $py -m pip install --no-input --disable-pip-version-check -r requirements.txt
}

if (-not $Serve) {
    foreach ($step in '01_fetch', '02_grids', '02b_response', '03_wind', '04_spread', '05_simulate', '05b_priority', '06_export') {
        Write-Host "=== $step ===" -ForegroundColor Cyan
        & $py -u "pipeline\$step.py"
        if ($LASTEXITCODE -ne 0) { throw "$step failed" }
    }
}

Write-Host "Map at http://localhost:$Port  (Ctrl+C to stop)" -ForegroundColor Green
Start-Process "http://localhost:$Port"
& $py -m http.server $Port --directory web --bind 127.0.0.1
