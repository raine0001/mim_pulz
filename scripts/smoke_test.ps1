param(
  [switch]$InstallDeps,
  [ValidateSet("app", "kaggle")]
  [string]$Target = "app"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if ($InstallDeps) {
  if ($Target -eq "app") {
    Write-Host "[0/4] install app dependencies"
    python -m pip install -r requirements.app.txt
  }
  else {
    Write-Host "[0/4] install kaggle dependencies"
    python -m pip install -r requirements.kaggle.txt
  }
}

Write-Host "[1/4] compile check"
python -m compileall app/corpus/query_api.py src/make_submission.py

Write-Host "[2/4] deterministic submission build"
python src/make_submission.py --verify-determinism --output outputs/submission_smoke.csv

Write-Host "[3/4] app endpoint checks"
$server = Start-Process -FilePath python -ArgumentList "app/corpus/query_api.py" -PassThru -WindowStyle Hidden
try {
  $deadline = (Get-Date).AddSeconds(30)
  $ok = $false
  while ((Get-Date) -lt $deadline) {
    try {
      $health = Invoke-RestMethod -Uri "http://127.0.0.1:50001/corpus/health" -TimeoutSec 5
      if ($health.ok -eq $true) { $ok = $true; break }
    }
    catch {}
    Start-Sleep -Milliseconds 500
  }
  if (-not $ok) { throw "health endpoint did not become ready in time" }
  Write-Host "health ok"

  $demo = Invoke-RestMethod -Uri "http://127.0.0.1:50001/corpus/demo?limit=3" -TimeoutSec 15
  if ($demo.ok -ne $true) { throw "demo endpoint returned unexpected payload" }
  Write-Host "demo ok"
}
finally {
  if ($server -and -not $server.HasExited) {
    Stop-Process -Id $server.Id -Force
  }
}

Write-Host "smoke test passed"

