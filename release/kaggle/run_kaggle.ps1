param(
  [ValidateSet("fetch-oracc", "canonical", "probe", "all")]
  [string]$Mode = "canonical",
  [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($InstallDeps) {
  python -m pip install -r requirements.txt -c constraints.txt
}

function Get-CompetitionDir {
  $kaggleInput = "/kaggle/input"
  if (Test-Path $kaggleInput) {
    $cands = Get-ChildItem -Path $kaggleInput -Directory -ErrorAction SilentlyContinue
    foreach ($cand in $cands) {
      if ($cand.FullName -like "*/datasets*") { continue }
      $sample = Join-Path $cand.FullName "sample_submission.csv"
      $train = Join-Path $cand.FullName "train.csv"
      $test = Join-Path $cand.FullName "test.csv"
      if ((Test-Path $sample) -and (Test-Path $train) -and (Test-Path $test)) {
        return $cand.FullName
      }
    }
  }
  return (Join-Path $PSScriptRoot "data/raw/competition")
}

$competitionDir = Get-CompetitionDir
Write-Host "Using competition dir: $competitionDir"

function Fetch-Oracc {
  python src/fetch_oracc_memory.py `
    --record-id 17220688 `
    --output-csv artifacts/oracc_memory.csv `
    --meta-json artifacts/oracc_memory.meta.json
}

function Build-Seed43 {
  python src/make_submission.py `
    --method retrieval_routed_reranked `
    --memory oracc_best `
    --competition-dir $competitionDir `
    --routing-map artifacts/routing_map.json `
    --reranker-canonical seed43 `
    --verify-determinism `
    --output submission.csv
}

function Build-Seed44 {
  python src/make_submission.py `
    --method retrieval_routed_reranked `
    --memory oracc_best `
    --competition-dir $competitionDir `
    --routing-map artifacts/routing_map.json `
    --reranker-canonical seed44 `
    --verify-determinism `
    --output submission_probe.csv
}

switch ($Mode) {
  "fetch-oracc" { Fetch-Oracc }
  "canonical" { Build-Seed43 }
  "probe" { Build-Seed44 }
  "all" {
    Fetch-Oracc
    Build-Seed43
    Build-Seed44
  }
}

