param(
  [ValidateSet("fetch-oracc","canonical","probe","all")]
  [string]$Mode = "canonical"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
Set-Location $root

function Fetch-Oracc {
  python src/fetch_oracc_memory.py `
    --record-id 17220688 `
    --output-csv data/processed/oracc_evacun_memory.csv `
    --meta-json artifacts/analysis/oracc_evacun_memory.meta.json
}

function Build-Seed43 {
  python src/make_submission.py `
    --method retrieval_routed_reranked `
    --memory oracc_best `
    --routing-map artifacts/profiles/routing_map.json `
    --reranker-canonical seed43 `
    --verify-determinism `
    --output artifacts/submissions/submission_routed_reranked_pairwise_combined.csv
}

function Build-Seed44 {
  python src/make_submission.py `
    --method retrieval_routed_reranked `
    --memory oracc_best `
    --routing-map artifacts/profiles/routing_map.json `
    --reranker-canonical seed44 `
    --verify-determinism `
    --output artifacts/submissions/submission_routed_reranked_pairwise_combined_seed44.csv
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

