param(
  [ValidateSet("fetch-oracc", "canonical", "probe", "all")]
  [string]$Mode = "canonical"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install -r requirements.txt -c constraints.txt

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
    --routing-map artifacts/routing_map.json `
    --reranker-canonical seed43 `
    --verify-determinism `
    --output submission.csv
}

function Build-Seed44 {
  python src/make_submission.py `
    --method retrieval_routed_reranked `
    --memory oracc_best `
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

