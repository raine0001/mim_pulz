#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG_DIR"

python -m pip install -r requirements.txt -c constraints.txt

MODE="${1:-canonical}"

fetch_oracc() {
  python src/fetch_oracc_memory.py \
    --record-id 17220688 \
    --output-csv artifacts/oracc_memory.csv \
    --meta-json artifacts/oracc_memory.meta.json
}

build_seed43() {
  python src/make_submission.py \
    --method retrieval_routed_reranked \
    --memory oracc_best \
    --routing-map artifacts/routing_map.json \
    --reranker-canonical seed43 \
    --verify-determinism \
    --output submission.csv
}

build_seed44() {
  python src/make_submission.py \
    --method retrieval_routed_reranked \
    --memory oracc_best \
    --routing-map artifacts/routing_map.json \
    --reranker-canonical seed44 \
    --verify-determinism \
    --output submission_probe.csv
}

case "$MODE" in
  fetch-oracc)
    fetch_oracc
    ;;
  canonical)
    build_seed43
    ;;
  probe)
    build_seed44
    ;;
  all)
    fetch_oracc
    build_seed43
    build_seed44
    ;;
  *)
    echo "Usage: $0 {fetch-oracc|canonical|probe|all}"
    exit 2
    ;;
esac

