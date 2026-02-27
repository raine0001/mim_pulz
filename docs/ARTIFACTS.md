# ARTIFACTS

## Canonical Registry

Authoritative registry:

- `release/manifest.json`

This file pins:

- canonical submission hash (seed43)
- probe submission hash (seed44)
- routing map hash
- reranker model hashes
- ORACC memory hash and row count
- retrieval profile hash
- Python version and git commit marker

## Release Package Paths

- `release/kaggle/artifacts/routing_map.json`
- `release/kaggle/artifacts/reranker_seed43.json`
- `release/kaggle/artifacts/reranker_seed44.json`
- `release/kaggle/artifacts/retrieval_oracc_memory.yaml`
- `release/kaggle/artifacts/oracc_memory.csv`

## Submission Outputs

Primary output directory:

- `artifacts/submissions/`

Typical files:

- `submission_*.csv`
- `submission_*.csv.sha256`
- `submission_*.csv.meta.json`
- `submission_*.csv.manifest.json`
- `submission_*.csv.evidence.json`
- `submission_*.routing.json`

## Notes

- Use `--verify-determinism` when creating submissions.
- Keep submission CSV, hash, and manifest from the same run together.
- Treat seed43 as canonical and seed44 as probe unless explicitly overridden.

