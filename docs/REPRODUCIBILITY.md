# REPRODUCIBILITY

## Environment

- Python 3.11+
- Install pinned dependencies:

```bash
python -m pip install -r release/kaggle/requirements.txt -c release/kaggle/constraints.txt
```

## Canonical Artifacts

Use the pinned artifacts in:

- `release/kaggle/artifacts/`
- `release/manifest.json`

## Reproduce ORACC Memory

```bash
python src/fetch_oracc_memory.py --record-id 17220688 --output-csv data/processed/oracc_evacun_memory.csv --meta-json artifacts/analysis/oracc_evacun_memory.meta.json
```

Then verify against the expected hash in `release/manifest.json`.

## Reproduce Canonical Submission (Seed43)

```bash
python src/make_submission.py --method retrieval_routed_reranked --memory oracc_best --routing-map artifacts/profiles/routing_map.json --reranker-canonical seed43 --verify-determinism --output artifacts/submissions/submission_routed_reranked_pairwise_combined.csv
```

## Reproduce Probe Submission (Seed44)

```bash
python src/make_submission.py --method retrieval_routed_reranked --memory oracc_best --routing-map artifacts/profiles/routing_map.json --reranker-canonical seed44 --verify-determinism --output artifacts/submissions/submission_routed_reranked_pairwise_combined_seed44.csv
```

## Run Demo/UI

```bash
python app/corpus/query_api.py
```

Open:

- `http://127.0.0.1:50001/corpus/mim`
- `http://127.0.0.1:50001/corpus/help`

## Smoke Test

```bash
bash scripts/smoke_test.sh
```

