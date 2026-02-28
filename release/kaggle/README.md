# Kaggle Release Package (Self-Contained)

This package is a standalone submission generator intended to run without the full project repo.

## Scope

- Deterministic Kaggle submission build
- Canonical routed+rerranked retrieval path
- Pinned canonical artifacts (routing map, reranker, ORACC memory profile)
- Hash-verifiable outputs and manifest alignment

## Install

From inside `release/kaggle/`:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

## Canonical Commands (from this folder)

1. Fetch ORACC memory (public Zenodo source):

```bash
python src/fetch_oracc_memory.py --record-id 17220688 --output-csv artifacts/oracc_memory.csv --meta-json artifacts/oracc_memory.meta.json
```

2. Build canonical submission (seed43):

```bash
python src/make_submission.py --method retrieval_routed_reranked --memory oracc_best --routing-map artifacts/routing_map.json --reranker-canonical seed43 --verify-determinism --output submission.csv
```

3. Build probe submission (seed44):

```bash
python src/make_submission.py --method retrieval_routed_reranked --memory oracc_best --routing-map artifacts/routing_map.json --reranker-canonical seed44 --verify-determinism --output submission_probe.csv
```

## One-Command Wrappers

- POSIX shell wrapper: `release/kaggle/run_kaggle.sh`
- PowerShell wrapper: `release/kaggle/run_kaggle.ps1`

Examples (run from this folder):

```bash
bash run_kaggle.sh canonical
bash run_kaggle.sh probe
bash run_kaggle.sh all
```

## Packaged Artifacts

Pinned copies in `release/kaggle/artifacts/`:

- `routing_map.json`
- `reranker_seed43.json`
- `reranker_seed44.json`
- `retrieval_oracc_memory.yaml`
- `oracc_memory.csv`

Authoritative hash registry:

- `release/manifest.json`

## External Data Disclosure (ORACC Memory)

- Source: Zenodo record `17220688`
- Fetch script: `src/fetch_oracc_memory.py`
- Output CSV: `artifacts/oracc_memory.csv`
- Expected rows: `44937`
- Expected SHA256: `c79a18608276d023af7a3aa9db783edaaa37d8055992e295ab6026f9189947de`

Re-generate and verify:

```bash
python src/fetch_oracc_memory.py --record-id 17220688 --output-csv artifacts/oracc_memory.csv --meta-json artifacts/oracc_memory.meta.json
python -c "from hashlib import sha256; import pathlib; p=pathlib.Path('artifacts/oracc_memory.csv'); print(sha256(p.read_bytes()).hexdigest())"
```

## Kaggle Notebook Usage

Typical notebook flow:

```bash
cp -r /kaggle/input/<your-dataset>/release/kaggle /kaggle/working/kaggle_pkg
cd /kaggle/working/kaggle_pkg
bash run_kaggle.sh canonical
```

Upload `/kaggle/working/kaggle_pkg/submission.csv`.
```
