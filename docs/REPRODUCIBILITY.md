# REPRODUCIBILITY

## Environment

- Python 3.11+
- Install pinned dependencies:

```bash
python -m pip install -r release/kaggle/requirements.txt -c release/kaggle/constraints.txt
```

- Golden lock snapshot (known-good machine state):
	- `release/kaggle/requirements.lock.txt`
	- `release/demo/requirements.lock.txt`
	- root fallback: `requirements.lock.txt`

- Target-specific dependency files:
	- `requirements.kaggle.txt`
	- `requirements.app.txt`

## Clean-Room Package Verification

Validate packages from scratch using fresh venvs and packaged scripts:

```bash
# Kaggle package
python -m venv .venv_pkg_kaggle
. .venv_pkg_kaggle/bin/activate
python -m pip install -r release/kaggle/requirements.txt -c release/kaggle/constraints.txt
powershell -ExecutionPolicy Bypass -File release/kaggle/run_kaggle.ps1 -Mode canonical
powershell -ExecutionPolicy Bypass -File release/kaggle/run_kaggle.ps1 -Mode probe

# Demo package
python -m venv .venv_pkg_demo
. .venv_pkg_demo/bin/activate
python -m pip install -r release/demo/requirements.txt
powershell -ExecutionPolicy Bypass -File release/demo/run_demo.ps1
```

Hash verification command (must report `mismatch_count=0`):

```powershell
$m=Get-Content release\manifest.json|ConvertFrom-Json
$items=@($m.canonical.routing_map,$m.canonical.reranker_seed43,$m.canonical.reranker_seed44,$m.canonical.oracc_memory,$m.canonical.submission_seed43,$m.canonical.submission_seed44_probe)
$bad=0
foreach($i in $items){
	$p=$i.path
	if(Test-Path $p){
		$a=(Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
		if($a -ne $i.sha256.ToLower()){ $bad++ }
	} else { $bad++ }
}
"mismatch_count=$bad"
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

