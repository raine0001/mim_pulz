# MIM_pulz - Adaptive AI Interpretation

## Project Overview

This project focuses on developing adaptive AI interpretation techniques for enhanced model understanding and decision-making processes.

## Project Structure

```
MIM_pulz/
├── data/
│   ├── raw/          # Raw datasets
│   └── processed/    # Preprocessed data
├── notebooks/        # Jupyter notebooks for exploration and analysis
├── src/             # Source code modules
├── models/          # Trained models and checkpoints
├── outputs/         # Results, plots, and submissions
├── config/          # Configuration files
└── requirements.txt # Python dependencies
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Place your raw data in the `data/raw/` folder
3. Start with the exploration notebook in `notebooks/`
4. Develop your models using the modules in `src/`

## Key Features

- Adaptive learning mechanisms
- Model interpretation techniques
- Automated feature engineering
- Cross-validation strategies
- Ensemble methods

## Usage

### Run and View the Local App

Start the Flask app from the project root:

```powershell
python app/corpus/query_api.py
```

When running, it serves on:

`http://127.0.0.1:50001`

Important: the root path (`/`) is not defined and will return `Not Found`.
Use the corpus routes below instead.

- Main app UI: `http://127.0.0.1:50001/corpus/mim`
- Corpus API index/help: `http://127.0.0.1:50001/corpus/`
- Full in-app help page: `http://127.0.0.1:50001/corpus/help`
- Health check: `http://127.0.0.1:50001/corpus/health`
- Demo JSON: `http://127.0.0.1:50001/corpus/demo`

If you want to access the app from another device on your LAN, change the host in
`app/corpus/query_api.py` from `127.0.0.1` to `0.0.0.0`, then open:

`http://<your-pc-ip>:50001/corpus/mim`

### Methodology-Visible UI (`/corpus/mim`)

The MIM UI now exposes routing methodology directly in the interface (not just in logs):

- Structural Intelligence Header:
  - `Fragmentation`, `Formula Density`, `Numeric Density`, `Template Type`, `Length Bucket`, `Domain Intent`
- Routing Decision Box:
  - selected policy (`INTERNAL` / `HYBRID` / `FALLBACK` / `STRONG_RERANK`)
  - internal and ORACC proxy scores
  - active threshold values
  - explicit rationale text
- View Mode Toggle:
  - `Internal Only`
  - `Routed (Recommended)`
  - `ORACC Only`
- Structural Filters:
  - fragment level, template type, policy, memory origin
- Routing Summary (Session):
  - per-route usage counts
  - structural distribution summary for current session

Evidence cards and detail drawers also include structural badges and per-item routing context so exemplar selection is traceable.

### Formal Help Documentation

Two formal help surfaces are now available:

- Repository help file: `HELP.md`
- Browser help page: `/corpus/help`

The help docs cover:

- all major UI panels and controls
- evidence navigation behavior and scope
- keyboard shortcuts (`J`, `K`, `/`, `Enter`, `E`, `Esc`)
- visual/source links (`Open visual candidates`, `Compare source vs visual`)
- full corpus browsing workflow (`/corpus/browse`)
- troubleshooting for no-image/context-only cases

### Release Packaging

Two deliverables are packaged separately:

- Kaggle package: `release/kaggle/`
- Demo/UI package: `release/demo/`

Top-level release manifest:

- `release/manifest.json` (canonical hashes for submission, routing map, rerankers, ORACC memory, runtime metadata)
- `RELEASE_NOTES.md` (versioned release summary, canonical seed, validation/per-seed deltas, probe notes)

Version matrix:

| Version | Release Notes | Manifest | Canonical SHA (seed43) | Probe SHA (seed44) |
| --- | --- | --- | --- | --- |
| `v1.0.0` | [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | [`release/manifest.json`](release/manifest.json) | `1b7c5f3456cbb588a9ae1ea8e8534532daf9bd381cb683c2db40bfc8628fd7bd` | `8b59ef6954961bfb32cc3fe481083518a16bf2a38222703e66752968b92fd494` |

Kaggle package includes:

- `release/kaggle/README.md`
- `release/kaggle/run_kaggle.sh` and `release/kaggle/run_kaggle.ps1`
- `release/kaggle/requirements.txt` + `release/kaggle/constraints.txt`
- pinned artifacts in `release/kaggle/artifacts/`

Demo package includes:

- `release/demo/README.md`
- `release/demo/run_demo.sh` and `release/demo/run_demo.ps1`
- `release/demo/requirements.txt`

Winner-grade docs:

- `docs/METHOD.md`
- `docs/REPRODUCIBILITY.md`
- `docs/ARTIFACTS.md`

Authorship and license:

- `AUTHORS.md`
- `LICENSE` (CC BY 4.0)

Smoke tests:

- POSIX shell: `bash scripts/smoke_test.sh`
- PowerShell: `powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1`

### Model Deliverable Workflow (Kaggle)

Canonical config:

- `configs/lora_mt5_small.yaml`
- Current defaults in that config:
  - `max_source_length=512`
  - `max_target_length=256`
  - `truncation_policy=drop_any` (drop train pairs truncated on source or target)
  - `task_prefix="translate transliteration to english:"`
  - `use_domain_tag=true` (adds `<domain=...>` intent tag from heuristics)
  - `no_repeat_ngram_size=3`, `repetition_penalty=1.1`, `early_stopping=true`
  - micro preflight gate enabled (`sample_size=8`, `steps=120`)

Canonical scripts:

- Train: `src/train_lora_mt5.py`
- Eval: `src/eval_models.py`
- Submission: `src/make_submission.py` (default `retrieval_routed_reranked` with canonical `seed43`, optional `--method lora`)
- Manifest helpers: `src/utils_manifest.py`

Canonical baseline (v2):

- Model: `CanonicalRetrievalV2`
- Config: `char TF-IDF`, `ngram=(3,5)`, `top_k=120`, `len_weight=0.4`, `len_mode=ratio`
- Stage-2 reranker: `BM25(char_wb, ngram=(3,5), k1=1.2, b=0.5)`, `stage2_pool=80`, `stage2_weight=0.2`
- Conditional domain override (confidence-gated rerank): `domain_bonus=0.02`, `domain_conf_threshold=0.45`, `domain_margin=0.02`
- Seed `42` validation chrF2: `38.6176`
- Notes: domain override is enabled but only triggers when the domain-reranked candidate beats global by margin; on current `seed=42` val split this remains a safe no-op.

Run a full retrain with gate + artifact output:

```powershell
python src/train_lora_mt5.py --config configs/lora_mt5_small.yaml
```

Run only the micro-overfit preflight (3 variants: full/prefix-on, lora/prefix-on, full/prefix-off control):

```powershell
python src/train_lora_mt5.py --config configs/lora_mt5_small.yaml --preflight-only
```

For diagnostics without failing the process on gate criteria:
`python src/train_lora_mt5.py --config configs/lora_mt5_small.yaml --preflight-only --no-enforce-gate`

Run hard micro diagnostic mode (8 clean samples, 1500 steps, exact-match/edit-distance/len checks, unicode audit):

```powershell
python src/train_lora_mt5.py --config configs/lora_mt5_small.yaml --micro-diagnostic
```

Equivalent Bash wrappers (optional):

```bash
bash scripts/train_lora.sh
bash scripts/eval_compare.sh
bash scripts/make_submission.sh
```

Each run is saved in:

`artifacts/runs/<run_id>/`

Contents include:

- `config.yaml`
- `data_fingerprint.json`
- `training_args.json`
- `trainer_state_summary.json`
- `metrics.json`
- `micro_preflight.json` (when preflight is enabled)
- `micro_diagnostic.json` and `unicode_audit.json` (when `--micro-diagnostic` is used)
- `submission.csv`
- `submission.csv.sha256`
- `manifest.json`

### Evaluation Discipline

Run canonical eval (same split, same metric, spam gate signal):

```powershell
python src/eval_models.py --adapter-dir artifacts/runs/<run_id>/adapter
```

Build the canonical final retrieval submission (default path, routed + reranked, canonical `seed43`):

```powershell
python src/make_submission.py --verify-determinism --output artifacts/submissions/submission_routed_reranked_pairwise_combined.csv
```

Build the strict retrieval-only baseline explicitly:

```powershell
python src/make_submission.py --method retrieval --memory internal --verify-determinism
```

Build a LoRA submission explicitly:

```powershell
python src/make_submission.py --method lora --run-dir artifacts/runs/<run_id> --verify-determinism
```

Run retrieval optimization sweep (char TF-IDF + char BM25, top_k and length tie-break tuning):

```powershell
python src/retrieval_sweep.py --json-out artifacts/analysis/retrieval_sweep_seed42.json
```

Run local retrieval search around the canonical winner (normalization variants + stage2 reranker + domain override tuning):

```powershell
python src/retrieval_local_search.py --json-out artifacts/analysis/retrieval_local_search_seed42.json
```

Run focused surgical sweep (top_k/pool/alpha-or-stage2-weight/length + normalization preselect + dynamic pool check):

```powershell
python src/retrieval_surgical_sweep.py --scoring-mode additive --json-out artifacts/analysis/retrieval_surgical_sweep_seed42_additive.json
python src/retrieval_surgical_sweep.py --scoring-mode alpha_blend --json-out artifacts/analysis/retrieval_surgical_sweep_seed42_alpha.json
```

Run soft domain-separation retrieval evaluation (global fallback + per-domain retrievers):

```powershell
python src/eval_domain_retrieval.py --json-out artifacts/analysis/domain_retrieval_eval_seed42.json
```

Build external ORACC retrieval memory (Zenodo EvaCun parallel corpus, record `17220688`):

```powershell
python src/fetch_oracc_memory.py --record-id 17220688 --output-csv data/processed/oracc_evacun_memory.csv --meta-json artifacts/analysis/oracc_evacun_memory.meta.json
```

Run ORACC-augmented retrieval calibration sweep (memory size + context allowlist + origin calibration + context override):

```powershell
python src/retrieval_oracc_sweep.py --memory-csv data/processed/oracc_evacun_memory.csv --json-out artifacts/analysis/retrieval_oracc_sweep_seed42.json
```

Run focused ORACC search (coordinate + random refinement around the best region):

```powershell
python src/retrieval_oracc_focused_sweep.py --memory-csv data/processed/oracc_evacun_memory.csv --json-out artifacts/analysis/retrieval_oracc_focused_sweep_seed42.json
```

Run the confidence-gated ORACC micro-sweep (`threshold x cap x bonus`, 36 runs):

```powershell
python src/retrieval_oracc_gated_micro_sweep.py --memory-csv data/processed/oracc_evacun_memory.csv --memory-limit 20000 --json-out artifacts/analysis/retrieval_oracc_gated_micro_sweep_seed42.json
```

Run context-override tuning directly with external memory loaded:

```powershell
python src/eval_domain_retrieval.py --retrieval-memory-csv data/processed/oracc_evacun_memory.csv --json-out artifacts/analysis/domain_retrieval_eval_oracc_seed42.json
```

Build submission with ORACC memory via config:

```powershell
python src/make_submission.py --config configs/retrieval_oracc_memory.yaml --verify-determinism
```

Build routed retrieval submission (policy per row from structural routing map):

```powershell
python src/make_submission.py --method retrieval_routed --memory oracc_best --routing-map artifacts/profiles/routing_map.json --verify-determinism
```

Evaluate border-aware section routing (MVP: `closing` vs `body`) on routed retrieval:

```powershell
python src/eval_routed_border.py --memory-profile oracc_best --section-bonus-list 0.0,0.01 --section-min-pool-list 5 --section-force-min-score-list 3.0,3.5,4.0 --section-tail-ratio-list 0.2 --json-out artifacts/analysis/border_section_eval_seed42_v3.json
```

Enable section-border constraints in routed submission (deterministic):

```powershell
python src/make_submission.py --method retrieval_routed --memory oracc_best --routing-map artifacts/profiles/routing_map.json --retrieval-enable-section-border --retrieval-section-force-min-score 4.0 --retrieval-section-closing-tail-ratio 0.2 --retrieval-section-min-pool 5 --retrieval-section-match-bonus 0.0 --verify-determinism
```

Evaluate bucket-gated uncertainty (fragmentation/bracket + ambiguity gate) and refresh bucket report:

```powershell
python src/eval_uncertainty_bucket_gate.py --memory-profile oracc_best --routing-map artifacts/profiles/routing_map.json --json-out artifacts/analysis/uncertainty_bucket_gate_seed42.json --bucket-csv-out artifacts/profiles/reports/val_bucket_metrics.csv
```

Run the minimal high-signal gate sweep (P80/P85/P90, internal-top, margin) plus triggered-only strict length-ratio tweak:

```powershell
python src/sweep_uncertainty_bucket_gate.py --memory-profile oracc_best --routing-map artifacts/profiles/routing_map.json --bracket-percentiles 80,85,90 --internal-top-thresholds 0.48,0.50,0.52 --margin-thresholds 0.01,0.015,0.02 --json-out artifacts/analysis/uncertainty_gate_minimal_sweep_seed42.json
```

Enable bucket-gated uncertainty at submission time (deterministic, conservative deltas):

```powershell
python src/make_submission.py --method retrieval_routed --memory oracc_best --routing-map artifacts/profiles/routing_map.json --retrieval-enable-uncertainty-adaptation --retrieval-uncertainty-high-threshold 0.03 --retrieval-uncertainty-bracket-percentile 80 --retrieval-uncertainty-internal-top-threshold 0.48 --retrieval-uncertainty-internal-gap-threshold 0.01 --retrieval-uncertainty-topk-add 20 --retrieval-uncertainty-external-cap-add 2 --retrieval-uncertainty-triggered-len-ratio-min 0.65 --retrieval-uncertainty-topk-boost 0.0 --retrieval-uncertainty-external-bonus 0.0 --retrieval-uncertainty-stage2-discount 0.0 --retrieval-uncertainty-len-discount 0.0 --verify-determinism
```

Current seed-42 bucket-gated uncertainty result:

- baseline routed chrF2: `38.8710`
- bucket-gated uncertainty chrF2: `38.8928`
- delta: `+0.0217`
- gated trigger rate: `20.83%` (65/312)
- fragmentary bucket delta: `+0.1853` (from `29.6638` to `29.8491`)
- complete bucket delta: `+0.0000` (unchanged at `39.9714`)

Current seed-42 section-border result:

- baseline routed chrF2: `38.8710`
- best section-border chrF2: `38.9322`
- delta: `+0.0612`
- best params: `force_min_score=4.0`, `tail_ratio=0.2`, `min_pool=5`, `section_bonus=0.0`

Optional routed-mode cache + telemetry:

```powershell
python src/make_submission.py --method retrieval_routed --memory oracc_best --routing-map artifacts/profiles/routing_map.json --profiles-cache artifacts/profiles/test_profiles_cache.json --routing-telemetry outputs/submission_routed.routing.json --verify-determinism
```

Export routed candidate pools for reranker training (top internal + capped ORACC, includes baseline selection + per-candidate BLEU/chrF2/combined deltas):

```powershell
python src/export_routed_candidates.py --memory-profile oracc_best --routing-map artifacts/profiles/routing_map.json --internal-top-k 120 --oracc-cap 25 --output-dir artifacts/analysis/routed_candidates_seed42_pairwise_combined
```

Train pairwise linear reranker optimized for combined metric gain (hard-query emphasis + conservative switch gate + digit veto):

```powershell
python src/train_routed_linear_reranker.py --candidates-dir artifacts/analysis/routed_candidates_seed42_pairwise_combined --model-out artifacts/models/routed_linear_reranker_seed42_pairwise_combined.json --report-json artifacts/analysis/routed_linear_reranker_eval_seed42_pairwise_combined.json --optimize-target combined
```

Build reranked routed submission using canonical seed (`seed43` default):

```powershell
python src/make_submission.py --verify-determinism
```

Probe the alternate canonical seed that changes test outputs:

```powershell
python src/make_submission.py --verify-determinism --reranker-canonical seed44 --output artifacts/submissions/submission_routed_reranked_pairwise_combined_seed44.csv
```

Compare canonical seed43 vs seed44 evidence decisions on the test rows:

```powershell
python -c "import json; from pathlib import Path; a=json.loads(Path('artifacts/submissions/submission_routed_reranked_pairwise_combined.csv.evidence.json').read_text(encoding='utf-8'))['rows']; b=json.loads(Path('artifacts/submissions/submission_routed_reranked_pairwise_combined_seed44.csv.evidence.json').read_text(encoding='utf-8'))['rows']; print(sum(1 for x,y in zip(a,b) if x['prediction']!=y['prediction']))"
```

Current pairwise-combined reranker status:

- seed 42: baseline combined `25.7525` -> reranked `25.9014` (`+0.1489`)
- seed 43: baseline combined `24.4867` -> reranked `24.7872` (`+0.3005`)
- seed 44: baseline combined `23.6451` -> reranked `23.7683` (`+0.1231`)
- mean delta combined (`42/43/44`): `+0.1908` (std `0.0783`)
- test submission hashes:
  - seed42/seed43: `1b7c5f3456cbb588a9ae1ea8e8534532daf9bd381cb683c2db40bfc8628fd7bd`
  - seed44: `8b59ef6954961bfb32cc3fe481083518a16bf2a38222703e66752968b92fd494`

`--reranker-model <path>` overrides `--reranker-canonical` when both are provided.

Routed-mode outputs include:

- `submission.csv.evidence.json` (route + labels + policy params + retrieval evidence per row)
- `submission.routing.json` (route usage + routing telemetry)
- manifest/meta fields for routing map path + sha256, route counts, and policy model IDs

Tune routing thresholds (keeps routes fixed, optimizes `internal_top_low/high` on val, and writes tuned map):

```powershell
python scripts/tune_routing_thresholds.py --memory oracc_best --routing-map artifacts/profiles/routing_map.json --output-map artifacts/profiles/routing_map.json --report-json artifacts/profiles/routing_threshold_tuning_oracc_best_seed42.json --seed 42 --val-frac 0.2 --backup-existing-map
```

Current tuned thresholds:

- `internal_top_low=0.50`
- `internal_top_high=0.60`
- validation chrF2 (`routed`): `38.8710`
- test route usage: `RETRIEVE_HYBRID=2`, `RETRIEVE_INTERNAL=2`

Tune routed policy parameters with thresholds fixed (optimizes `P1/P2/P3` policy params on val):

```powershell
python scripts/tune_routing_policy_params.py --memory oracc_best --routing-map artifacts/profiles/routing_map.json --output-map artifacts/profiles/routing_map.json --report-json artifacts/profiles/routing_policy_tuning_oracc_best_seed42.json --seed 42 --val-frac 0.2 --backup-existing-map
```

Current tuned policy params:

- `P1 (RETRIEVE_HYBRID)`: `{"top_k": 80}`
- `P2 (RETRIEVE_ORACC_FALLBACK)`: `{"oracc_cap": 5, "gate": "low_conf"}`
- `P3 (RERANK_STRONG)`: `{"stage2_pool": 80, "stage2_weight": 0.2}`

This prints:

- Baseline identity (`CanonicalRetrievalV2(...)`)
- LoRA base model and decode config
- `Baseline` chrF2
- `LoRA` chrF2
- `Delta (LoRA - baseline)`
- `spam_rate` and gate pass/fail reason

Important:

- The canonical retrieval baseline is `38.6176` chrF2 (`seed=42`, split `val_frac=0.2`) with `CanonicalRetrievalV2`.
- Raw `google/mt5-small` through seq2seq decode is a separate control and is expected to score much lower without task adaptation.
- Retrieval submissions now write evidence artifacts:
  - `outputs/submission.csv.meta.json` includes memory stats and evidence path.
  - `outputs/submission.csv.evidence.json` records per-test exemplar evidence (chosen/global context, scores, origin, top support examples).
- Best ORACC configuration found so far (`seed=42`, `val_frac=0.2`) reaches `38.8528` chrF2 with:
  - `top_k=100`, `len_weight=0.6`, `stage2_pool=40`, `stage2_weight=0.35`
  - external memory limit `5000`, context allowlist `[legal, letter]`
  - origin calibration `competition_origin_bonus=0.01`, `external_origin_bonus=-0.05`
  - domain override disabled (`enable_domain_override=false`)

### Structural Profiling + Routing

Run the structural profiling pass (labels + routing map + bucketed eval):

```powershell
python scripts/profile_corpus.py --seed 42 --val-frac 0.2
```

Outputs:

- `artifacts/profiles/corpus_profiles.parquet`
- `artifacts/profiles/corpus_profiles.jsonl`
- `artifacts/profiles/routing_map.json`
- `artifacts/profiles/profile_report_seed42.json`
- `artifacts/profiles/reports/val_bucket_metrics.csv`
- `artifacts/profiles/reports/routing_ablation.json`
- `artifacts/profiles/profile_samples/*.jsonl`

Current seed-42 run summary:

- `internal_only`: `38.6176` chrF2
- `routed`: `38.7082` chrF2
- `delta`: `+0.0906`

### Reproducibility Checklist

1. Use fixed seed (`42` by default).
2. Save data fingerprint (`train/test/sub hashes + split index hashes`).
3. Save run manifest (`git hash, script hash, config hash, command`).
4. Run with `--verify-determinism`.
5. Archive `submission.csv`, `submission.csv.sha256`, and `manifest.json` from the same `run_id`.

## Results

[Document your findings and performance metrics here]

## Contributing

[Add contribution guidelines if applicable]
