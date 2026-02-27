# RELEASE NOTES

## v1.0.0 (2026-02-26)

### Summary

First formal release with split packaging for:

- Kaggle reproducible submission path
- Local demo/UI method-trace path

### Canonical Selection

- Canonical reranker seed: `seed43`
- Canonical submission file:
  - `artifacts/submissions/submission_routed_reranked_pairwise_combined.csv`
  - SHA256: `1b7c5f3456cbb588a9ae1ea8e8534532daf9bd381cb683c2db40bfc8628fd7bd`

### Validation Scores Across Seeds (Combined Metric)

Source artifacts:

- `artifacts/analysis/routed_linear_reranker_eval_seed42_pairwise_combined.json`
- `artifacts/analysis/routed_linear_reranker_eval_seed43_pairwise_combined.json`
- `artifacts/analysis/routed_linear_reranker_eval_seed44_pairwise_combined.json`

Per-seed combined metrics:

- `seed42`: baseline `25.7525` -> reranked(gated) `25.9014` (delta `+0.1489`)
- `seed43`: baseline `24.4867` -> reranked(gated) `24.7872` (delta `+0.3005`)
- `seed44`: baseline `23.6451` -> reranked(gated) `23.7683` (delta `+0.1231`)

Mean combined delta across seeds: `+0.1908`

### Probe Variant

- Known probe variant: `seed44`
- Probe submission file:
  - `artifacts/submissions/submission_routed_reranked_pairwise_combined_seed44.csv`
  - SHA256: `8b59ef6954961bfb32cc3fe481083518a16bf2a38222703e66752968b92fd494`

### Test SHA Difference Notes

- Canonical alias and explicit `seed43` submission hashes are identical:
  - `submission_routed_reranked_pairwise_combined.csv`
  - `submission_routed_reranked_pairwise_combined_seed43.csv`
  - SHA256: `1b7c5f3456cbb588a9ae1ea8e8534532daf9bd381cb683c2db40bfc8628fd7bd`
- `seed44` differs from canonical/seed43:
  - SHA256: `8b59ef6954961bfb32cc3fe481083518a16bf2a38222703e66752968b92fd494`

### Public Leaderboard

- Public LB score: not recorded in repository at release cut.
- Update this section when submission score is published.

### Release Registry

- Canonical hash + environment registry:
  - `release/manifest.json`

