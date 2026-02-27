from __future__ import annotations
from pathlib import Path

from mim_pulz.config import is_kaggle, PATHS
from mim_pulz.utils import ensure_dir
from mim_pulz.data import load_deep_past_competition
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator


def _guess_competition_dir() -> Path:
    if is_kaggle():
        roots = [p for p in PATHS.kaggle_input_root.iterdir() if p.is_dir()]
        if not roots:
            raise FileNotFoundError("No competition dataset found under /kaggle/input/")
        return roots[0]

    local = PATHS.data_raw / "competition"
    if not local.exists():
        raise FileNotFoundError(f"Local competition folder not found: {local}")
    return local


def run(
    output_path: str | Path = "submission.csv",
    dry_run: bool = False,
    dry_rows: int = 8,
) -> Path:
    comp_dir = _guess_competition_dir()
    schema_path = PATHS.root / "config" / "schema.json"

    data = load_deep_past_competition(comp_dir, schema_path)

    # Fit baseline
    model = CanonicalRetrievalTranslator(config=CANONICAL_BASELINE_V2)
    model.fit(
        train_texts=data.train[data.schema.train_text_col].astype(str).tolist(),
        train_targets=data.train[data.schema.train_target_col].astype(str).tolist(),
    )

    test_df = data.test
    sub_template = data.sample_submission
    if dry_run:
        test_df = test_df.head(dry_rows).copy()
        sub_template = sub_template.head(dry_rows).copy()

    preds = model.predict(test_df[data.schema.test_text_col].astype(str).tolist())

    sub = sub_template.copy()
    sub[data.schema.submission_target_col] = preds

    out_dir = PATHS.kaggle_working if is_kaggle() else ensure_dir(PATHS.outputs)
    out_path = Path(out_dir) / Path(output_path).name
    sub.to_csv(out_path, index=False)
    return out_path
