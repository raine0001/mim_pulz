from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

# Windows consoles default to cp1252; force UTF-8 so sample prints don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mim_pulz.data import load_deep_past_competition
from mim_pulz.config import PATHS
# from mim_pulz.baseline_rerank import DualTfidfLexRerankTranslator
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator
# from mim_pulz.baseline_segment import SegmentStitchTranslator
# from mim_pulz.baseline_ibm_rerank import DualTfidfIbmRerankTranslator
# from mim_pulz.baseline_tone import ToneAwareDualTfidfTranslator
# from mim_pulz.baseline_gate import GatedDualTfidfTranslator


def main() -> None:
    # Load data using the same schema as submit.py
    comp_dir = PATHS.data_raw / "competition"
    schema_path = PATHS.root / "config" / "schema.json"
    data = load_deep_past_competition(comp_dir, schema_path)

    df = data.train.copy()
    text_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col

    # Basic cleanup
    df[text_col] = df[text_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")

    # Reproducible split
    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    val_frac = 0.20
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Fit baseline
    model = CanonicalRetrievalTranslator(config=CANONICAL_BASELINE_V2)
    # model = DualTfidfLexRerankTranslator(top_k=80)  // results  35.12
    # model = DualTfidfIbmRerankTranslator(top_k=120)  // results 24.44
    # model = ToneAwareDualTfidfTranslator(top_k=80, w_len=0.30, w_tone=0.40)  // results 36.22
    # model = GatedDualTfidfTranslator(top_k=120)  // results 34.03

    print("using model:", type(model).__name__)

    model.fit(
        train_src=train_df[text_col].tolist(),
        train_tgt=train_df[tgt_col].tolist(),
    )

    preds = model.predict(val_df[text_col].tolist())


    # Score with chrF (character F-score)
    metric = CHRF(word_order=2)  # common default
    score = metric.corpus_score(preds, [val_df[tgt_col].tolist()])

    print(f"Validation size: {len(val_df)} / Train size: {len(train_df)}")
    print(f"chrF2 score: {score.score:.4f}")

    # Show a few examples
    print("\n--- Samples (pred vs gold) ---")
    for i in range(min(5, len(val_df))):
        print(f"\n[{i}] Transliteration:\n{val_df.loc[i, text_col]}")
        print(f"\nGOLD:\n{val_df.loc[i, tgt_col]}")
        print(f"\nPRED:\n{preds[i]}")


if __name__ == "__main__":
    main()
