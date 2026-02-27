from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from sacrebleu.metrics import CHRF
import torch


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.lora_submission import (
    InferenceConfig,
    generate_predictions,
    load_lora_bundle,
    read_epoch_chrf2,
    resolve_default_adapter_dir,
    set_deterministic,
)
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator
from mim_pulz.seq2seq_format import DEFAULT_TASK_PREFIX
from utils_manifest import write_json


@dataclass(frozen=True)
class EvalSummary:
    seed: int
    val_frac: float
    baseline_model_kind: str
    baseline_model_id: str
    baseline_domain_override_rate: float
    baseline_domain_override_count: int
    baseline_use_domain_override: bool
    lora_base_model: str
    lora_num_beams: int
    lora_no_repeat_ngram_size: int
    lora_repetition_penalty: float
    lora_early_stopping: bool
    lora_task_prefix: str
    lora_use_domain_tag: bool
    lora_input_max_length: int
    lora_max_length: int | None
    lora_max_new_tokens: int
    train_rows: int
    val_rows: int
    baseline_chrF2: float
    lora_chrF2: float
    delta_chrF2: float
    spam_rate: float
    spam_count: int
    total_preds: int
    gate_pass: bool
    gate_reason: str
    min_delta_over_baseline: float
    spam_threshold: float
    adapter_dir: str


def split_train_val(df, seed: int, val_frac: float):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, train_idx, val_idx


def _tokenize_for_spam(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÿ]+", (text or "").lower())


def is_spam_output(text: str) -> bool:
    toks = _tokenize_for_spam(text)
    if len(toks) < 6:
        return False

    counts = Counter(toks)
    max_ratio = max(counts.values()) / len(toks)
    if max_ratio >= 0.45:
        return True

    run = 1
    best_run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
    if best_run >= 6:
        return True

    trigrams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    if trigrams:
        uniq_ratio = len(set(trigrams)) / len(trigrams)
        if uniq_ratio <= 0.30:
            return True

    return False


def spam_signature(preds: list[str]) -> tuple[float, list[int]]:
    spam_idx = [i for i, p in enumerate(preds) if is_spam_output(p)]
    rate = (len(spam_idx) / len(preds)) if preds else 0.0
    return rate, spam_idx


def evaluate_models(
    competition_dir: Path,
    schema_path: Path,
    adapter_dir: Path,
    inference: InferenceConfig,
    seed: int = 42,
    val_frac: float = 0.20,
    min_delta_over_baseline: float = 0.1,
    spam_threshold: float = 0.35,
) -> EvalSummary:
    data = load_deep_past_competition(competition_dir, schema_path)
    text_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col

    df = data.train.copy()
    df[text_col] = df[text_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")

    train_df, val_df, _, _ = split_train_val(df, seed=seed, val_frac=val_frac)
    metric = CHRF(word_order=2)

    baseline = CanonicalRetrievalTranslator(config=CANONICAL_BASELINE_V2)
    baseline.fit(
        train_src=train_df[text_col].tolist(),
        train_tgt=train_df[tgt_col].tolist(),
    )
    baseline_preds, baseline_debug = baseline.predict(val_df[text_col].tolist(), return_debug=True)
    baseline_score = metric.corpus_score(baseline_preds, [val_df[tgt_col].tolist()]).score
    baseline_override_count = int(sum(1 for r in baseline_debug if r["override_used"]))
    baseline_override_rate = (baseline_override_count / len(baseline_debug)) if baseline_debug else 0.0

    set_deterministic(seed)
    bundle = load_lora_bundle(adapter_dir=adapter_dir, config=inference)
    lora_preds = generate_predictions(val_df[text_col].tolist(), bundle, inference)
    lora_score = metric.corpus_score(lora_preds, [val_df[tgt_col].tolist()]).score
    spam_rate, spam_idx = spam_signature(lora_preds)

    delta = lora_score - baseline_score
    gate_pass = (delta >= min_delta_over_baseline) and (spam_rate <= spam_threshold)
    reasons = []
    if delta < min_delta_over_baseline:
        reasons.append(f"delta {delta:.4f} < required {min_delta_over_baseline:.4f}")
    if spam_rate > spam_threshold:
        reasons.append(f"spam_rate {spam_rate:.3f} > threshold {spam_threshold:.3f}")
    gate_reason = "pass" if gate_pass else "; ".join(reasons)

    return EvalSummary(
        seed=seed,
        val_frac=val_frac,
        baseline_model_kind="retrieval",
        baseline_model_id=CANONICAL_BASELINE_V2.model_id(),
        baseline_domain_override_rate=baseline_override_rate,
        baseline_domain_override_count=baseline_override_count,
        baseline_use_domain_override=bool(CANONICAL_BASELINE_V2.enable_domain_override),
        lora_base_model=inference.base_model,
        lora_num_beams=inference.num_beams,
        lora_no_repeat_ngram_size=inference.no_repeat_ngram_size,
        lora_repetition_penalty=inference.repetition_penalty,
        lora_early_stopping=inference.early_stopping,
        lora_task_prefix=inference.task_prefix,
        lora_use_domain_tag=inference.use_domain_tag,
        lora_input_max_length=inference.input_max_length,
        lora_max_length=inference.max_length,
        lora_max_new_tokens=inference.max_new_tokens,
        train_rows=len(train_df),
        val_rows=len(val_df),
        baseline_chrF2=baseline_score,
        lora_chrF2=lora_score,
        delta_chrF2=delta,
        spam_rate=spam_rate,
        spam_count=len(spam_idx),
        total_preds=len(lora_preds),
        gate_pass=gate_pass,
        gate_reason=gate_reason,
        min_delta_over_baseline=min_delta_over_baseline,
        spam_threshold=spam_threshold,
        adapter_dir=str(adapter_dir),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical eval path: baseline vs LoRA on the same deterministic split."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--adapter-dir", type=Path, default=resolve_default_adapter_dir())
    parser.add_argument("--base-model", type=str, default="google/mt5-small")
    parser.add_argument("--input-max-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--task-prefix", type=str, default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--use-domain-tag", action="store_true")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--min-delta-over-baseline", type=float, default=0.1)
    parser.add_argument("--spam-threshold", type=float, default=0.35)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    inference = InferenceConfig(
        base_model=args.base_model,
        input_max_length=args.input_max_length,
        max_length=args.max_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping,
        task_prefix=args.task_prefix,
        use_domain_tag=args.use_domain_tag,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    summary = evaluate_models(
        competition_dir=args.competition_dir,
        schema_path=args.schema,
        adapter_dir=args.adapter_dir,
        inference=inference,
        seed=args.seed,
        val_frac=args.val_frac,
        min_delta_over_baseline=args.min_delta_over_baseline,
        spam_threshold=args.spam_threshold,
    )

    print(f"Baseline model kind\t{summary.baseline_model_kind}")
    print(f"Baseline model id\t{summary.baseline_model_id}")
    print(
        "Baseline domain override\t"
        f"enabled={summary.baseline_use_domain_override}, "
        f"rate={summary.baseline_domain_override_rate:.3f}, "
        f"count={summary.baseline_domain_override_count}"
    )
    print(f"LoRA base model\t{summary.lora_base_model}")
    print(
        "LoRA decode config\t"
        f"input_max={summary.lora_input_max_length}, "
        f"max_length={summary.lora_max_length}, "
        f"max_new_tokens={summary.lora_max_new_tokens}, "
        f"beams={summary.lora_num_beams}, "
        f"no_repeat_ngram={summary.lora_no_repeat_ngram_size}, "
        f"repetition_penalty={summary.lora_repetition_penalty}, "
        f"early_stopping={summary.lora_early_stopping}, "
        f"task_prefix={summary.lora_task_prefix!r}, "
        f"use_domain_tag={summary.lora_use_domain_tag}"
    )
    print("")
    print("Model\tchrF2")
    print(f"Baseline({summary.baseline_model_id})\t{summary.baseline_chrF2:.4f}")
    print(f"LoRA({Path(summary.adapter_dir).name})\t{summary.lora_chrF2:.4f}")
    print(f"Delta\t{summary.delta_chrF2:.4f}")
    print("")
    print(f"Spam rate\t{summary.spam_rate:.3f} ({summary.spam_count}/{summary.total_preds})")
    print(f"Gate pass\t{summary.gate_pass}")
    print(f"Gate reason\t{summary.gate_reason}")

    epoch_rows = read_epoch_chrf2(Path(summary.adapter_dir) / "trainer_state.json")
    if epoch_rows:
        print("")
        print("Epoch\tchrF2")
        for row in epoch_rows:
            print(f"{row['epoch']:.1f}\t{row['eval_chrf2']:.4f}")

    if args.json_out is not None:
        write_json(args.json_out, asdict(summary))
        print("")
        print(f"Wrote eval summary: {args.json_out}")


if __name__ == "__main__":
    main()
