from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.domain_intent import infer_dialog_domain
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, RetrievalConfig, CanonicalRetrievalTranslator
from utils_manifest import write_json


def _split(df, seed: int, val_frac: float):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def _score(preds: list[str], gold: list[str]) -> dict:
    metric = CHRF(word_order=2)
    s1 = float(metric.corpus_score(preds, [gold]).score)
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return {
        "chrf2": s1,
        "chrf2_reference": s2,
        "abs_diff": float(abs(s1 - s2)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence-gated context override on top of canonical retrieval baseline."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument(
        "--retrieval-memory-csv",
        action="append",
        default=None,
        help="Optional external retrieval memory table path (repeat for multiple files).",
    )
    parser.add_argument("--retrieval-memory-source-col", type=str, default="source")
    parser.add_argument("--retrieval-memory-target-col", type=str, default="target")
    parser.add_argument("--retrieval-memory-context-col", type=str, default="context")
    parser.add_argument("--retrieval-memory-origin-col", type=str, default="origin")
    parser.add_argument("--retrieval-memory-limit", type=int, default=0)
    parser.add_argument(
        "--retrieval-memory-context-allow",
        action="append",
        default=None,
        help="Optional context allowlist for external memory (repeatable).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "domain_retrieval_eval_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_deep_past_competition(args.competition_dir, args.schema)
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col
    df = data.train.copy()
    df[src_col] = df[src_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")
    train_df, val_df = _split(df, seed=args.seed, val_frac=args.val_frac)
    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()

    memory_paths = tuple(str(Path(p).resolve()) for p in (args.retrieval_memory_csv or []))
    base_cfg = RetrievalConfig(
        analyzer=CANONICAL_BASELINE_V2.analyzer,
        ngram_min=CANONICAL_BASELINE_V2.ngram_min,
        ngram_max=CANONICAL_BASELINE_V2.ngram_max,
        max_features=CANONICAL_BASELINE_V2.max_features,
        top_k=CANONICAL_BASELINE_V2.top_k,
        len_weight=CANONICAL_BASELINE_V2.len_weight,
        len_mode=CANONICAL_BASELINE_V2.len_mode,
        strip_punct=CANONICAL_BASELINE_V2.strip_punct,
        lowercase=CANONICAL_BASELINE_V2.lowercase,
        collapse_whitespace=CANONICAL_BASELINE_V2.collapse_whitespace,
        stage2_type=CANONICAL_BASELINE_V2.stage2_type,
        stage2_pool=CANONICAL_BASELINE_V2.stage2_pool,
        stage2_weight=CANONICAL_BASELINE_V2.stage2_weight,
        stage2_bm25_k1=CANONICAL_BASELINE_V2.stage2_bm25_k1,
        stage2_bm25_b=CANONICAL_BASELINE_V2.stage2_bm25_b,
        enable_domain_override=CANONICAL_BASELINE_V2.enable_domain_override,
        domain_candidate_top_k=CANONICAL_BASELINE_V2.domain_candidate_top_k,
        domain_bonus=CANONICAL_BASELINE_V2.domain_bonus,
        domain_conf_threshold=CANONICAL_BASELINE_V2.domain_conf_threshold,
        domain_margin=CANONICAL_BASELINE_V2.domain_margin,
        external_memory_paths=memory_paths,
        external_source_col=args.retrieval_memory_source_col,
        external_target_col=args.retrieval_memory_target_col,
        external_context_col=args.retrieval_memory_context_col,
        external_origin_col=args.retrieval_memory_origin_col,
        external_memory_limit=int(args.retrieval_memory_limit),
        external_context_allowlist=tuple(args.retrieval_memory_context_allow or []),
        evidence_k=1,
    )

    base_model = CanonicalRetrievalTranslator(config=base_cfg)
    base_model.fit(train_src=train_src, train_tgt=train_tgt)
    base_preds, base_debug = base_model.predict(val_src, return_debug=True)
    base_score = _score(base_preds, val_tgt)
    base_override_count = int(sum(1 for r in base_debug if r["override_used"]))

    rows = []
    bonus_grid = [0.02, 0.04, 0.06, 0.08]
    conf_grid = [0.45, 0.55, 0.65]
    margin_grid = [0.02, 0.05, 0.10]
    domain_k_grid = [120, 200, 300]
    for domain_bonus, conf_thr, margin, domain_k in product(
        bonus_grid, conf_grid, margin_grid, domain_k_grid
    ):
        cfg = RetrievalConfig(
            analyzer=base_cfg.analyzer,
            ngram_min=base_cfg.ngram_min,
            ngram_max=base_cfg.ngram_max,
            max_features=base_cfg.max_features,
            top_k=base_cfg.top_k,
            len_weight=base_cfg.len_weight,
            len_mode=base_cfg.len_mode,
            strip_punct=base_cfg.strip_punct,
            lowercase=base_cfg.lowercase,
            collapse_whitespace=base_cfg.collapse_whitespace,
            stage2_type=base_cfg.stage2_type,
            stage2_pool=base_cfg.stage2_pool,
            stage2_weight=base_cfg.stage2_weight,
            stage2_bm25_k1=base_cfg.stage2_bm25_k1,
            stage2_bm25_b=base_cfg.stage2_bm25_b,
            enable_domain_override=True,
            domain_candidate_top_k=int(domain_k),
            domain_bonus=float(domain_bonus),
            domain_conf_threshold=float(conf_thr),
            domain_margin=float(margin),
            external_memory_paths=base_cfg.external_memory_paths,
            external_source_col=base_cfg.external_source_col,
            external_target_col=base_cfg.external_target_col,
            external_context_col=base_cfg.external_context_col,
            external_origin_col=base_cfg.external_origin_col,
            external_memory_limit=base_cfg.external_memory_limit,
            external_context_allowlist=base_cfg.external_context_allowlist,
            evidence_k=1,
        )
        model = CanonicalRetrievalTranslator(config=cfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, debug_rows = model.predict(val_src, return_debug=True)
        score = _score(preds, val_tgt)
        override_count = int(sum(1 for r in debug_rows if r["override_used"]))
        rows.append(
            {
                "config": asdict(cfg),
                "score": score,
                "delta_vs_baseline": float(score["chrf2"] - base_score["chrf2"]),
                "override_count": override_count,
                "override_rate": float(override_count / len(debug_rows)) if debug_rows else 0.0,
            }
        )

    rows = sorted(rows, key=lambda x: x["score"]["chrf2"], reverse=True)
    top = rows[0]

    domain_counts = {}
    for s in train_src:
        d = infer_dialog_domain(s)
        domain_counts[d] = domain_counts.get(d, 0) + 1

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "baseline_config": asdict(base_cfg),
        "baseline_model_id": base_cfg.model_id(),
        "baseline": base_score,
        "baseline_override_count": base_override_count,
        "baseline_memory_total_rows": int(len(base_model.train_src_raw)),
        "baseline_memory_external_rows": int(base_model.external_rows_loaded),
        "baseline_external_memory_stats": base_model.external_memory_stats,
        "domain_counts_train": domain_counts,
        "best_gated_domain_override": top,
        "top10_gated_domain_override": rows[:10],
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print("Baseline chrF2\t{:.4f}".format(base_score["chrf2"]))
    print("Best gated override chrF2\t{:.4f}".format(top["score"]["chrf2"]))
    print("Delta vs baseline\t{:.4f}".format(top["delta_vs_baseline"]))
    print(f"Best config\t{top['config']}")
    print("Override rate\t{:.3f}".format(top["override_rate"]))
    print(f"Metric abs diff (baseline)\t{base_score['abs_diff']:.8f}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
