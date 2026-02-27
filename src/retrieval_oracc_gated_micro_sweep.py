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
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator, RetrievalConfig
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


def _evaluate(
    cfg: RetrievalConfig,
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
) -> dict:
    model = CanonicalRetrievalTranslator(config=cfg)
    model.fit(train_src=train_src, train_tgt=train_tgt)
    preds, debug_rows = model.predict(val_src, return_debug=True)
    score = _score(preds, val_tgt)
    override_count = int(sum(1 for r in debug_rows if r["override_used"]))
    external_pick_count = int(sum(1 for r in debug_rows if str(r.get("chosen_origin", "")) != "competition"))
    external_allowed_count = int(sum(1 for r in debug_rows if bool(r.get("external_allowed", False))))
    return {
        "config": asdict(cfg),
        "score": score,
        "override_count": override_count,
        "override_rate": float(override_count / len(debug_rows)) if debug_rows else 0.0,
        "external_pick_count": external_pick_count,
        "external_pick_rate": float(external_pick_count / len(debug_rows)) if debug_rows else 0.0,
        "external_allowed_count": external_allowed_count,
        "external_allowed_rate": float(external_allowed_count / len(debug_rows)) if debug_rows else 0.0,
        "memory_total_rows": int(len(model.train_src_raw)),
        "memory_external_rows": int(model.external_rows_loaded),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Micro sweep for ORACC fallback gating: threshold x cap x bonus (36 runs)."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--memory-csv",
        type=Path,
        default=PATHS.data_processed / "oracc_evacun_memory.csv",
    )
    parser.add_argument("--memory-source-col", type=str, default="source")
    parser.add_argument("--memory-target-col", type=str, default="target")
    parser.add_argument("--memory-context-col", type=str, default="context")
    parser.add_argument("--memory-origin-col", type=str, default="origin")
    parser.add_argument("--memory-limit", type=int, default=20_000)
    parser.add_argument(
        "--context-allow",
        action="append",
        default=None,
        help="External context allowlist. Repeatable. Defaults to legal+letter.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_oracc_gated_micro_sweep_seed42.json",
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

    context_allow = tuple(args.context_allow) if args.context_allow else ("legal", "letter")

    no_external = _evaluate(
        CANONICAL_BASELINE_V2,
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
    )

    ungated_external_cfg = RetrievalConfig(
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
        external_memory_paths=(str(args.memory_csv.resolve()),),
        external_source_col=args.memory_source_col,
        external_target_col=args.memory_target_col,
        external_context_col=args.memory_context_col,
        external_origin_col=args.memory_origin_col,
        external_memory_limit=int(args.memory_limit),
        external_context_allowlist=context_allow,
        evidence_k=1,
    )
    ungated_external = _evaluate(
        ungated_external_cfg,
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
    )

    threshold_grid = [0.10, 0.12, 0.14, 0.16]
    cap_grid = [10, 25, 50]
    bonus_grid = [0.00, 0.01, 0.02]

    rows = []
    for t, cap, bonus in product(threshold_grid, cap_grid, bonus_grid):
        cfg = RetrievalConfig(
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
            external_memory_paths=(str(args.memory_csv.resolve()),),
            external_source_col=args.memory_source_col,
            external_target_col=args.memory_target_col,
            external_context_col=args.memory_context_col,
            external_origin_col=args.memory_origin_col,
            external_memory_limit=int(args.memory_limit),
            external_context_allowlist=context_allow,
            external_enable_fallback=True,
            external_internal_top_threshold=float(t),
            external_internal_gap_threshold=-1.0,
            external_candidate_cap=int(cap),
            external_gate_bonus=float(bonus),
            external_gate_margin=0.0,
            evidence_k=1,
        )
        result = _evaluate(
            cfg,
            train_src=train_src,
            train_tgt=train_tgt,
            val_src=val_src,
            val_tgt=val_tgt,
        )
        result["delta_vs_no_external"] = float(result["score"]["chrf2"] - no_external["score"]["chrf2"])
        result["delta_vs_ungated_external"] = float(result["score"]["chrf2"] - ungated_external["score"]["chrf2"])
        rows.append(result)
        print(
            f"T={t:.2f} cap={cap:>2d} bonus={bonus:.2f} "
            f"chrF2={result['score']['chrf2']:.4f} "
            f"delta_no_ext={result['delta_vs_no_external']:+.4f} "
            f"ext_allow={result['external_allowed_rate']:.3f} ext_pick={result['external_pick_rate']:.3f}"
        )

    rows = sorted(rows, key=lambda x: x["score"]["chrf2"], reverse=True)
    best = rows[0]
    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "memory_csv": str(args.memory_csv.resolve()),
        "memory_limit": int(args.memory_limit),
        "context_allow": list(context_allow),
        "no_external_baseline": no_external,
        "ungated_external_baseline": ungated_external,
        "grid": {
            "threshold": threshold_grid,
            "cap": cap_grid,
            "bonus": bonus_grid,
        },
        "best": best,
        "all_rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print("")
    print(f"No-external baseline chrF2\t{no_external['score']['chrf2']:.4f}")
    print(f"Ungated ORACC baseline chrF2\t{ungated_external['score']['chrf2']:.4f}")
    print(f"Best gated ORACC chrF2\t{best['score']['chrf2']:.4f}")
    print(f"Delta vs no-external\t{best['delta_vs_no_external']:.4f}")
    print(f"Delta vs ungated ORACC\t{best['delta_vs_ungated_external']:.4f}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
