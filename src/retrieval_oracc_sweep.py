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
    return {
        "config": asdict(cfg),
        "score": score,
        "override_count": override_count,
        "override_rate": float(override_count / len(debug_rows)) if debug_rows else 0.0,
        "external_pick_count": external_pick_count,
        "external_pick_rate": float(external_pick_count / len(debug_rows)) if debug_rows else 0.0,
        "memory_total_rows": int(len(model.train_src_raw)),
        "memory_external_rows": int(model.external_rows_loaded),
        "external_memory_stats": model.external_memory_stats,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep ORACC-augmented retrieval memory with context/origin calibration."
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
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_oracc_sweep_seed42.json",
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

    canonical_no_ext = _evaluate(
        CANONICAL_BASELINE_V2,
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
    )

    base_ext_cfg = RetrievalConfig(
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
        evidence_k=1,
    )
    baseline_ext = _evaluate(
        base_ext_cfg,
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
    )

    memory_limit_grid = [5_000, 20_000, 0]
    comp_bonus_grid = [0.0, 0.03, 0.06]
    ext_bonus_grid = [0.0, -0.02]
    stage2_pool_grid = [50, 80]
    domain_bonus_grid = [0.0, 0.03]
    conf_grid = [0.35, 0.45]
    margin_grid = [0.02]
    ctx_allow_grid = [
        (),
        ("economic", "legal", "letter"),
        ("economic", "legal"),
        ("legal", "letter"),
    ]

    rows = []
    for mem_limit, comp_bonus, ext_bonus, pool, dom_bonus, conf_thr, margin, ctx_allow in product(
        memory_limit_grid,
        comp_bonus_grid,
        ext_bonus_grid,
        stage2_pool_grid,
        domain_bonus_grid,
        conf_grid,
        margin_grid,
        ctx_allow_grid,
    ):
        cfg = RetrievalConfig(
            analyzer=base_ext_cfg.analyzer,
            ngram_min=base_ext_cfg.ngram_min,
            ngram_max=base_ext_cfg.ngram_max,
            max_features=base_ext_cfg.max_features,
            top_k=base_ext_cfg.top_k,
            len_weight=base_ext_cfg.len_weight,
            len_mode=base_ext_cfg.len_mode,
            strip_punct=base_ext_cfg.strip_punct,
            lowercase=base_ext_cfg.lowercase,
            collapse_whitespace=base_ext_cfg.collapse_whitespace,
            stage2_type=base_ext_cfg.stage2_type,
            stage2_pool=int(pool),
            stage2_weight=base_ext_cfg.stage2_weight,
            stage2_bm25_k1=base_ext_cfg.stage2_bm25_k1,
            stage2_bm25_b=base_ext_cfg.stage2_bm25_b,
            enable_domain_override=True,
            domain_candidate_top_k=base_ext_cfg.domain_candidate_top_k,
            domain_bonus=float(dom_bonus),
            domain_conf_threshold=float(conf_thr),
            domain_margin=float(margin),
            external_memory_paths=base_ext_cfg.external_memory_paths,
            external_source_col=base_ext_cfg.external_source_col,
            external_target_col=base_ext_cfg.external_target_col,
            external_context_col=base_ext_cfg.external_context_col,
            external_origin_col=base_ext_cfg.external_origin_col,
            external_memory_limit=int(mem_limit),
            external_context_allowlist=tuple(ctx_allow),
            competition_origin_bonus=float(comp_bonus),
            external_origin_bonus=float(ext_bonus),
            evidence_k=1,
        )
        result = _evaluate(
            cfg,
            train_src=train_src,
            train_tgt=train_tgt,
            val_src=val_src,
            val_tgt=val_tgt,
        )
        result["delta_vs_no_external"] = float(result["score"]["chrf2"] - canonical_no_ext["score"]["chrf2"])
        result["delta_vs_external_baseline"] = float(result["score"]["chrf2"] - baseline_ext["score"]["chrf2"])
        rows.append(result)

    rows = sorted(rows, key=lambda x: x["score"]["chrf2"], reverse=True)
    best = rows[0]

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "memory_csv": str(args.memory_csv.resolve()),
        "canonical_no_external": canonical_no_ext,
        "external_baseline": baseline_ext,
        "search_space": {
            "memory_limit": memory_limit_grid,
            "competition_origin_bonus": comp_bonus_grid,
            "external_origin_bonus": ext_bonus_grid,
            "stage2_pool": stage2_pool_grid,
            "domain_bonus": domain_bonus_grid,
            "domain_conf_threshold": conf_grid,
            "domain_margin": margin_grid,
            "external_context_allowlist": [list(x) for x in ctx_allow_grid],
        },
        "best": best,
        "top20": rows[:20],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print(f"No-external baseline chrF2\t{canonical_no_ext['score']['chrf2']:.4f}")
    print(f"External baseline chrF2\t{baseline_ext['score']['chrf2']:.4f}")
    print(f"Best ORACC sweep chrF2\t{best['score']['chrf2']:.4f}")
    print(f"Delta vs no-external\t{best['delta_vs_no_external']:.4f}")
    print(f"Delta vs external baseline\t{best['delta_vs_external_baseline']:.4f}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
