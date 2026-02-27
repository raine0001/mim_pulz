from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

import numpy as np
import sacrebleu


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.eval_metrics import corpus_metrics, slot_fidelity_metrics
from mim_pulz.retrieval import (
    CANONICAL_BASELINE_V2,
    CanonicalRetrievalTranslator,
    RetrievalConfig,
    _config_with_overrides,
    _route_policy_params,
    build_policy_config,
)
from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, resolve_routing_map_path, profile_source
from structural_profile import build_token_frequency
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
    s = corpus_metrics(preds, gold)
    s1 = float(s["chrf2"])
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return {
        "bleu": float(s["bleu"]),
        "chrf2": s1,
        "combined": float(s["combined"]),
        "chrf2_reference": s2,
        "abs_diff": float(abs(s1 - s2)),
    }


def _parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [float(x) for x in vals]


def _parse_int_list(raw: str) -> list[int]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [int(x) for x in vals]


def _profile_config(memory_profile: str, repo_root: Path) -> RetrievalConfig:
    if memory_profile == "internal":
        return CANONICAL_BASELINE_V2
    if memory_profile == "oracc_best":
        return RetrievalConfig(
            analyzer="char_wb",
            ngram_min=3,
            ngram_max=5,
            max_features=300_000,
            top_k=100,
            len_weight=0.6,
            len_mode="ratio",
            strip_punct=False,
            lowercase=False,
            collapse_whitespace=True,
            stage2_type="bm25",
            stage2_pool=40,
            stage2_weight=0.35,
            stage2_bm25_k1=1.2,
            stage2_bm25_b=0.5,
            enable_domain_override=False,
            domain_candidate_top_k=120,
            domain_bonus=0.01,
            domain_conf_threshold=0.25,
            domain_margin=0.02,
            external_memory_paths=(str((repo_root / "data" / "processed" / "oracc_evacun_memory.csv").resolve()),),
            external_source_col="source",
            external_target_col="target",
            external_context_col="context",
            external_origin_col="origin",
            external_memory_limit=5000,
            external_context_allowlist=("legal", "letter"),
            external_candidate_cap=25,
            external_enable_fallback=False,
            external_internal_top_threshold=-1.0,
            external_internal_gap_threshold=-1.0,
            external_force_contexts=(),
            external_gate_bonus=0.0,
            external_gate_margin=0.0,
            competition_origin_bonus=0.01,
            external_origin_bonus=-0.05,
            enable_uncertainty_adaptation=True,
            uncertainty_high_threshold=0.03,
            uncertainty_closing_threshold=0.02,
            uncertainty_bracket_percentile=80.0,
            uncertainty_internal_top_threshold=0.48,
            uncertainty_internal_gap_threshold=0.01,
            uncertainty_topk_add=20,
            uncertainty_external_cap_add=2,
            uncertainty_topk_boost=0.0,
            uncertainty_external_bonus=0.0,
            uncertainty_stage2_discount=0.0,
            uncertainty_len_discount=0.0,
            uncertainty_triggered_len_ratio_min=0.65,
            uncertainty_skeleton_blend=0.05,
            uncertainty_number_bonus=0.005,
            uncertainty_formula_bonus=0.005,
            uncertainty_slot_bonus=0.0,
            uncertainty_variant_bonus=0.002,
            uncertainty_candidate_uncertain_penalty=0.0,
            enable_skeleton_retrieval_path=True,
            skeleton_candidate_top_k=120,
            evidence_k=1,
        )
    raise ValueError(f"Unknown memory profile: {memory_profile}")


def _evaluate_routed(
    *,
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
    base_cfg: RetrievalConfig,
    loaded_map,
    return_rows: bool = False,
) -> dict:
    routing_thresholds = dict(loaded_map.routing_map.get("thresholds", {}))
    route_to_params = {
        route: _route_policy_params(loaded_map.routing_map, route)
        for route in ROUTE_TO_POLICY
    }
    policy_configs = {
        policy_name: build_policy_config(
            base_cfg,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_thresholds,
        )
        for route_name, policy_name in ROUTE_TO_POLICY.items()
    }

    policy_preds: dict[str, list[str]] = {}
    policy_debug: dict[str, list[dict]] = {}
    for policy_name, pcfg in policy_configs.items():
        model = CanonicalRetrievalTranslator(config=pcfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, dbg = model.predict(val_src, return_debug=True)
        policy_preds[policy_name] = preds
        policy_debug[policy_name] = dbg

    internal_debug = policy_debug["internal_only"]
    token_freq = build_token_frequency(train_src)

    final_preds: list[str] = []
    final_debug: list[dict] = []
    route_counts: dict[str, int] = {}
    per_row: list[dict] = []

    for i, src in enumerate(val_src):
        features, labels = profile_source(
            src,
            token_freq=token_freq,
            profile_thresholds=loaded_map.profile_thresholds,
        )
        dbg = internal_debug[i]
        internal_top_score = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
        raw_gap = dbg.get("internal_gap", None)
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=labels,
            internal_top_score=internal_top_score,
            internal_gap=internal_gap,
            routing_map=loaded_map.routing_map,
        )
        policy_name = str(decision["policy_name"])
        route_name = str(decision["route"])
        pred = policy_preds[policy_name][i]
        dbg = policy_debug[policy_name][i]
        final_preds.append(pred)
        final_debug.append(dbg)
        route_counts[route_name] = route_counts.get(route_name, 0) + 1
        if return_rows:
            per_row.append(
                {
                    "row_index": int(i),
                    "query": src,
                    "gold": val_tgt[i],
                    "prediction": pred,
                    "sentence_chrf2": float(sacrebleu.sentence_chrf(pred, [val_tgt[i]], word_order=2).score),
                    "route": route_name,
                    "policy_name": policy_name,
                    "query_section": str(dbg.get("query_section", "body")),
                    "query_context": str(dbg.get("query_context", "unknown")),
                    "section_match_chosen": bool(dbg.get("section_match_chosen", False)),
                    "chosen_origin": str(dbg.get("chosen_origin", "")),
                    "chosen_context": str(dbg.get("chosen_context", "")),
                    "chosen_section": str(dbg.get("chosen_section", "")),
                    "internal_top_cosine": float(dbg.get("internal_top_cosine", 0.0)),
                }
            )

    score = _score(final_preds, val_tgt)
    section_closing_count = int(sum(1 for dbg in final_debug if str(dbg.get("query_section", "")) == "closing"))
    section_match_count = int(sum(1 for dbg in final_debug if bool(dbg.get("section_match_chosen", False))))
    closing_match_count = int(
        sum(
            1
            for dbg in final_debug
            if str(dbg.get("query_section", "")) == "closing" and bool(dbg.get("section_match_chosen", False))
        )
    )

    return {
        "score": score,
        "slot_fidelity": slot_fidelity_metrics(final_preds, val_tgt),
        "route_counts": route_counts,
        "section": {
            "closing_count": section_closing_count,
            "closing_rate": float(section_closing_count / len(final_debug)) if final_debug else 0.0,
            "match_count": section_match_count,
            "match_rate": float(section_match_count / len(final_debug)) if final_debug else 0.0,
            "closing_match_count": closing_match_count,
            "closing_match_rate": float(closing_match_count / section_closing_count) if section_closing_count else 0.0,
        },
        "rows": per_row if return_rows else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate border-aware section routing for routed retrieval (baseline vs section-border)."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    parser.add_argument("--routing-map", type=Path, default=None)
    parser.add_argument("--section-bonus-list", type=str, default="0.04")
    parser.add_argument("--section-min-pool-list", type=str, default="10")
    parser.add_argument("--section-force-min-score-list", type=str, default="2.4")
    parser.add_argument("--section-tail-ratio-list", type=str, default="0.15")
    parser.add_argument("--top-delta-k", type=int, default=20)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "border_section_eval_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = PATHS.root

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

    routing_map_path = resolve_routing_map_path(args.routing_map, repo_root)
    loaded_map = load_routing_map(routing_map_path)

    base_cfg = _profile_config(args.memory_profile, repo_root)
    base_result = _evaluate_routed(
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
        base_cfg=base_cfg,
        loaded_map=loaded_map,
        return_rows=True,
    )

    bonus_list = _parse_float_list(args.section_bonus_list)
    min_pool_list = _parse_int_list(args.section_min_pool_list)
    min_score_list = _parse_float_list(args.section_force_min_score_list)
    tail_list = _parse_float_list(args.section_tail_ratio_list)

    rows: list[dict] = []
    best_full_eval: dict | None = None
    for bonus, min_pool, min_score, tail in product(bonus_list, min_pool_list, min_score_list, tail_list):
        cfg = _config_with_overrides(
            base_cfg,
            enable_section_border=True,
            section_force_closing=True,
            section_force_min_score=float(min_score),
            section_match_bonus=float(bonus),
            section_min_pool=int(min_pool),
            section_closing_tail_ratio=float(tail),
        )
        out = _evaluate_routed(
            train_src=train_src,
            train_tgt=train_tgt,
            val_src=val_src,
            val_tgt=val_tgt,
            base_cfg=cfg,
            loaded_map=loaded_map,
            return_rows=False,
        )
        row_item = (
            {
                "config": {
                    "enable_section_border": True,
                    "section_force_closing": True,
                    "section_force_min_score": float(min_score),
                    "section_match_bonus": float(bonus),
                    "section_min_pool": int(min_pool),
                    "section_closing_tail_ratio": float(tail),
                },
                "score": out["score"],
                "slot_fidelity": out["slot_fidelity"],
                "route_counts": out["route_counts"],
                "section": out["section"],
                "delta_vs_baseline_bleu": float(out["score"]["bleu"] - base_result["score"]["bleu"]),
                "delta_vs_baseline": float(out["score"]["chrf2"] - base_result["score"]["chrf2"]),
                "delta_vs_baseline_combined": float(out["score"]["combined"] - base_result["score"]["combined"]),
            }
        )
        rows.append(row_item)
        if best_full_eval is None or row_item["score"]["combined"] > best_full_eval["score"]["combined"]:
            full_out = _evaluate_routed(
                train_src=train_src,
                train_tgt=train_tgt,
                val_src=val_src,
                val_tgt=val_tgt,
                base_cfg=cfg,
                loaded_map=loaded_map,
                return_rows=True,
            )
            best_full_eval = {
                "config": dict(row_item["config"]),
                "score": dict(row_item["score"]),
                "slot_fidelity": dict(row_item["slot_fidelity"]),
                "route_counts": dict(row_item["route_counts"]),
                "section": dict(row_item["section"]),
                "delta_vs_baseline_bleu": float(row_item["delta_vs_baseline_bleu"]),
                "delta_vs_baseline": float(row_item["delta_vs_baseline"]),
                "delta_vs_baseline_combined": float(row_item["delta_vs_baseline_combined"]),
                "rows": full_out["rows"] or [],
            }

    rows_sorted = sorted(rows, key=lambda r: r["score"]["combined"], reverse=True)
    best = rows_sorted[0] if rows_sorted else None

    row_delta_summary = {
        "count": 0,
        "mean_delta": 0.0,
        "median_delta": 0.0,
        "top_improved": [],
        "top_regressed": [],
        "by_query_section": {},
        "by_route": {},
    }
    if best_full_eval is not None and base_result.get("rows") and best_full_eval.get("rows"):
        b_rows = list(base_result["rows"] or [])
        s_rows = list(best_full_eval["rows"] or [])
        if len(b_rows) == len(s_rows):
            deltas = []
            detailed = []
            by_section: dict[str, list[float]] = {}
            by_route: dict[str, list[float]] = {}
            for i in range(len(b_rows)):
                b = b_rows[i]
                s = s_rows[i]
                d = float(s["sentence_chrf2"] - b["sentence_chrf2"])
                deltas.append(d)
                detailed.append(
                    {
                        "row_index": int(i),
                        "delta_sentence_chrf2": d,
                        "baseline_sentence_chrf2": float(b["sentence_chrf2"]),
                        "section_sentence_chrf2": float(s["sentence_chrf2"]),
                        "query_section": str(s["query_section"]),
                        "baseline_route": str(b["route"]),
                        "section_route": str(s["route"]),
                        "query": str(s["query"]),
                        "gold": str(s["gold"]),
                        "baseline_prediction": str(b["prediction"]),
                        "section_prediction": str(s["prediction"]),
                    }
                )
                by_section.setdefault(str(s["query_section"]), []).append(d)
                by_route.setdefault(str(s["route"]), []).append(d)
            detailed_sorted_up = sorted(detailed, key=lambda r: r["delta_sentence_chrf2"], reverse=True)
            detailed_sorted_down = sorted(detailed, key=lambda r: r["delta_sentence_chrf2"])
            row_delta_summary = {
                "count": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "median_delta": float(np.median(deltas)) if deltas else 0.0,
                "top_improved": detailed_sorted_up[: int(max(1, args.top_delta_k))],
                "top_regressed": detailed_sorted_down[: int(max(1, args.top_delta_k))],
                "by_query_section": {
                    k: {
                        "n": len(v),
                        "mean_delta": float(np.mean(v)) if v else 0.0,
                        "median_delta": float(np.median(v)) if v else 0.0,
                    }
                    for k, v in sorted(by_section.items())
                },
                "by_route": {
                    k: {
                        "n": len(v),
                        "mean_delta": float(np.mean(v)) if v else 0.0,
                        "median_delta": float(np.median(v)) if v else 0.0,
                    }
                    for k, v in sorted(by_route.items())
                },
            }

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "memory_profile": str(args.memory_profile),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "routing_map_path": str(loaded_map.path.resolve()),
        "routing_map_sha256": loaded_map.sha256,
        "base_config": asdict(base_cfg),
        "baseline_routed": base_result,
        "best_section_border": best_full_eval if best_full_eval is not None else best,
        "row_delta_summary": row_delta_summary,
        "sweep_results": rows_sorted,
    }

    out_path = args.json_out if args.json_out.is_absolute() else (repo_root / args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)

    print(f"Baseline routed BLEU\t{base_result['score']['bleu']:.4f}")
    print(f"Baseline routed chrF2\t{base_result['score']['chrf2']:.4f}")
    print(f"Baseline routed combined\t{base_result['score']['combined']:.4f}")
    if best is not None:
        print(f"Best section-border BLEU\t{best['score']['bleu']:.4f}")
        print(f"Best section-border chrF2\t{best['score']['chrf2']:.4f}")
        print(f"Best section-border combined\t{best['score']['combined']:.4f}")
        print(f"Delta combined\t{best['delta_vs_baseline_combined']:.4f}")
        print(f"Delta chrF2\t{best['delta_vs_baseline']:.4f}")
        print(f"Best config\t{best['config']}")
        print(f"Section match rate\t{best['section']['match_rate']:.4f}")
        print(f"Closing match rate\t{best['section']['closing_match_rate']:.4f}")
        print(
            "Witness/date preservation\t"
            f"{best['slot_fidelity']['witness_date_marker_preservation_recall']:.4f}"
        )
    print(f"Wrote\t{out_path}")


if __name__ == "__main__":
    main()

