from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys
from typing import Any

import numpy as np


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_uncertainty_bucket_gate import _profile_config, _split
from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.eval_metrics import corpus_metrics, slot_fidelity_metrics
from mim_pulz.retrieval import (
    CanonicalRetrievalTranslator,
    RetrievalConfig,
    _config_with_overrides,
    _route_policy_params,
    build_policy_config,
)
from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, resolve_routing_map_path, profile_source
from structural_profile import build_token_frequency
from utils_manifest import write_json


def _parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [float(x) for x in vals]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Minimal high-signal uncertainty gate sweep (structural bracket percentile + ambiguity thresholds)."
    )
    p.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    p.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    p.add_argument("--optimize-target", type=str, choices=["combined", "chrf2", "bleu"], default="combined")
    p.add_argument("--routing-map", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_map.json")
    p.add_argument("--bracket-percentiles", type=str, default="80,85,90")
    p.add_argument("--internal-top-thresholds", type=str, default="0.48,0.50,0.52")
    p.add_argument("--margin-thresholds", type=str, default="0.01,0.015,0.02")
    p.add_argument("--topk-add", type=int, default=20)
    p.add_argument("--external-cap-add", type=int, default=2)
    p.add_argument("--uncertainty-high-threshold", type=float, default=0.03)
    p.add_argument("--uncertainty-closing-threshold", type=float, default=0.02)
    p.add_argument("--uncertainty-skeleton-blend", type=float, default=0.05)
    p.add_argument("--uncertainty-number-bonus", type=float, default=0.005)
    p.add_argument("--uncertainty-formula-bonus", type=float, default=0.005)
    p.add_argument("--uncertainty-slot-bonus", type=float, default=0.0)
    p.add_argument("--uncertainty-variant-bonus", type=float, default=0.002)
    p.add_argument("--uncertainty-candidate-uncertain-penalty", type=float, default=0.0)
    p.add_argument(
        "--enable-skeleton-retrieval-path",
        dest="enable_skeleton_retrieval_path",
        action="store_true",
    )
    p.add_argument(
        "--no-enable-skeleton-retrieval-path",
        dest="enable_skeleton_retrieval_path",
        action="store_false",
    )
    p.set_defaults(enable_skeleton_retrieval_path=True)
    p.add_argument("--skeleton-candidate-top-k", type=int, default=120)
    p.add_argument(
        "--tweak-len-ratio-list",
        type=str,
        default="0.55,0.60,0.65",
        help="Triggered-only strict length-ratio filter sweep.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "uncertainty_gate_minimal_sweep_seed42.json",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = PATHS.root
    routing_map_path = resolve_routing_map_path(args.routing_map, repo_root)
    loaded_map = load_routing_map(routing_map_path)
    routing_thresholds = dict(loaded_map.routing_map.get("thresholds", {}))

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

    token_freq = build_token_frequency(train_src)
    n_rows = len(val_src)

    base_cfg = _profile_config(args.memory_profile, repo_root)
    route_to_params = {
        route: _route_policy_params(loaded_map.routing_map, route)
        for route in ROUTE_TO_POLICY
    }

    # Fit once per policy; subsequent sweep mutates config only (fit artifacts stay valid).
    policy_models: dict[str, CanonicalRetrievalTranslator] = {}
    for route_name, policy_name in ROUTE_TO_POLICY.items():
        pcfg = build_policy_config(
            base_cfg,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_thresholds,
        )
        model = CanonicalRetrievalTranslator(config=pcfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        policy_models[policy_name] = model

    def run_with_config(run_cfg: RetrievalConfig) -> dict[str, Any]:
        policy_preds: dict[str, list[str]] = {}
        policy_debug: dict[str, list[dict]] = {}
        for route_name, policy_name in ROUTE_TO_POLICY.items():
            pcfg = build_policy_config(
                run_cfg,
                policy_name=policy_name,
                policy_params=route_to_params[route_name],
                routing_thresholds=routing_thresholds,
            )
            model = policy_models[policy_name]
            model.config = pcfg
            preds, dbg = model.predict(val_src, return_debug=True)
            policy_preds[policy_name] = preds
            policy_debug[policy_name] = dbg

        internal_debug = policy_debug["internal_only"]
        final_preds: list[str] = []
        final_debug: list[dict] = []
        for i, src in enumerate(val_src):
            features, labels = profile_source(
                src,
                token_freq=token_freq,
                profile_thresholds=loaded_map.profile_thresholds,
            )
            idbg = internal_debug[i]
            internal_top_score = float(idbg.get("internal_top_cosine", idbg.get("global_score", 0.0)))
            raw_gap = idbg.get("internal_gap", None)
            internal_gap = None if raw_gap is None else float(raw_gap)
            decision = choose_policy(
                labels=labels,
                internal_top_score=internal_top_score,
                internal_gap=internal_gap,
                routing_map=loaded_map.routing_map,
            )
            policy_name = str(decision["policy_name"])
            final_preds.append(policy_preds[policy_name][i])
            final_debug.append(policy_debug[policy_name][i])

        overall_metrics = corpus_metrics(final_preds, val_tgt)
        slot_metrics = slot_fidelity_metrics(final_preds, val_tgt)
        trigger_idxs = [i for i, dbg in enumerate(final_debug) if bool(dbg.get("high_uncertainty", False))]
        trigger_rate = float(len(trigger_idxs) / n_rows) if n_rows else 0.0
        if trigger_idxs:
            pred_trig = [final_preds[i] for i in trigger_idxs]
            gold_trig = [val_tgt[i] for i in trigger_idxs]
            triggered_metrics = corpus_metrics(pred_trig, gold_trig)
            trig_internal_top = float(
                np.mean([float(final_debug[i].get("internal_top_cosine", 0.0)) for i in trigger_idxs])
            )
        else:
            triggered_metrics = {"bleu": 0.0, "chrf2": 0.0, "combined": 0.0}
            trig_internal_top = 0.0

        return {
            "overall_metrics": overall_metrics,
            "slot_fidelity": slot_metrics,
            "predictions": final_preds,
            "debug_rows": final_debug,
            "trigger_idxs": trigger_idxs,
            "trigger_rate": trigger_rate,
            "triggered_metrics": triggered_metrics,
            "triggered_avg_internal_top": trig_internal_top,
        }

    baseline = run_with_config(_config_with_overrides(base_cfg, enable_uncertainty_adaptation=False))

    bracket_percentiles = _parse_float_list(args.bracket_percentiles)
    internal_top_thresholds = _parse_float_list(args.internal_top_thresholds)
    margin_thresholds = _parse_float_list(args.margin_thresholds)

    sweep_rows: list[dict[str, Any]] = []
    for br_pct, top_thr, gap_thr in product(bracket_percentiles, internal_top_thresholds, margin_thresholds):
        cfg = _config_with_overrides(
            base_cfg,
            enable_uncertainty_adaptation=True,
            uncertainty_high_threshold=float(args.uncertainty_high_threshold),
            uncertainty_closing_threshold=float(args.uncertainty_closing_threshold),
            uncertainty_bracket_percentile=float(br_pct),
            uncertainty_internal_top_threshold=float(top_thr),
            uncertainty_internal_gap_threshold=float(gap_thr),
            uncertainty_topk_add=int(args.topk_add),
            uncertainty_external_cap_add=int(args.external_cap_add),
            uncertainty_topk_boost=0.0,
            uncertainty_external_bonus=0.0,
            uncertainty_stage2_discount=0.0,
            uncertainty_len_discount=0.0,
            uncertainty_skeleton_blend=float(args.uncertainty_skeleton_blend),
            uncertainty_number_bonus=float(args.uncertainty_number_bonus),
            uncertainty_formula_bonus=float(args.uncertainty_formula_bonus),
            uncertainty_slot_bonus=float(args.uncertainty_slot_bonus),
            uncertainty_variant_bonus=float(args.uncertainty_variant_bonus),
            uncertainty_candidate_uncertain_penalty=float(args.uncertainty_candidate_uncertain_penalty),
            enable_skeleton_retrieval_path=bool(args.enable_skeleton_retrieval_path),
            skeleton_candidate_top_k=int(args.skeleton_candidate_top_k),
            uncertainty_triggered_len_ratio_min=0.0,
        )
        run = run_with_config(cfg)
        trig_idxs = run["trigger_idxs"]
        if trig_idxs:
            base_trig = [baseline["predictions"][i] for i in trig_idxs]
            gold_trig = [val_tgt[i] for i in trig_idxs]
            baseline_subset_metrics = corpus_metrics(base_trig, gold_trig)
            baseline_same_subset = float(baseline_subset_metrics[str(args.optimize_target)])
            triggered_delta = float(
                run["triggered_metrics"][str(args.optimize_target)] - baseline_same_subset
            )
        else:
            baseline_same_subset = 0.0
            triggered_delta = 0.0
        run_target = float(run["overall_metrics"][str(args.optimize_target)])
        base_target = float(baseline["overall_metrics"][str(args.optimize_target)])
        sweep_rows.append(
            {
                "bracket_percentile": float(br_pct),
                "internal_top_threshold": float(top_thr),
                "margin_threshold": float(gap_thr),
                "trigger_rate": float(run["trigger_rate"]),
                "trigger_n": int(len(trig_idxs)),
                "overall_bleu": float(run["overall_metrics"]["bleu"]),
                "overall_chrF2": float(run["overall_metrics"]["chrf2"]),
                "overall_combined": float(run["overall_metrics"]["combined"]),
                "overall_target": run_target,
                "overall_target_delta": float(run_target - base_target),
                "triggered_bleu": float(run["triggered_metrics"]["bleu"]),
                "triggered_chrF2": float(run["triggered_metrics"]["chrf2"]),
                "triggered_combined": float(run["triggered_metrics"]["combined"]),
                "baseline_target_on_triggered_subset": float(baseline_same_subset),
                "triggered_subset_delta": float(triggered_delta),
            }
        )

    sweep_rows_sorted = sorted(
        sweep_rows,
        key=lambda r: (float(r["overall_target"]), float(r["triggered_subset_delta"])),
        reverse=True,
    )
    best_gate = sweep_rows_sorted[0]

    # Triggered-only tweak: strict length ratio filter.
    tweak_rows: list[dict[str, Any]] = []
    tweak_len_list = _parse_float_list(args.tweak_len_ratio_list)
    for len_min in tweak_len_list:
        cfg = _config_with_overrides(
            base_cfg,
            enable_uncertainty_adaptation=True,
            uncertainty_high_threshold=float(args.uncertainty_high_threshold),
            uncertainty_closing_threshold=float(args.uncertainty_closing_threshold),
            uncertainty_bracket_percentile=float(best_gate["bracket_percentile"]),
            uncertainty_internal_top_threshold=float(best_gate["internal_top_threshold"]),
            uncertainty_internal_gap_threshold=float(best_gate["margin_threshold"]),
            uncertainty_topk_add=int(args.topk_add),
            uncertainty_external_cap_add=int(args.external_cap_add),
            uncertainty_topk_boost=0.0,
            uncertainty_external_bonus=0.0,
            uncertainty_stage2_discount=0.0,
            uncertainty_len_discount=0.0,
            uncertainty_skeleton_blend=float(args.uncertainty_skeleton_blend),
            uncertainty_number_bonus=float(args.uncertainty_number_bonus),
            uncertainty_formula_bonus=float(args.uncertainty_formula_bonus),
            uncertainty_slot_bonus=float(args.uncertainty_slot_bonus),
            uncertainty_variant_bonus=float(args.uncertainty_variant_bonus),
            uncertainty_candidate_uncertain_penalty=float(args.uncertainty_candidate_uncertain_penalty),
            enable_skeleton_retrieval_path=bool(args.enable_skeleton_retrieval_path),
            skeleton_candidate_top_k=int(args.skeleton_candidate_top_k),
            uncertainty_triggered_len_ratio_min=float(len_min),
        )
        run = run_with_config(cfg)
        trig_idxs = run["trigger_idxs"]
        if trig_idxs:
            base_trig = [baseline["predictions"][i] for i in trig_idxs]
            gold_trig = [val_tgt[i] for i in trig_idxs]
            baseline_subset_metrics = corpus_metrics(base_trig, gold_trig)
            baseline_same_subset = float(baseline_subset_metrics[str(args.optimize_target)])
            triggered_delta = float(
                run["triggered_metrics"][str(args.optimize_target)] - baseline_same_subset
            )
        else:
            baseline_same_subset = 0.0
            triggered_delta = 0.0
        run_target = float(run["overall_metrics"][str(args.optimize_target)])
        base_target = float(baseline["overall_metrics"][str(args.optimize_target)])
        tweak_rows.append(
            {
                "triggered_len_ratio_min": float(len_min),
                "trigger_rate": float(run["trigger_rate"]),
                "trigger_n": int(len(trig_idxs)),
                "overall_bleu": float(run["overall_metrics"]["bleu"]),
                "overall_chrF2": float(run["overall_metrics"]["chrf2"]),
                "overall_combined": float(run["overall_metrics"]["combined"]),
                "overall_target": run_target,
                "overall_target_delta": float(run_target - base_target),
                "triggered_bleu": float(run["triggered_metrics"]["bleu"]),
                "triggered_chrF2": float(run["triggered_metrics"]["chrf2"]),
                "triggered_combined": float(run["triggered_metrics"]["combined"]),
                "baseline_target_on_triggered_subset": float(baseline_same_subset),
                "triggered_subset_delta": float(triggered_delta),
            }
        )
    tweak_rows_sorted = sorted(
        tweak_rows,
        key=lambda r: (float(r["overall_target"]), float(r["triggered_subset_delta"])),
        reverse=True,
    )

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "memory_profile": str(args.memory_profile),
        "optimize_target": str(args.optimize_target),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "routing_map_path": str(loaded_map.path.resolve()),
        "routing_map_sha256": loaded_map.sha256,
        "base_config": asdict(base_cfg),
        "baseline": {
            "overall_metrics": baseline["overall_metrics"],
            "slot_fidelity": baseline["slot_fidelity"],
            "overall_target": float(baseline["overall_metrics"][str(args.optimize_target)]),
            "trigger_rate": float(baseline["trigger_rate"]),
            "trigger_n": int(len(baseline["trigger_idxs"])),
        },
        "sweep_space": {
            "bracket_percentiles": bracket_percentiles,
            "internal_top_thresholds": internal_top_thresholds,
            "margin_thresholds": margin_thresholds,
            "topk_add": int(args.topk_add),
            "external_cap_add": int(args.external_cap_add),
            "skeleton_blend": float(args.uncertainty_skeleton_blend),
            "number_bonus": float(args.uncertainty_number_bonus),
            "formula_bonus": float(args.uncertainty_formula_bonus),
            "slot_bonus": float(args.uncertainty_slot_bonus),
            "variant_bonus": float(args.uncertainty_variant_bonus),
            "candidate_uncertain_penalty": float(args.uncertainty_candidate_uncertain_penalty),
            "enable_skeleton_retrieval_path": bool(args.enable_skeleton_retrieval_path),
            "skeleton_candidate_top_k": int(args.skeleton_candidate_top_k),
        },
        "best_gate": best_gate,
        "gate_sweep": sweep_rows_sorted,
        "triggered_tweak": {
            "name": "strict_length_ratio_filter",
            "best_gate_anchor": {
                "bracket_percentile": float(best_gate["bracket_percentile"]),
                "internal_top_threshold": float(best_gate["internal_top_threshold"]),
                "margin_threshold": float(best_gate["margin_threshold"]),
            },
            "rows": tweak_rows_sorted,
        },
    }

    out_path = args.json_out if args.json_out.is_absolute() else (repo_root / args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)

    print(
        "Baseline overall\t"
        f"BLEU={baseline['overall_metrics']['bleu']:.4f},"
        f"chrF2={baseline['overall_metrics']['chrf2']:.4f},"
        f"combined={baseline['overall_metrics']['combined']:.4f},"
        f"target({args.optimize_target})={baseline['overall_metrics'][str(args.optimize_target)]:.4f}"
    )
    print(
        "Best gate\t"
        f"br_pct={best_gate['bracket_percentile']},"
        f"itop<{best_gate['internal_top_threshold']},"
        f"gap<{best_gate['margin_threshold']},"
        f"trigger_rate={best_gate['trigger_rate']:.4f},"
        f"overall_target={best_gate['overall_target']:.4f},"
        f"triggered_delta={best_gate['triggered_subset_delta']:.4f}"
    )
    if tweak_rows_sorted:
        best_tweak = tweak_rows_sorted[0]
        print(
            "Best strict-len tweak\t"
            f"len_min={best_tweak['triggered_len_ratio_min']},"
            f"trigger_rate={best_tweak['trigger_rate']:.4f},"
            f"overall_target={best_tweak['overall_target']:.4f},"
            f"triggered_delta={best_tweak['triggered_subset_delta']:.4f}"
        )
    print(f"Wrote\t{out_path}")


if __name__ == "__main__":
    main()
