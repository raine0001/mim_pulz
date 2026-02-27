from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, profile_source, resolve_routing_map_path
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


def _score(preds: list[str], gold: list[str]) -> dict[str, float]:
    return corpus_metrics(preds, gold)


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

    token_freq = build_token_frequency(train_src)
    internal_debug = policy_debug["internal_only"]

    preds: list[str] = []
    rows: list[dict] = []
    route_counts: dict[str, int] = {}

    for i, src in enumerate(val_src):
        features, labels = profile_source(
            src,
            token_freq=token_freq,
            profile_thresholds=loaded_map.profile_thresholds,
        )
        idebug = internal_debug[i]
        internal_top_score = float(idebug.get("internal_top_cosine", idebug.get("global_score", 0.0)))
        raw_gap = idebug.get("internal_gap", None)
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=labels,
            internal_top_score=internal_top_score,
            internal_gap=internal_gap,
            routing_map=loaded_map.routing_map,
        )
        policy_name = str(decision["policy_name"])
        route_name = str(decision["route"])
        dbg = policy_debug[policy_name][i]
        pred = policy_preds[policy_name][i]

        preds.append(pred)
        route_counts[route_name] = route_counts.get(route_name, 0) + 1
        rows.append(
            {
                "row_index": int(i),
                "gold": val_tgt[i],
                "prediction": pred,
                "route": route_name,
                "policy_name": policy_name,
                "labels": labels,
                "query_section": str(dbg.get("query_section", "body")),
                "query_fragmentation": str(dbg.get("query_fragmentation", labels.get("fragmentation", "unknown"))),
                "internal_top_cosine": float(dbg.get("internal_top_cosine", 0.0)),
                "internal_gap": dbg.get("internal_gap", None),
                "high_uncertainty": bool(dbg.get("high_uncertainty", False)),
                "uncertainty_ratio": float(dbg.get("uncertainty_profile", {}).get("uncertainty_ratio", 0.0)),
            }
        )

    metrics = _score(preds, val_tgt)
    return {
        "metrics": metrics,
        "chrf2": float(metrics["chrf2"]),
        "predictions": preds,
        "rows": rows,
        "route_counts": route_counts,
        "slot_fidelity": slot_fidelity_metrics(preds, val_tgt),
    }


def _count_routes(rows: list[dict], idxs: list[int]) -> dict[str, int]:
    ctr = Counter(rows[i]["route"] for i in idxs)
    return dict(sorted(ctr.items()))


def _route_shift(base_counts: dict[str, int], gated_counts: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(base_counts).union(gated_counts))
    return {k: int(gated_counts.get(k, 0) - base_counts.get(k, 0)) for k in keys}


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate bucket-gated uncertainty adaptation (fragmentation+ambiguity) for routed retrieval."
    )
    p.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    p.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    p.add_argument("--routing-map", type=Path, default=None)
    p.add_argument("--uncertainty-high-threshold", type=float, default=0.03)
    p.add_argument("--uncertainty-closing-threshold", type=float, default=0.02)
    p.add_argument("--uncertainty-bracket-percentile", type=float, default=80.0)
    p.add_argument("--uncertainty-internal-top-threshold", type=float, default=0.48)
    p.add_argument("--uncertainty-internal-gap-threshold", type=float, default=0.01)
    p.add_argument("--uncertainty-topk-add", type=int, default=20)
    p.add_argument("--uncertainty-external-cap-add", type=int, default=2)
    p.add_argument("--uncertainty-topk-boost", type=float, default=0.0)
    p.add_argument("--uncertainty-external-bonus", type=float, default=0.0)
    p.add_argument("--uncertainty-stage2-discount", type=float, default=0.0)
    p.add_argument("--uncertainty-len-discount", type=float, default=0.0)
    p.add_argument("--uncertainty-triggered-len-ratio-min", type=float, default=0.65)
    p.add_argument("--uncertainty-variant-bonus", type=float, default=0.002)
    p.add_argument(
        "--bucket-csv-out",
        type=Path,
        default=PATHS.root / "artifacts" / "profiles" / "reports" / "val_bucket_metrics.csv",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "uncertainty_bucket_gate_seed42.json",
    )
    return p


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
    gated_cfg = _config_with_overrides(
        base_cfg,
        enable_uncertainty_adaptation=True,
        uncertainty_high_threshold=float(args.uncertainty_high_threshold),
        uncertainty_closing_threshold=float(args.uncertainty_closing_threshold),
        uncertainty_bracket_percentile=float(args.uncertainty_bracket_percentile),
        uncertainty_internal_top_threshold=float(args.uncertainty_internal_top_threshold),
        uncertainty_internal_gap_threshold=float(args.uncertainty_internal_gap_threshold),
        uncertainty_topk_add=int(args.uncertainty_topk_add),
        uncertainty_external_cap_add=int(args.uncertainty_external_cap_add),
        uncertainty_topk_boost=float(args.uncertainty_topk_boost),
        uncertainty_external_bonus=float(args.uncertainty_external_bonus),
        uncertainty_stage2_discount=float(args.uncertainty_stage2_discount),
        uncertainty_len_discount=float(args.uncertainty_len_discount),
        uncertainty_triggered_len_ratio_min=float(args.uncertainty_triggered_len_ratio_min),
        uncertainty_variant_bonus=float(args.uncertainty_variant_bonus),
    )

    base_eval = _evaluate_routed(
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
        base_cfg=base_cfg,
        loaded_map=loaded_map,
    )
    gated_eval = _evaluate_routed(
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
        base_cfg=gated_cfg,
        loaded_map=loaded_map,
    )

    n_rows = len(val_src)
    if len(base_eval["rows"]) != n_rows or len(gated_eval["rows"]) != n_rows:
        raise RuntimeError("Row count mismatch in routed evaluations.")

    dims = ["fragmentation", "uncertainty_bucket", "query_section"]
    bucket_rows: list[dict] = []
    for dim in dims:
        if dim == "fragmentation":
            values = sorted({str(r["labels"].get("fragmentation", "unknown")) for r in base_eval["rows"]})
        elif dim == "uncertainty_bucket":
            values = ["high", "low"]
        else:
            values = sorted({str(r.get("query_section", "unknown")) for r in base_eval["rows"]})
        for v in values:
            if dim == "fragmentation":
                idxs = [i for i, r in enumerate(base_eval["rows"]) if str(r["labels"].get("fragmentation", "unknown")) == v]
            elif dim == "uncertainty_bucket":
                idxs = [
                    i
                    for i, r in enumerate(base_eval["rows"])
                    if ("high" if float(r.get("uncertainty_ratio", 0.0)) >= float(args.uncertainty_high_threshold) else "low") == v
                ]
            else:
                idxs = [i for i, r in enumerate(base_eval["rows"]) if str(r.get("query_section", "unknown")) == v]
            if not idxs:
                continue
            base_preds = [base_eval["predictions"][i] for i in idxs]
            gated_preds = [gated_eval["predictions"][i] for i in idxs]
            gold = [val_tgt[i] for i in idxs]
            base_metrics = _score(base_preds, gold)
            gated_metrics = _score(gated_preds, gold)
            base_slot = slot_fidelity_metrics(base_preds, gold)
            gated_slot = slot_fidelity_metrics(gated_preds, gold)
            base_counts = _count_routes(base_eval["rows"], idxs)
            gated_counts = _count_routes(gated_eval["rows"], idxs)
            base_trigger = float(sum(1 for i in idxs if bool(base_eval["rows"][i]["high_uncertainty"])) / len(idxs))
            gated_trigger = float(sum(1 for i in idxs if bool(gated_eval["rows"][i]["high_uncertainty"])) / len(idxs))
            bucket_rows.append(
                {
                    "dimension": dim,
                    "bucket": v,
                    "n": int(len(idxs)),
                    "baseline_bleu": float(base_metrics["bleu"]),
                    "baseline_chrF2": float(base_metrics["chrf2"]),
                    "baseline_combined": float(base_metrics["combined"]),
                    "bucket_gated_uncertainty_bleu": float(gated_metrics["bleu"]),
                    "bucket_gated_uncertainty_chrF2": float(gated_metrics["chrf2"]),
                    "bucket_gated_uncertainty_combined": float(gated_metrics["combined"]),
                    "delta_bleu": float(gated_metrics["bleu"] - base_metrics["bleu"]),
                    "delta_chrF2": float(gated_metrics["chrf2"] - base_metrics["chrf2"]),
                    "delta_combined": float(gated_metrics["combined"] - base_metrics["combined"]),
                    "baseline_digit_preservation": float(base_slot["digit_preservation_recall"]),
                    "gated_digit_preservation": float(gated_slot["digit_preservation_recall"]),
                    "baseline_measure_unit_preservation": float(base_slot["measure_unit_preservation_recall"]),
                    "gated_measure_unit_preservation": float(gated_slot["measure_unit_preservation_recall"]),
                    "baseline_witness_date_preservation": float(base_slot["witness_date_marker_preservation_recall"]),
                    "gated_witness_date_preservation": float(gated_slot["witness_date_marker_preservation_recall"]),
                    "baseline_trigger_rate": base_trigger,
                    "gated_trigger_rate": gated_trigger,
                    "baseline_route_counts": json.dumps(base_counts, sort_keys=True),
                    "gated_route_counts": json.dumps(gated_counts, sort_keys=True),
                    "route_shift": json.dumps(_route_shift(base_counts, gated_counts), sort_keys=True),
                }
            )

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_csv_path = args.bucket_csv_out if args.bucket_csv_out.is_absolute() else (repo_root / args.bucket_csv_out)
    bucket_csv_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_df.to_csv(bucket_csv_path, index=False)

    gated_rows = gated_eval["rows"]
    base_rows = base_eval["rows"]
    gated_triggered = [r for r in gated_rows if bool(r["high_uncertainty"])]
    base_triggered = [r for r in base_rows if bool(r["high_uncertainty"])]

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "memory_profile": str(args.memory_profile),
        "train_rows": int(len(train_df)),
        "val_rows": int(n_rows),
        "routing_map_path": str(loaded_map.path.resolve()),
        "routing_map_sha256": loaded_map.sha256,
        "baseline_config": asdict(base_cfg),
        "bucket_gated_uncertainty_config": asdict(gated_cfg),
        "overall": {
            "baseline": base_eval["metrics"],
            "bucket_gated_uncertainty": gated_eval["metrics"],
            "delta_bleu": float(gated_eval["metrics"]["bleu"] - base_eval["metrics"]["bleu"]),
            "delta_chrF2": float(gated_eval["metrics"]["chrf2"] - base_eval["metrics"]["chrf2"]),
            "delta_combined": float(gated_eval["metrics"]["combined"] - base_eval["metrics"]["combined"]),
            "baseline_chrF2": float(base_eval["metrics"]["chrf2"]),
            "bucket_gated_uncertainty_chrF2": float(gated_eval["metrics"]["chrf2"]),
        },
        "slot_fidelity": {
            "baseline": base_eval["slot_fidelity"],
            "bucket_gated_uncertainty": gated_eval["slot_fidelity"],
            "delta_digit_preservation": float(
                gated_eval["slot_fidelity"]["digit_preservation_recall"]
                - base_eval["slot_fidelity"]["digit_preservation_recall"]
            ),
            "delta_measure_unit_preservation": float(
                gated_eval["slot_fidelity"]["measure_unit_preservation_recall"]
                - base_eval["slot_fidelity"]["measure_unit_preservation_recall"]
            ),
            "delta_witness_date_preservation": float(
                gated_eval["slot_fidelity"]["witness_date_marker_preservation_recall"]
                - base_eval["slot_fidelity"]["witness_date_marker_preservation_recall"]
            ),
        },
        "route_usage": {
            "baseline": base_eval["route_counts"],
            "bucket_gated_uncertainty": gated_eval["route_counts"],
            "shift": _route_shift(base_eval["route_counts"], gated_eval["route_counts"]),
        },
        "trigger_telemetry": {
            "baseline_trigger_count": int(len(base_triggered)),
            "baseline_trigger_rate": float(len(base_triggered) / n_rows) if n_rows else 0.0,
            "gated_trigger_count": int(len(gated_triggered)),
            "gated_trigger_rate": float(len(gated_triggered) / n_rows) if n_rows else 0.0,
            "baseline_avg_internal_top_all": _safe_mean([float(r["internal_top_cosine"]) for r in base_rows]),
            "baseline_avg_internal_top_triggered": _safe_mean([float(r["internal_top_cosine"]) for r in base_triggered]),
            "gated_avg_internal_top_all": _safe_mean([float(r["internal_top_cosine"]) for r in gated_rows]),
            "gated_avg_internal_top_triggered": _safe_mean([float(r["internal_top_cosine"]) for r in gated_triggered]),
        },
        "bucket_metrics_csv": str(bucket_csv_path.resolve()),
    }

    out_path = args.json_out if args.json_out.is_absolute() else (repo_root / args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)

    print(f"Baseline routed BLEU\t{base_eval['metrics']['bleu']:.4f}")
    print(f"Baseline routed chrF2\t{base_eval['metrics']['chrf2']:.4f}")
    print(f"Baseline routed combined\t{base_eval['metrics']['combined']:.4f}")
    print(f"Bucket-gated uncertainty BLEU\t{gated_eval['metrics']['bleu']:.4f}")
    print(f"Bucket-gated uncertainty chrF2\t{gated_eval['metrics']['chrf2']:.4f}")
    print(f"Bucket-gated uncertainty combined\t{gated_eval['metrics']['combined']:.4f}")
    print(f"Delta combined\t{gated_eval['metrics']['combined'] - base_eval['metrics']['combined']:.4f}")
    print(f"Gated uncertainty trigger rate\t{summary['trigger_telemetry']['gated_trigger_rate']:.4f}")
    print(f"Wrote\t{out_path}")
    print(f"Wrote\t{bucket_csv_path}")


if __name__ == "__main__":
    main()

