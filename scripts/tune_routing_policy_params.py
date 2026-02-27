from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from sacrebleu.metrics import CHRF


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from make_submission import _profile_config
from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.retrieval import CanonicalRetrievalTranslator, build_policy_config
from structural_profile import ProfileThresholds, build_token_frequency, extract_source_features, label_features, route_decision


ROUTE_TO_POLICY = {
    "RETRIEVE_INTERNAL": "internal_only",
    "RETRIEVE_HYBRID": "hybrid",
    "RETRIEVE_ORACC_FALLBACK": "fallback",
    "RERANK_STRONG": "strong_rerank",
}


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _split(df, seed: int, val_frac: float):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    return (
        df.iloc[idx[val_size:]].reset_index(drop=True),
        df.iloc[idx[:val_size]].reset_index(drop=True),
    )


def _score_chrf2(preds: list[str], gold: list[str]) -> float:
    return float(CHRF(word_order=2).corpus_score(preds, [gold]).score)


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if str(x).strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(float(x.strip())) for x in str(raw).split(",") if str(x).strip()]


def _profile_thresholds_from_payload(payload: dict[str, Any]) -> ProfileThresholds:
    raw = payload.get("thresholds", {})
    return ProfileThresholds(
        fragment_bracket_p90=float(raw["fragment_bracket_p90"]),
        fragment_unknown_p90=float(raw["fragment_unknown_p90"]),
        formula_high_p85=float(raw["formula_high_p85"]),
        formula_mid_p60=float(raw["formula_mid_p60"]),
        numeric_high_p85=float(raw["numeric_high_p85"]),
        pn_high_p85=float(raw["pn_high_p85"]),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tune routed policy params (P1/P2/P3) with fixed routing thresholds."
    )
    p.add_argument("--routing-map", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_map.json")
    p.add_argument("--output-map", type=Path, default=None, help="Where to write updated map (default: overwrite --routing-map).")
    p.add_argument("--report-json", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_policy_tuning_oracc_best_seed42.json")
    p.add_argument("--backup-existing-map", action="store_true")
    p.add_argument("--memory", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    p.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    p.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--hybrid-top-k", type=str, default="80,100,120,160")
    p.add_argument("--fallback-oracc-cap", type=str, default="5,10,25")
    p.add_argument("--strong-stage2-pool", type=str, default="80,120")
    p.add_argument("--strong-stage2-weight", type=str, default="0.2,0.35,0.5")
    p.add_argument("--topk", type=int, default=10, help="How many best configs to keep in report.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = PATHS.root
    routing_map_path = _resolve(args.routing_map, repo_root)
    output_map_path = _resolve(args.output_map, repo_root) if args.output_map is not None else routing_map_path
    report_path = _resolve(args.report_json, repo_root)

    payload = json.loads(routing_map_path.read_text(encoding="utf-8"))
    routing_map = dict(payload["routing_map"])
    route_defs = dict(routing_map["routes"])
    profile_thr = _profile_thresholds_from_payload(payload)

    if args.backup_existing_map and output_map_path.exists():
        backup = output_map_path.with_name(output_map_path.stem + ".pre_policy_tuned.json")
        if not backup.exists():
            shutil.copyfile(output_map_path, backup)

    data = load_deep_past_competition(_resolve(args.competition_dir, repo_root), _resolve(args.schema, repo_root))
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col

    train_all = data.train.copy()
    train_all[src_col] = train_all[src_col].astype(str).fillna("")
    train_all[tgt_col] = train_all[tgt_col].astype(str).fillna("")

    train_df, val_df = _split(train_all, seed=int(args.seed), val_frac=float(args.val_frac))
    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()

    base_cfg = _profile_config(args.memory, repo_root)
    route_to_params = {
        str(route_def["route"]): dict(route_def.get("policy_params", {}))
        for route_def in route_defs.values()
    }
    internal_cfg = build_policy_config(
        base_cfg,
        policy_name="internal_only",
        policy_params=route_to_params.get("RETRIEVE_INTERNAL", {}),
        routing_thresholds=routing_map.get("thresholds", {}),
    )
    internal_model = CanonicalRetrievalTranslator(config=internal_cfg)
    internal_model.fit(train_src=train_src, train_tgt=train_tgt)
    internal_preds, internal_debug = internal_model.predict(val_src, return_debug=True)

    tok_freq = build_token_frequency(train_src)
    labels = []
    for src in val_src:
        features = extract_source_features(src, token_freq=tok_freq)
        labels.append(label_features(features, profile_thr))

    routes = []
    route_counts: dict[str, int] = {}
    for i, lbl in enumerate(labels):
        dbg = internal_debug[i]
        top = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
        gap = dbg.get("internal_gap", None)
        dec = route_decision(
            lbl,
            internal_top_score=top,
            internal_gap=None if gap is None else float(gap),
            routing_map=routing_map,
        )
        route_name = str(dec["route"])
        routes.append(route_name)
        route_counts[route_name] = route_counts.get(route_name, 0) + 1

    hybrid_top_k_values = _parse_int_list(args.hybrid_top_k)
    fallback_oracc_cap_values = _parse_int_list(args.fallback_oracc_cap)
    strong_stage2_pool_values = _parse_int_list(args.strong_stage2_pool)
    strong_stage2_weight_values = _parse_float_list(args.strong_stage2_weight)

    cache: dict[tuple[str, tuple[tuple[str, Any], ...]], tuple[list[str], str]] = {}

    def get_policy_preds(policy_name: str, params: dict[str, Any]) -> tuple[list[str], str]:
        key = (policy_name, tuple(sorted(params.items())))
        if key in cache:
            return cache[key]
        cfg = build_policy_config(
            base_cfg,
            policy_name=policy_name,
            policy_params=params,
            routing_thresholds=routing_map.get("thresholds", {}),
        )
        model = CanonicalRetrievalTranslator(config=cfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds = model.predict(val_src)
        out = (preds, cfg.model_id())
        cache[key] = out
        return out

    hybrid_variants = []
    for top_k in hybrid_top_k_values:
        params = {"top_k": int(top_k)}
        preds, model_id = get_policy_preds("hybrid", params)
        hybrid_variants.append((params, preds, model_id))

    fallback_variants = []
    for cap in fallback_oracc_cap_values:
        params = {"oracc_cap": int(cap), "gate": "low_conf"}
        preds, model_id = get_policy_preds("fallback", params)
        fallback_variants.append((params, preds, model_id))

    strong_variants = []
    for pool in strong_stage2_pool_values:
        for weight in strong_stage2_weight_values:
            params = {"stage2_pool": int(pool), "stage2_weight": float(weight)}
            preds, model_id = get_policy_preds("strong_rerank", params)
            strong_variants.append((params, preds, model_id))

    rows: list[dict[str, Any]] = []
    for h_params, h_preds, h_model_id in hybrid_variants:
        for f_params, f_preds, f_model_id in fallback_variants:
            for s_params, s_preds, s_model_id in strong_variants:
                routed_preds = []
                for i, route_name in enumerate(routes):
                    if route_name == "RETRIEVE_INTERNAL":
                        routed_preds.append(internal_preds[i])
                    elif route_name == "RETRIEVE_HYBRID":
                        routed_preds.append(h_preds[i])
                    elif route_name == "RETRIEVE_ORACC_FALLBACK":
                        routed_preds.append(f_preds[i])
                    elif route_name == "RERANK_STRONG":
                        routed_preds.append(s_preds[i])
                    else:
                        raise ValueError(f"Unsupported route: {route_name}")
                rows.append(
                    {
                        "val_chrf2": _score_chrf2(routed_preds, val_tgt),
                        "hybrid_params": h_params,
                        "fallback_params": f_params,
                        "strong_params": s_params,
                        "hybrid_model_id": h_model_id,
                        "fallback_model_id": f_model_id,
                        "strong_model_id": s_model_id,
                    }
                )

    rows_sorted = sorted(rows, key=lambda x: float(x["val_chrf2"]), reverse=True)
    best = rows_sorted[0]
    topk_rows = rows_sorted[: max(1, int(args.topk))]
    baseline_internal = _score_chrf2(internal_preds, val_tgt)

    report = {
        "memory_profile": args.memory,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "route_counts": route_counts,
        "baseline_internal_chrf2": baseline_internal,
        "n_combinations": len(rows),
        "search_space": {
            "hybrid_top_k": hybrid_top_k_values,
            "fallback_oracc_cap": fallback_oracc_cap_values,
            "strong_stage2_pool": strong_stage2_pool_values,
            "strong_stage2_weight": strong_stage2_weight_values,
        },
        "best": best,
        "top": topk_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tuned_payload = dict(payload)
    tuned_payload["routing_map"] = dict(routing_map)
    tuned_payload["routing_map"]["routes"] = dict(route_defs)
    tuned_payload["routing_map"]["routes"]["P1"] = {
        "route": "RETRIEVE_HYBRID",
        "policy_params": dict(best["hybrid_params"]),
    }
    tuned_payload["routing_map"]["routes"]["P2"] = {
        "route": "RETRIEVE_ORACC_FALLBACK",
        "policy_params": dict(best["fallback_params"]),
    }
    tuned_payload["routing_map"]["routes"]["P3"] = {
        "route": "RERANK_STRONG",
        "policy_params": dict(best["strong_params"]),
    }
    tuned_payload["policy_tuning"] = {
        "profile": args.memory,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "best_val_chrf2": float(best["val_chrf2"]),
        "baseline_internal_chrf2": float(baseline_internal),
        "selected_policy_params": {
            "P1": dict(best["hybrid_params"]),
            "P2": dict(best["fallback_params"]),
            "P3": dict(best["strong_params"]),
        },
        "report_json": str(report_path.resolve()),
    }
    output_map_path.parent.mkdir(parents=True, exist_ok=True)
    output_map_path.write_text(json.dumps(tuned_payload, indent=2), encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Best val chrF2: {float(best['val_chrf2']):.4f}")
    print(f"Selected P1 params: {best['hybrid_params']}")
    print(f"Selected P2 params: {best['fallback_params']}")
    print(f"Selected P3 params: {best['strong_params']}")
    print(f"Wrote tuned routing map: {output_map_path}")


if __name__ == "__main__":
    main()

