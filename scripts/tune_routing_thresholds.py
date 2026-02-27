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
        idx[val_size:],
        idx[:val_size],
    )


def _score_chrf2(preds: list[str], gold: list[str]) -> float:
    return float(CHRF(word_order=2).corpus_score(preds, [gold]).score)


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
    p = argparse.ArgumentParser(description="Tune structural routing thresholds for retrieval_routed.")
    p.add_argument("--routing-map", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_map.json")
    p.add_argument("--output-map", type=Path, default=None, help="Where to write tuned map (default: overwrite --routing-map).")
    p.add_argument("--report-json", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_threshold_tuning_oracc_best_seed42.json")
    p.add_argument("--backup-existing-map", action="store_true", help="Backup target map to *.pre_tuned.json if it exists.")
    p.add_argument("--memory", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    p.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    p.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--low-min", type=float, default=0.20)
    p.add_argument("--low-max", type=float, default=0.55)
    p.add_argument("--low-steps", type=int, default=15)
    p.add_argument("--high-min", type=float, default=0.45)
    p.add_argument("--high-max", type=float, default=0.90)
    p.add_argument("--high-steps", type=int, default=19)
    p.add_argument("--min-test-non-internal", type=int, default=1)
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

    data = load_deep_past_competition(_resolve(args.competition_dir, repo_root), _resolve(args.schema, repo_root))
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col
    test_src_col = data.schema.test_text_col

    train_all = data.train.copy()
    train_all[src_col] = train_all[src_col].astype(str).fillna("")
    train_all[tgt_col] = train_all[tgt_col].astype(str).fillna("")
    test_df = data.test.copy()
    test_df[test_src_col] = test_df[test_src_col].astype(str).fillna("")

    train_df, val_df, _, _ = _split(train_all, seed=int(args.seed), val_frac=float(args.val_frac))
    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()
    test_src = test_df[test_src_col].tolist()

    base_cfg = _profile_config(args.memory, repo_root)
    route_to_params = {
        str(route_def["route"]): dict(route_def.get("policy_params", {}))
        for route_def in route_defs.values()
    }
    policy_cfgs = {
        policy_name: build_policy_config(
            base_cfg,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_map.get("thresholds", {}),
        )
        for route_name, policy_name in ROUTE_TO_POLICY.items()
    }

    policy_preds: dict[str, list[str]] = {}
    policy_debug: dict[str, list[dict]] = {}
    for policy_name, cfg in policy_cfgs.items():
        model = CanonicalRetrievalTranslator(config=cfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, debug_rows = model.predict(val_src, return_debug=True)
        policy_preds[policy_name] = preds
        policy_debug[policy_name] = debug_rows

    internal_debug = policy_debug["internal_only"]
    internal_top = np.asarray(
        [float(d.get("internal_top_cosine", d.get("global_score", 0.0))) for d in internal_debug],
        dtype=np.float32,
    )
    internal_gap = [d.get("internal_gap", None) for d in internal_debug]

    tok_freq = build_token_frequency(train_src)
    val_labels = []
    for src in val_src:
        features = extract_source_features(src, token_freq=tok_freq)
        val_labels.append(label_features(features, profile_thr))

    full_internal = CanonicalRetrievalTranslator(config=policy_cfgs["internal_only"])
    full_internal.fit(train_src=train_all[src_col].tolist(), train_tgt=train_all[tgt_col].tolist())
    _, full_test_debug = full_internal.predict(test_src, return_debug=True)
    full_tok_freq = build_token_frequency(train_all[src_col].tolist())
    test_labels = [
        label_features(extract_source_features(src, token_freq=full_tok_freq), profile_thr)
        for src in test_src
    ]

    def eval_thresholds(t_low: float, t_high: float) -> tuple[float, dict[str, int], dict[str, int], int]:
        rmap = {
            "thresholds": {"internal_top_low": float(t_low), "internal_top_high": float(t_high)},
            "routes": route_defs,
        }
        val_route_counts: dict[str, int] = {}
        routed_preds: list[str] = []
        for i, labels in enumerate(val_labels):
            gap = internal_gap[i]
            dec = route_decision(
                labels,
                internal_top_score=float(internal_top[i]),
                internal_gap=None if gap is None else float(gap),
                routing_map=rmap,
            )
            route_name = str(dec["route"])
            val_route_counts[route_name] = val_route_counts.get(route_name, 0) + 1
            routed_preds.append(policy_preds[ROUTE_TO_POLICY[route_name]][i])
        val_score = _score_chrf2(routed_preds, val_tgt)

        test_route_counts: dict[str, int] = {}
        for i, labels in enumerate(test_labels):
            dbg = full_test_debug[i]
            top = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
            gap = dbg.get("internal_gap", None)
            dec = route_decision(
                labels,
                internal_top_score=top,
                internal_gap=None if gap is None else float(gap),
                routing_map=rmap,
            )
            route_name = str(dec["route"])
            test_route_counts[route_name] = test_route_counts.get(route_name, 0) + 1
        non_internal = sum(v for k, v in test_route_counts.items() if k != "RETRIEVE_INTERNAL")
        return val_score, val_route_counts, test_route_counts, int(non_internal)

    low_values = np.unique(np.round(np.linspace(args.low_min, args.low_max, args.low_steps), 6))
    high_values = np.unique(np.round(np.linspace(args.high_min, args.high_max, args.high_steps), 6))
    rows: list[dict[str, Any]] = []
    for t_low in low_values:
        for t_high in high_values:
            if float(t_low) >= float(t_high):
                continue
            val_score, val_counts, test_counts, test_non_internal = eval_thresholds(float(t_low), float(t_high))
            rows.append(
                {
                    "t_low": float(t_low),
                    "t_high": float(t_high),
                    "val_chrf2": float(val_score),
                    "val_counts": val_counts,
                    "test_counts": test_counts,
                    "test_non_internal": int(test_non_internal),
                }
            )

    best_any = max(rows, key=lambda x: float(x["val_chrf2"]))
    rows_with_trigger = [r for r in rows if int(r["test_non_internal"]) >= int(args.min_test_non_internal)]
    if rows_with_trigger:
        best_with_trigger = max(rows_with_trigger, key=lambda x: float(x["val_chrf2"]))
        selected = best_with_trigger
        selection_reason = f"best_with_test_non_internal>={int(args.min_test_non_internal)}"
    else:
        best_with_trigger = None
        selected = best_any
        selection_reason = "fallback_best_any_no_trigger_candidate"

    report = {
        "memory_profile": args.memory,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "search_space": {
            "low_values": [float(x) for x in low_values.tolist()],
            "high_values": [float(x) for x in high_values.tolist()],
            "n_candidates": len(rows),
            "min_test_non_internal": int(args.min_test_non_internal),
        },
        "best_any": best_any,
        "best_with_test_trigger": best_with_trigger,
        "selected": selected,
        "selection_reason": selection_reason,
        "top10_by_val_with_trigger": sorted(
            rows_with_trigger,
            key=lambda x: float(x["val_chrf2"]),
            reverse=True,
        )[:10],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.backup_existing_map and output_map_path.exists():
        backup = output_map_path.with_name(output_map_path.stem + ".pre_tuned.json")
        if not backup.exists():
            shutil.copyfile(output_map_path, backup)

    tuned_payload = dict(payload)
    tuned_payload["routing_map"] = dict(routing_map)
    tuned_payload["routing_map"]["thresholds"] = {
        "internal_top_low": float(selected["t_low"]),
        "internal_top_high": float(selected["t_high"]),
    }
    tuned_payload["tuning"] = {
        "profile": args.memory,
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "objective": "maximize val chrF2 with deterministic routed policy",
        "selected": selection_reason,
        "selected_thresholds": {
            "internal_top_low": float(selected["t_low"]),
            "internal_top_high": float(selected["t_high"]),
        },
        "validation_chrF2": float(selected["val_chrf2"]),
        "test_route_counts": dict(selected["test_counts"]),
        "report_json": str(report_path.resolve()),
    }
    output_map_path.parent.mkdir(parents=True, exist_ok=True)
    output_map_path.write_text(json.dumps(tuned_payload, indent=2), encoding="utf-8")

    print(f"Report: {report_path}")
    print(
        "Selected thresholds:",
        {
            "internal_top_low": float(selected["t_low"]),
            "internal_top_high": float(selected["t_high"]),
        },
    )
    print(f"Validation chrF2: {float(selected['val_chrf2']):.4f}")
    print(f"Test route counts: {dict(selected['test_counts'])}")
    print(f"Wrote tuned map: {output_map_path}")


if __name__ == "__main__":
    main()

