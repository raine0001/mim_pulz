from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import sacrebleu
from sklearn.metrics.pairwise import cosine_similarity


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.eval_metrics import corpus_metrics
from mim_pulz.retrieval import (
    CANONICAL_BASELINE_V2,
    CanonicalRetrievalTranslator,
    RetrievalConfig,
    _route_policy_params,
    build_policy_config,
)
from mim_pulz.routed_reranker import (
    FEATURE_NAMES,
    build_candidate_pool_top_internal_oracc,
    build_feature_matrix_for_candidates,
    feature_dict_from_row,
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


def _candidate_row_payload(
    *,
    model: CanonicalRetrievalTranslator,
    query: str,
    gold: str,
    row: np.ndarray,
    candidate_idx: np.ndarray,
    feature_mat: np.ndarray,
    chosen_idx: int | None,
) -> list[dict]:
    out: list[dict] = []
    baseline_pos = None
    if chosen_idx is not None:
        for pos, idx in enumerate(candidate_idx):
            if int(idx) == int(chosen_idx):
                baseline_pos = int(pos)
                break
    if baseline_pos is None:
        base_score_idx = FEATURE_NAMES.index("base_score")
        baseline_pos = int(np.argmax(feature_mat[:, base_score_idx])) if len(candidate_idx) > 0 else 0

    baseline_bleu = 0.0
    baseline_chrf = 0.0
    baseline_combined = 0.0
    if len(candidate_idx) > 0:
        baseline_target = model.train_tgt[int(candidate_idx[baseline_pos])]
        baseline_bleu = float(sacrebleu.sentence_bleu(baseline_target, [gold]).score)
        baseline_chrf = float(sacrebleu.sentence_chrf(baseline_target, [gold], word_order=2).score)
        baseline_combined = float(math.sqrt(max(0.0, baseline_bleu) * max(0.0, baseline_chrf)))

    for pos, idx in enumerate(candidate_idx):
        idx_i = int(idx)
        target = model.train_tgt[idx_i]
        sent_bleu = float(sacrebleu.sentence_bleu(target, [gold]).score)
        sent_chrf = float(sacrebleu.sentence_chrf(target, [gold], word_order=2).score)
        sent_combined = float(math.sqrt(max(0.0, sent_bleu) * max(0.0, sent_chrf)))
        out.append(
            {
                "rank_in_pool": int(pos + 1),
                "memory_idx": idx_i,
                "origin": str(model.train_origins[idx_i]),
                "context": str(model.train_domains[idx_i]),
                "score_cosine": float(row[idx_i]),
                "target_text": target,
                "source_text": model.train_src_raw[idx_i],
                "sentence_bleu": sent_bleu,
                "sentence_chrf2": sent_chrf,
                "sentence_combined": sent_combined,
                "delta_bleu_vs_baseline": float(sent_bleu - baseline_bleu),
                "delta_chrf2_vs_baseline": float(sent_chrf - baseline_chrf),
                "delta_combined_vs_baseline": float(sent_combined - baseline_combined),
                "is_current_choice": bool(chosen_idx is not None and idx_i == int(chosen_idx)),
                "features": feature_dict_from_row(feature_mat[pos]),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export routed candidate sets (top internal + capped ORACC) for reranker training/eval."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    parser.add_argument("--routing-map", type=Path, default=None)
    parser.add_argument("--internal-top-k", type=int, default=120)
    parser.add_argument("--oracc-cap", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "routed_candidates_seed42",
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
    routing_thresholds = dict(loaded_map.routing_map.get("thresholds", {}))

    base_cfg = _profile_config(args.memory_profile, repo_root)

    # Policy models used by routed baseline selection.
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

    policy_models: dict[str, CanonicalRetrievalTranslator] = {}
    policy_preds_val: dict[str, list[str]] = {}
    policy_debug_val: dict[str, list[dict]] = {}
    for policy_name, cfg in policy_configs.items():
        model = CanonicalRetrievalTranslator(config=cfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, dbg = model.predict(val_src, return_debug=True)
        policy_models[policy_name] = model
        policy_preds_val[policy_name] = preds
        policy_debug_val[policy_name] = dbg

    internal_debug_val = policy_debug_val["internal_only"]
    token_freq = build_token_frequency(train_src)

    routed_val_preds: list[str] = []
    val_rows: list[dict] = []

    # Candidate model for export pool: top internal + capped ORACC.
    candidate_cfg = build_policy_config(
        base_cfg,
        policy_name="hybrid",
        policy_params={
            "top_k": int(max(1, args.internal_top_k)),
            "oracc_cap": int(max(0, args.oracc_cap)),
        },
        routing_thresholds=routing_thresholds,
    )
    candidate_model = CanonicalRetrievalTranslator(config=candidate_cfg)
    candidate_model.fit(train_src=train_src, train_tgt=train_tgt)

    q_norm_val = [candidate_model._norm(x) for x in val_src]
    q_mat_val = candidate_model.vectorizer.transform(q_norm_val)
    sims_val = cosine_similarity(q_mat_val, candidate_model.train_matrix)

    for i, query in enumerate(val_src):
        feat_labels_features, feat_labels = profile_source(
            query,
            token_freq=token_freq,
            profile_thresholds=loaded_map.profile_thresholds,
        )
        internal_dbg = internal_debug_val[i]
        internal_top = float(internal_dbg.get("internal_top_cosine", internal_dbg.get("global_score", 0.0)))
        raw_gap = internal_dbg.get("internal_gap")
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=feat_labels,
            internal_top_score=internal_top,
            internal_gap=internal_gap,
            routing_map=loaded_map.routing_map,
        )

        chosen_policy = str(decision["policy_name"])
        routed_dbg = policy_debug_val[chosen_policy][i]
        chosen_idx = int(routed_dbg["chosen_idx"])
        routed_pred = policy_preds_val[chosen_policy][i]
        routed_val_preds.append(routed_pred)

        row = sims_val[i]
        pool = build_candidate_pool_top_internal_oracc(
            model=candidate_model,
            row=row,
            internal_top_k=int(max(1, args.internal_top_k)),
            oracc_cap=int(max(0, args.oracc_cap)),
            required_indices=(
                int(chosen_idx),
                int(internal_dbg.get("chosen_idx", chosen_idx)),
            ),
            exclude_indices=(),
        )
        feature_mat = build_feature_matrix_for_candidates(
            model=candidate_model,
            query_text_raw=query,
            row=row,
            candidate_idx=pool,
        )
        candidates = _candidate_row_payload(
            model=candidate_model,
            query=query,
            gold=val_tgt[i],
            row=row,
            candidate_idx=pool,
            feature_mat=feature_mat,
            chosen_idx=chosen_idx,
        )
        baseline_candidates = [c for c in candidates if bool(c.get("is_current_choice", False))]
        if baseline_candidates:
            baseline_cand = baseline_candidates[0]
        else:
            baseline_cand = max(candidates, key=lambda c: float(dict(c.get("features", {})).get("base_score", 0.0)))

        val_rows.append(
            {
                "split": "val",
                "row_index": int(i),
                "query": query,
                "gold": val_tgt[i],
                "labels": feat_labels,
                "features": feat_labels_features,
                "current_route": str(decision["route"]),
                "current_policy_name": chosen_policy,
                "current_policy_params": dict(decision.get("policy_params", {})),
                "current_rationale": str(decision.get("rationale", "")),
                "current_chosen_idx": int(chosen_idx),
                "current_prediction": routed_pred,
                "current_baseline_memory_idx": int(baseline_cand.get("memory_idx", chosen_idx)),
                "current_prediction_sentence_bleu": float(baseline_cand.get("sentence_bleu", 0.0)),
                "current_prediction_sentence_chrf2": float(
                    baseline_cand.get("sentence_chrf2", 0.0)
                ),
                "current_prediction_sentence_combined": float(baseline_cand.get("sentence_combined", 0.0)),
                "candidates": candidates,
            }
        )

    routed_score_val = _score(routed_val_preds, val_tgt)

    # Train-side candidate rows for reranker fitting (self candidate excluded).
    train_rows: list[dict] = []
    q_norm_train = [candidate_model._norm(x) for x in train_src]
    q_mat_train = candidate_model.vectorizer.transform(q_norm_train)
    sims_train = cosine_similarity(q_mat_train, candidate_model.train_matrix)
    for i, query in enumerate(train_src):
        row = sims_train[i]
        pool = build_candidate_pool_top_internal_oracc(
            model=candidate_model,
            row=row,
            internal_top_k=int(max(1, args.internal_top_k)),
            oracc_cap=int(max(0, args.oracc_cap)),
            required_indices=(),
            exclude_indices=(int(i),),
        )
        feature_mat = build_feature_matrix_for_candidates(
            model=candidate_model,
            query_text_raw=query,
            row=row,
            candidate_idx=pool,
        )
        candidates = _candidate_row_payload(
            model=candidate_model,
            query=query,
            gold=train_tgt[i],
            row=row,
            candidate_idx=pool,
            feature_mat=feature_mat,
            chosen_idx=None,
        )
        baseline_cand = max(candidates, key=lambda c: float(dict(c.get("features", {})).get("base_score", 0.0)))
        train_rows.append(
            {
                "split": "train",
                "row_index": int(i),
                "query": query,
                "gold": train_tgt[i],
                "current_baseline_memory_idx": int(baseline_cand.get("memory_idx", -1)),
                "current_prediction": str(baseline_cand.get("target_text", "")),
                "current_prediction_sentence_bleu": float(baseline_cand.get("sentence_bleu", 0.0)),
                "current_prediction_sentence_chrf2": float(baseline_cand.get("sentence_chrf2", 0.0)),
                "current_prediction_sentence_combined": float(baseline_cand.get("sentence_combined", 0.0)),
                "candidates": candidates,
            }
        )

    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / "val_candidates.jsonl"
    train_path = output_dir / "train_candidates.jsonl"

    with val_path.open("w", encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with train_path.open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "memory_profile": str(args.memory_profile),
        "routing_map_path": str(loaded_map.path.resolve()),
        "routing_map_sha256": loaded_map.sha256,
        "routing_thresholds": routing_thresholds,
        "base_config": asdict(base_cfg),
        "candidate_pool_config": {
            "internal_top_k": int(args.internal_top_k),
            "oracc_cap": int(args.oracc_cap),
            "candidate_config": asdict(candidate_cfg),
        },
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "routed_val_score": routed_score_val,
        "train_jsonl": str(train_path.resolve()),
        "val_jsonl": str(val_path.resolve()),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(f"Train candidate rows\t{len(train_rows)}")
    print(f"Val candidate rows\t{len(val_rows)}")
    print(f"Routed baseline chrF2\t{routed_score_val['chrf2']:.4f}")
    print(f"Train JSONL\t{train_path}")
    print(f"Val JSONL\t{val_path}")
    print(f"Manifest\t{manifest_path}")


if __name__ == "__main__":
    main()

