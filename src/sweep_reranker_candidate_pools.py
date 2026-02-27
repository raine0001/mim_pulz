from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.retrieval import (
    CANONICAL_BASELINE_V2,
    CanonicalRetrievalTranslator,
    RetrievalConfig,
    _route_policy_params,
    build_policy_config,
)
from mim_pulz.routed_reranker import FEATURE_NAMES, build_feature_matrix_for_candidates
from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, profile_source, resolve_routing_map_path
from structural_profile import build_token_frequency
from utils_manifest import write_json


BASE_SCORE_IDX = FEATURE_NAMES.index("base_score")


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
    metric = CHRF(word_order=2)
    s1 = float(metric.corpus_score(preds, [gold]).score)
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return {
        "chrf2": s1,
        "chrf2_reference": s2,
        "abs_diff": float(abs(s1 - s2)),
    }


def _parse_int_list(raw: str) -> list[int]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [int(v) for v in vals]


def _parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [float(v) for v in vals]


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


def _augment_features(raw_x: np.ndarray) -> np.ndarray:
    mat = np.asarray(raw_x, dtype=np.float32)
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-6
    z = (mat - mu) / sd
    top_idx = int(np.argmax(mat[:, BASE_SCORE_IDX]))
    rel = mat - mat[top_idx : top_idx + 1, :]
    return np.concatenate([mat, z, rel], axis=1)


def _build_pool(
    *,
    internal_rank: np.ndarray,
    external_rank: np.ndarray,
    global_rank: np.ndarray,
    internal_top_k: int,
    oracc_cap: int,
    policy_pool: int,
    required_indices: tuple[int, ...] = (),
    exclude_indices: tuple[int, ...] = (),
) -> np.ndarray:
    exclude = {int(x) for x in exclude_indices}
    out: list[int] = []
    seen: set[int] = set()

    for idx in internal_rank[: max(0, int(internal_top_k))]:
        i = int(idx)
        if i in exclude or i in seen:
            continue
        seen.add(i)
        out.append(i)
    for idx in external_rank[: max(0, int(oracc_cap))]:
        i = int(idx)
        if i in exclude or i in seen:
            continue
        seen.add(i)
        out.append(i)
    for idx in global_rank[: max(0, int(policy_pool))]:
        i = int(idx)
        if i in exclude or i in seen:
            continue
        seen.add(i)
        out.append(i)
    for idx in required_indices:
        i = int(idx)
        if i in exclude or i in seen:
            continue
        seen.add(i)
        out.append(i)

    if not out:
        # fallback to first available internal candidate
        for idx in internal_rank:
            i = int(idx)
            if i not in exclude:
                out.append(i)
                break
    if not out:
        for idx in global_rank:
            i = int(idx)
            if i not in exclude:
                out.append(i)
                break
    return np.asarray(out, dtype=np.int32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Targeted candidate-generation sweep for routed reranker headroom."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--routing-map", type=Path, default=PATHS.root / "artifacts" / "profiles" / "routing_map.json")
    parser.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)

    parser.add_argument("--internal-top-k-list", type=str, default="100,120,160")
    parser.add_argument("--oracc-cap-list", type=str, default="15,25,40")
    parser.add_argument("--policy-pool-list", type=str, default="0,20")

    parser.add_argument("--positive-band", type=float, default=0.5)
    parser.add_argument("--logistic-c", type=float, default=0.25)
    parser.add_argument("--gate-prob-margins", type=str, default="0.02,0.05,0.08,0.10,0.12")
    parser.add_argument("--gate-base-drops", type=str, default="0.02,0.03,0.05,0.08")

    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "reranker_candidate_pool_sweep_seed42.json",
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

    # Build routed baseline policy models once.
    route_to_params = {route: _route_policy_params(loaded_map.routing_map, route) for route in ROUTE_TO_POLICY}
    policy_configs = {
        policy_name: build_policy_config(
            base_cfg,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_thresholds,
        )
        for route_name, policy_name in ROUTE_TO_POLICY.items()
    }

    policy_preds_val: dict[str, list[str]] = {}
    policy_debug_val: dict[str, list[dict]] = {}
    for policy_name, pcfg in policy_configs.items():
        model = CanonicalRetrievalTranslator(config=pcfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, dbg = model.predict(val_src, return_debug=True)
        policy_preds_val[policy_name] = preds
        policy_debug_val[policy_name] = dbg

    internal_debug_val = policy_debug_val["internal_only"]
    token_freq = build_token_frequency(train_src)

    routed_val_preds: list[str] = []
    routed_choice_idx: list[int] = []
    routed_internal_idx: list[int] = []
    route_counts: dict[str, int] = {}
    for i, src in enumerate(val_src):
        features, labels = profile_source(src, token_freq=token_freq, profile_thresholds=loaded_map.profile_thresholds)
        dbg = internal_debug_val[i]
        internal_top_score = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
        raw_gap = dbg.get("internal_gap")
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=labels,
            internal_top_score=internal_top_score,
            internal_gap=internal_gap,
            routing_map=loaded_map.routing_map,
        )
        policy_name = str(decision["policy_name"])
        route_name = str(decision["route"])
        dbg_sel = policy_debug_val[policy_name][i]
        routed_choice_idx.append(int(dbg_sel.get("chosen_idx", 0)))
        routed_internal_idx.append(int(policy_debug_val["internal_only"][i].get("chosen_idx", 0)))
        routed_val_preds.append(policy_preds_val[policy_name][i])
        route_counts[route_name] = route_counts.get(route_name, 0) + 1

    baseline_score = _score(routed_val_preds, val_tgt)

    internal_list = sorted(set(_parse_int_list(args.internal_top_k_list)))
    oracc_list = sorted(set(_parse_int_list(args.oracc_cap_list)))
    policy_pool_list = sorted(set(_parse_int_list(args.policy_pool_list)))
    gate_prob_list = sorted(set(_parse_float_list(args.gate_prob_margins)))
    gate_base_list = sorted(set(_parse_float_list(args.gate_base_drops)))

    max_internal = max(internal_list) if internal_list else 120
    max_oracc = max(oracc_list) if oracc_list else 25
    max_policy_pool = max(policy_pool_list) if policy_pool_list else 0

    candidate_cfg = build_policy_config(
        base_cfg,
        policy_name="hybrid",
        policy_params={"top_k": max_internal, "oracc_cap": max_oracc},
        routing_thresholds=routing_thresholds,
    )
    candidate_model = CanonicalRetrievalTranslator(config=candidate_cfg)
    candidate_model.fit(train_src=train_src, train_tgt=train_tgt)

    q_norm_train = [candidate_model._norm(x) for x in train_src]
    q_mat_train = candidate_model.vectorizer.transform(q_norm_train)
    sims_train = (q_mat_train @ candidate_model.train_matrix.T).toarray().astype(np.float32)

    q_norm_val = [candidate_model._norm(x) for x in val_src]
    q_mat_val = candidate_model.vectorizer.transform(q_norm_val)
    sims_val = (q_mat_val @ candidate_model.train_matrix.T).toarray().astype(np.float32)

    # Precompute ranked lists + sentence-chrf lookup for max union per row.
    train_cache: list[dict[str, Any]] = []
    for i, row in enumerate(sims_train):
        internal_rank = candidate_model._candidate_indices_from_subset(row, candidate_model.internal_indices, min(max_internal + 8, len(candidate_model.internal_indices)))
        external_rank = candidate_model._candidate_indices_from_subset(row, candidate_model.external_indices, min(max_oracc + 8, len(candidate_model.external_indices))) if candidate_model.external_indices.size > 0 else np.asarray([], dtype=np.int32)
        global_rank = candidate_model._candidate_indices(row, min(max_policy_pool + 20, len(row))) if max_policy_pool > 0 else np.asarray([], dtype=np.int32)
        max_pool = _build_pool(
            internal_rank=internal_rank,
            external_rank=external_rank,
            global_rank=global_rank,
            internal_top_k=max_internal,
            oracc_cap=max_oracc,
            policy_pool=max_policy_pool,
            required_indices=(),
            exclude_indices=(i,),
        )
        sent_lookup: dict[int, float] = {}
        gold = train_tgt[i]
        for idx in max_pool.tolist():
            idx_i = int(idx)
            sent_lookup[idx_i] = float(sacrebleu.sentence_chrf(candidate_model.train_tgt[idx_i], [gold], word_order=2).score)
        train_cache.append(
            {
                "row": row,
                "internal_rank": internal_rank,
                "external_rank": external_rank,
                "global_rank": global_rank,
                "sent_lookup": sent_lookup,
            }
        )

    val_cache: list[dict[str, Any]] = []
    for i, row in enumerate(sims_val):
        internal_rank = candidate_model._candidate_indices_from_subset(row, candidate_model.internal_indices, min(max_internal + 8, len(candidate_model.internal_indices)))
        external_rank = candidate_model._candidate_indices_from_subset(row, candidate_model.external_indices, min(max_oracc + 8, len(candidate_model.external_indices))) if candidate_model.external_indices.size > 0 else np.asarray([], dtype=np.int32)
        global_rank = candidate_model._candidate_indices(row, min(max_policy_pool + 20, len(row))) if max_policy_pool > 0 else np.asarray([], dtype=np.int32)
        max_pool = _build_pool(
            internal_rank=internal_rank,
            external_rank=external_rank,
            global_rank=global_rank,
            internal_top_k=max_internal,
            oracc_cap=max_oracc,
            policy_pool=max_policy_pool,
            required_indices=(routed_internal_idx[i], routed_choice_idx[i]),
            exclude_indices=(),
        )
        sent_lookup: dict[int, float] = {}
        gold = val_tgt[i]
        for idx in max_pool.tolist():
            idx_i = int(idx)
            sent_lookup[idx_i] = float(sacrebleu.sentence_chrf(candidate_model.train_tgt[idx_i], [gold], word_order=2).score)
        val_cache.append(
            {
                "row": row,
                "internal_rank": internal_rank,
                "external_rank": external_rank,
                "global_rank": global_rank,
                "sent_lookup": sent_lookup,
                "routed_idx": routed_choice_idx[i],
                "required": (routed_internal_idx[i], routed_choice_idx[i]),
            }
        )

    results: list[dict[str, Any]] = []
    for internal_top_k, oracc_cap, policy_pool in product(internal_list, oracc_list, policy_pool_list):
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []

        # Build train matrices for this combo.
        for i, cache in enumerate(train_cache):
            pool = _build_pool(
                internal_rank=cache["internal_rank"],
                external_rank=cache["external_rank"],
                global_rank=cache["global_rank"],
                internal_top_k=internal_top_k,
                oracc_cap=oracc_cap,
                policy_pool=policy_pool,
                required_indices=(),
                exclude_indices=(i,),
            )
            if pool.size < 2:
                continue
            raw_x = build_feature_matrix_for_candidates(
                model=candidate_model,
                query_text_raw=train_src[i],
                row=cache["row"],
                candidate_idx=pool,
            )
            if raw_x.shape[0] < 2:
                continue
            aug_x = _augment_features(raw_x)

            sent_scores = np.asarray([float(cache["sent_lookup"].get(int(idx), 0.0)) for idx in pool], dtype=np.float32)
            best = float(sent_scores.max())
            labels = (sent_scores >= (best - float(args.positive_band))).astype(np.int32)
            if int(labels.sum()) == 0:
                labels[int(np.argmax(sent_scores))] = 1
            if int(labels.sum()) == len(labels):
                labels[int(np.argmin(sent_scores))] = 0

            x_rows.append(aug_x)
            y_rows.append(labels)

        if not x_rows:
            continue

        x_train = np.vstack(x_rows).astype(np.float32)
        y_train = np.concatenate(y_rows).astype(np.int32)

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
                C=float(args.logistic_c),
                random_state=int(args.seed),
            ),
        )
        clf.fit(x_train, y_train)

        # Evaluate val predictions and keep per-row scores for gate sweep.
        raw_preds: list[str] = []
        oracle_preds: list[str] = []
        per_row_eval: list[dict[str, Any]] = []

        for i, cache in enumerate(val_cache):
            pool = _build_pool(
                internal_rank=cache["internal_rank"],
                external_rank=cache["external_rank"],
                global_rank=cache["global_rank"],
                internal_top_k=internal_top_k,
                oracc_cap=oracc_cap,
                policy_pool=policy_pool,
                required_indices=cache["required"],
                exclude_indices=(),
            )
            raw_x = build_feature_matrix_for_candidates(
                model=candidate_model,
                query_text_raw=val_src[i],
                row=cache["row"],
                candidate_idx=pool,
            )
            aug_x = _augment_features(raw_x)
            proba = clf.predict_proba(aug_x)[:, 1]

            best_pos = int(np.argmax(proba))
            raw_preds.append(candidate_model.train_tgt[int(pool[best_pos])])

            # routed baseline candidate position in pool
            routed_idx = int(cache["routed_idx"])
            routed_pos_arr = np.where(pool == routed_idx)[0]
            routed_pos = int(routed_pos_arr[0]) if routed_pos_arr.size > 0 else 0

            # oracle in this pool
            sent_scores = [float(cache["sent_lookup"].get(int(idx), 0.0)) for idx in pool]
            oracle_pos = int(np.argmax(np.asarray(sent_scores, dtype=np.float32)))
            oracle_preds.append(candidate_model.train_tgt[int(pool[oracle_pos])])

            per_row_eval.append(
                {
                    "pool": pool,
                    "raw_x": raw_x,
                    "proba": proba,
                    "routed_pos": routed_pos,
                }
            )

        raw_score = _score(raw_preds, val_tgt)
        oracle_score = _score(oracle_preds, val_tgt)

        # Gate sweep.
        best_gate_score = raw_score
        best_gate = {"prob_margin": 0.0, "base_score_drop": -1.0, "changed_count": int(sum(1 for row in per_row_eval if int(np.argmax(row["proba"])) != row["routed_pos"]))}

        for pm in gate_prob_list:
            for bd in gate_base_list:
                gated_preds: list[str] = []
                changed = 0
                for row_eval in per_row_eval:
                    pool = row_eval["pool"]
                    raw_x = row_eval["raw_x"]
                    proba = row_eval["proba"]
                    routed_pos = int(row_eval["routed_pos"])
                    best_pos = int(np.argmax(proba))

                    prob_gap = float(proba[best_pos] - proba[routed_pos])
                    best_base = float(raw_x[best_pos, BASE_SCORE_IDX])
                    routed_base = float(raw_x[routed_pos, BASE_SCORE_IDX])
                    if not (prob_gap >= float(pm) and best_base >= (routed_base - float(bd))):
                        best_pos = routed_pos
                    if best_pos != routed_pos:
                        changed += 1
                    gated_preds.append(candidate_model.train_tgt[int(pool[best_pos])])

                gated_score = _score(gated_preds, val_tgt)
                if gated_score["chrf2"] > best_gate_score["chrf2"]:
                    best_gate_score = gated_score
                    best_gate = {
                        "prob_margin": float(pm),
                        "base_score_drop": float(bd),
                        "changed_count": int(changed),
                    }

        results.append(
            {
                "internal_top_k": int(internal_top_k),
                "oracc_cap": int(oracc_cap),
                "policy_pool": int(policy_pool),
                "train_examples": int(len(y_train)),
                "positive_rate": float(float(y_train.mean())),
                "raw_reranked": raw_score,
                "gated_reranked": best_gate_score,
                "oracle_pool": oracle_score,
                "delta_vs_baseline_raw": float(raw_score["chrf2"] - baseline_score["chrf2"]),
                "delta_vs_baseline_gated": float(best_gate_score["chrf2"] - baseline_score["chrf2"]),
                "selected_gate": {
                    **best_gate,
                    "changed_rate": float(best_gate["changed_count"] / max(1, len(val_cache))),
                },
            }
        )

    results_sorted = sorted(results, key=lambda r: r["gated_reranked"]["chrf2"], reverse=True)
    best = results_sorted[0] if results_sorted else None

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "memory_profile": str(args.memory_profile),
        "routing_map_path": str(loaded_map.path.resolve()),
        "routing_map_sha256": loaded_map.sha256,
        "route_counts_baseline": route_counts,
        "baseline_routed": baseline_score,
        "base_config": asdict(base_cfg),
        "candidate_config": asdict(candidate_cfg),
        "search_space": {
            "internal_top_k_list": internal_list,
            "oracc_cap_list": oracc_list,
            "policy_pool_list": policy_pool_list,
            "positive_band": float(args.positive_band),
            "logistic_c": float(args.logistic_c),
            "gate_prob_margins": gate_prob_list,
            "gate_base_drops": gate_base_list,
        },
        "best": best,
        "results": results_sorted,
    }

    out_path = args.json_out if args.json_out.is_absolute() else (repo_root / args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, summary)

    print(f"Baseline routed chrF2\t{baseline_score['chrf2']:.4f}")
    if best is not None:
        print(f"Best gated reranked chrF2\t{best['gated_reranked']['chrf2']:.4f}")
        print(f"Delta\t{best['delta_vs_baseline_gated']:.4f}")
        print(f"Best combo\tinternal_top_k={best['internal_top_k']}, oracc_cap={best['oracc_cap']}, policy_pool={best['policy_pool']}")
        print(f"Gate\t{best['selected_gate']}")
        print(f"Oracle@pool\t{best['oracle_pool']['chrf2']:.4f}")
    print(f"Combos\t{len(results_sorted)}")
    print(f"Wrote\t{out_path}")


if __name__ == "__main__":
    main()

