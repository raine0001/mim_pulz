from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import CHRF
from sklearn.metrics.pairwise import cosine_similarity


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator, RetrievalConfig
from structural_profile import (
    ProfileThresholds,
    build_token_frequency,
    default_routing_map,
    extract_source_features,
    extract_target_features,
    label_features,
    route_decision,
    top_flags,
)


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
        train_idx,
        val_idx,
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


def _subset_chrf(preds: list[str], gold: list[str], idxs: list[int]) -> float:
    if not idxs:
        return float("nan")
    metric = CHRF(word_order=2)
    p = [preds[i] for i in idxs]
    g = [gold[i] for i in idxs]
    return float(metric.corpus_score(p, [g]).score)


def _oracc_best_profile(repo_root: Path) -> RetrievalConfig:
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


def _predict_external_only(model: CanonicalRetrievalTranslator, texts: list[str], top_k: int = 50) -> list[str]:
    if model.train_matrix is None:
        raise RuntimeError("Model not fit.")
    q_norm = [model._norm(x) for x in texts]
    q_mat = model.vectorizer.transform(q_norm)
    sims = cosine_similarity(q_mat, model.train_matrix)
    preds = []
    for i, q_raw in enumerate(texts):
        row = sims[i]
        cands = model._candidate_indices_from_subset(row, model.external_indices, k=min(top_k, len(row)))
        if cands.size == 0:
            cands = model._candidate_indices(row, min(top_k, len(row)))
        q = model._norm(q_raw)
        bm25_row = (
            model.stage2_bm25.score_one(q)
            if (model.config.stage2_type == "bm25" and model.stage2_bm25 is not None)
            else None
        )
        scores, _, _ = model._rerank_components(
            row=row,
            query=q,
            q_len=len(q),
            candidates=cands,
            bm25_row=bm25_row,
        )
        best_idx = int(cands[int(np.argmax(scores))])
        preds.append(model.train_tgt[best_idx])
    return preds


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structural profiling pass: labels + routing + bucketed policy reports."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=PATHS.root / "artifacts" / "profiles")
    parser.add_argument("--sample-per-bucket", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir
    reports_dir = out_dir / "reports"
    samples_dir = out_dir / "profile_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    data = load_deep_past_competition(args.competition_dir, args.schema)
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col
    test_id_col = data.schema.test_id_col
    test_src_col = data.schema.test_text_col

    train_all = data.train.copy()
    train_all[src_col] = train_all[src_col].astype(str).fillna("")
    train_all[tgt_col] = train_all[tgt_col].astype(str).fillna("")
    test_df = data.test.copy()
    test_df[test_src_col] = test_df[test_src_col].astype(str).fillna("")

    train_df, val_df, train_idx, val_idx = _split(train_all, seed=args.seed, val_frac=args.val_frac)
    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()
    test_src = test_df[test_src_col].tolist()

    # Structural features and thresholds.
    tok_freq = build_token_frequency(train_src)
    train_feats = [extract_source_features(x, token_freq=tok_freq) for x in train_src]
    thr = ProfileThresholds.from_feature_rows(train_feats)

    # Policy models.
    internal_cfg = CANONICAL_BASELINE_V2
    internal_model = CanonicalRetrievalTranslator(config=internal_cfg)
    internal_model.fit(train_src=train_src, train_tgt=train_tgt)
    internal_preds, internal_debug = internal_model.predict(val_src, return_debug=True)

    strong_cfg_dict = asdict(CANONICAL_BASELINE_V2)
    strong_cfg_dict["stage2_pool"] = 120
    strong_cfg_dict["stage2_weight"] = 0.35
    strong_cfg = RetrievalConfig(**strong_cfg_dict)
    strong_model = CanonicalRetrievalTranslator(config=strong_cfg)
    strong_model.fit(train_src=train_src, train_tgt=train_tgt)
    strong_preds = strong_model.predict(val_src)

    oracc_available = (_oracc_best_profile(PATHS.root).external_memory_paths[0] and Path(_oracc_best_profile(PATHS.root).external_memory_paths[0]).exists())
    if oracc_available:
        hybrid_cfg = _oracc_best_profile(PATHS.root)
        hybrid_model = CanonicalRetrievalTranslator(config=hybrid_cfg)
        hybrid_model.fit(train_src=train_src, train_tgt=train_tgt)
        hybrid_preds = hybrid_model.predict(val_src)

        fallback_cfg_dict = asdict(hybrid_cfg)
        fallback_cfg_dict.update(
            {
                "external_enable_fallback": True,
                "external_internal_top_threshold": 0.14,
                "external_internal_gap_threshold": -1.0,
                "external_candidate_cap": 10,
                "external_gate_bonus": 0.0,
                "external_gate_margin": 0.0,
            }
        )
        fallback_cfg = RetrievalConfig(**fallback_cfg_dict)
        fallback_model = CanonicalRetrievalTranslator(config=fallback_cfg)
        fallback_model.fit(train_src=train_src, train_tgt=train_tgt)
        fallback_preds = fallback_model.predict(val_src)
        oracc_only_preds = _predict_external_only(hybrid_model, val_src, top_k=50)
    else:
        hybrid_preds = internal_preds
        fallback_preds = internal_preds
        oracc_only_preds = internal_preds

    policy_preds = {
        "internal_only": internal_preds,
        "oracc_only": oracc_only_preds,
        "hybrid": hybrid_preds,
        "fallback": fallback_preds,
        "strong_rerank": strong_preds,
    }

    policy_scores = {k: _score(v, val_tgt) for k, v in policy_preds.items()}

    internal_top_scores = [float(d.get("internal_top_cosine", d.get("global_score", 0.0))) for d in internal_debug]
    internal_gaps = [d.get("internal_gap", None) for d in internal_debug]
    score_stats = {
        "p25": float(np.percentile(np.asarray(internal_top_scores, dtype=np.float32), 25)),
        "p75": float(np.percentile(np.asarray(internal_top_scores, dtype=np.float32), 75)),
        "mean": float(np.mean(np.asarray(internal_top_scores, dtype=np.float32))),
    }
    routing_map = default_routing_map(score_stats)

    # Build profile rows.
    profile_rows: list[dict] = []

    # Train split rows (no score-aware routing).
    for i, row in train_df.iterrows():
        src = str(row[src_col])
        tgt = str(row[tgt_col])
        feats = extract_source_features(src, token_freq=tok_freq)
        feats.update(extract_target_features(tgt))
        labels = label_features(feats, thr)
        decision = route_decision(labels, internal_top_score=None, internal_gap=None, routing_map=routing_map)
        profile_rows.append(
            {
                "record_uid": f"train_{i}",
                "split": "train",
                "dataset_index": int(train_idx[i]),
                "source_text": src,
                "target_text": tgt,
                "features": feats,
                "labels": labels,
                "route": decision,
                "top_flags": top_flags(feats, labels),
            }
        )

    # Val rows with score-aware routing + route policy prediction.
    route_preds = []
    route_names = []
    for i, row in val_df.iterrows():
        src = str(row[src_col])
        tgt = str(row[tgt_col])
        feats = extract_source_features(src, token_freq=tok_freq)
        feats.update(extract_target_features(tgt))
        labels = label_features(feats, thr)
        top_score = internal_top_scores[i]
        gap = internal_gaps[i]
        decision = route_decision(labels, internal_top_score=top_score, internal_gap=gap, routing_map=routing_map)
        route_name = str(decision["route"])
        route_names.append(route_name)
        if route_name == "RETRIEVE_INTERNAL":
            pred = policy_preds["internal_only"][i]
        elif route_name == "RERANK_STRONG":
            pred = policy_preds["strong_rerank"][i]
        elif route_name == "RETRIEVE_HYBRID":
            pred = policy_preds["hybrid"][i]
        else:
            pred = policy_preds["fallback"][i]
        route_preds.append(pred)
        profile_rows.append(
            {
                "record_uid": f"val_{i}",
                "split": "val",
                "dataset_index": int(val_idx[i]),
                "source_text": src,
                "target_text": tgt,
                "features": feats,
                "labels": labels,
                "route": decision,
                "top_flags": top_flags(feats, labels),
            }
        )

    # Test rows: fit internal model on full train to estimate confidence scores.
    full_internal = CanonicalRetrievalTranslator(config=CANONICAL_BASELINE_V2)
    full_internal.fit(
        train_src=train_all[src_col].tolist(),
        train_tgt=train_all[tgt_col].tolist(),
    )
    _, test_debug = full_internal.predict(test_src, return_debug=True)
    for i, row in test_df.iterrows():
        src = str(row[test_src_col])
        feats = extract_source_features(src, token_freq=tok_freq)
        labels = label_features(feats, thr)
        top_score = float(test_debug[i].get("internal_top_cosine", test_debug[i].get("global_score", 0.0)))
        gap = test_debug[i].get("internal_gap", None)
        decision = route_decision(labels, internal_top_score=top_score, internal_gap=gap, routing_map=routing_map)
        profile_rows.append(
            {
                "record_uid": f"test_{i}",
                "split": "test",
                "dataset_index": int(i),
                "test_id": row[test_id_col] if test_id_col in test_df.columns else i,
                "source_text": src,
                "target_text": "",
                "features": feats,
                "labels": labels,
                "route": decision,
                "top_flags": top_flags(feats, labels),
            }
        )

    # Bucketed eval table.
    val_profiles = [r for r in profile_rows if r["split"] == "val"]
    bucket_dims = [
        "length_bucket",
        "fragmentation",
        "formula_density",
        "numeric_density",
        "proper_name_density",
        "template_type",
    ]
    bucket_rows = []
    for dim in bucket_dims:
        values = sorted({vp["labels"][dim] for vp in val_profiles})
        for v in values:
            idxs = [i for i, vp in enumerate(val_profiles) if vp["labels"][dim] == v]
            for policy_name, preds in policy_preds.items():
                bucket_rows.append(
                    {
                        "dimension": dim,
                        "bucket": v,
                        "n": len(idxs),
                        "policy": policy_name,
                        "chrf2": _subset_chrf(preds, val_tgt, idxs),
                    }
                )
            bucket_rows.append(
                {
                    "dimension": dim,
                    "bucket": v,
                    "n": len(idxs),
                    "policy": "routed",
                    "chrf2": _subset_chrf(route_preds, val_tgt, idxs),
                }
            )
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_csv = reports_dir / "val_bucket_metrics.csv"
    bucket_df.to_csv(bucket_csv, index=False)

    # Routing ablation.
    best_policy_per_item = []
    routed_beats_internal = 0
    for i in range(len(val_tgt)):
        sent_scores = {}
        for pn, preds in policy_preds.items():
            sent_scores[pn] = float(sacrebleu.sentence_chrf(preds[i], [val_tgt[i]], word_order=2).score)
        best_p = max(sent_scores, key=sent_scores.get)
        best_policy_per_item.append(best_p)
        route_sent = float(sacrebleu.sentence_chrf(route_preds[i], [val_tgt[i]], word_order=2).score)
        base_sent = float(sacrebleu.sentence_chrf(policy_preds["internal_only"][i], [val_tgt[i]], word_order=2).score)
        if route_sent > base_sent:
            routed_beats_internal += 1

    routed_score = _score(route_preds, val_tgt)
    route_counts = {}
    for rn in route_names:
        route_counts[rn] = route_counts.get(rn, 0) + 1
    best_policy_counts = {}
    for p in best_policy_per_item:
        best_policy_counts[p] = best_policy_counts.get(p, 0) + 1
    routing_ablation = {
        "baseline_internal_chrF2": policy_scores["internal_only"]["chrf2"],
        "routed_chrF2": routed_score["chrf2"],
        "delta_routed_vs_internal": float(routed_score["chrf2"] - policy_scores["internal_only"]["chrf2"]),
        "route_counts": route_counts,
        "counterfactual_best_policy_counts": best_policy_counts,
        "routed_beats_internal_count": int(routed_beats_internal),
        "val_rows": len(val_tgt),
    }
    (reports_dir / "routing_ablation.json").write_text(
        json.dumps(routing_ablation, indent=2),
        encoding="utf-8",
    )

    # Flatten profile rows for parquet/jsonl.
    flat_rows = []
    for r in profile_rows:
        labels = r["labels"]
        conf = labels["confidence"]
        row = {
            "record_uid": r["record_uid"],
            "split": r["split"],
            "dataset_index": r["dataset_index"],
            "test_id": str(r.get("test_id", "")),
            "source_text": r["source_text"],
            "target_text": r["target_text"],
            "route": r["route"]["route"],
            "route_policy_params_json": json.dumps(r["route"]["policy_params"], ensure_ascii=False),
            "route_rationale": r["route"]["rationale"],
            "top_flags_json": json.dumps(r["top_flags"], ensure_ascii=False),
            "overall_confidence": float(labels["overall_confidence"]),
            "length_bucket": labels["length_bucket"],
            "fragmentation": labels["fragmentation"],
            "formula_density": labels["formula_density"],
            "numeric_density": labels["numeric_density"],
            "proper_name_density": labels["proper_name_density"],
            "template_type": labels["template_type"],
            "layout_hint": labels["layout_hint"],
            "role_hint": labels["role_hint"],
            "conf_length_bucket": float(conf["length_bucket"]),
            "conf_fragmentation": float(conf["fragmentation"]),
            "conf_formula_density": float(conf["formula_density"]),
            "conf_numeric_density": float(conf["numeric_density"]),
            "conf_proper_name_density": float(conf["proper_name_density"]),
            "conf_template_type": float(conf["template_type"]),
            "conf_layout_hint": float(conf["layout_hint"]),
            "conf_role_hint": float(conf["role_hint"]),
            "feature_vector_json": json.dumps(r["features"], ensure_ascii=False),
        }
        for k, v in r["features"].items():
            row[f"f_{k}"] = float(v)
        flat_rows.append(row)

    profile_df = pd.DataFrame(flat_rows)
    parquet_path = out_dir / "corpus_profiles.parquet"
    jsonl_path = out_dir / "corpus_profiles.jsonl"
    profile_df.to_parquet(parquet_path, index=False)
    _write_jsonl(jsonl_path, profile_df.to_dict(orient="records"))

    # Routing map + report.
    routing_map_path = out_dir / "routing_map.json"
    routing_map_payload = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "thresholds": thr.to_dict(),
        "internal_score_stats": score_stats,
        "routing_map": routing_map,
    }
    routing_map_path.write_text(json.dumps(routing_map_payload, indent=2), encoding="utf-8")

    profile_report = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "thresholds": thr.to_dict(),
        "policy_scores_val": policy_scores,
        "routed_score_val": routed_score,
        "delta_routed_vs_internal": float(routed_score["chrf2"] - policy_scores["internal_only"]["chrf2"]),
        "route_counts_val": route_counts,
        "outputs": {
            "corpus_profiles_parquet": str(parquet_path.resolve()),
            "corpus_profiles_jsonl": str(jsonl_path.resolve()),
            "routing_map_json": str(routing_map_path.resolve()),
            "bucket_metrics_csv": str(bucket_csv.resolve()),
            "routing_ablation_json": str((reports_dir / "routing_ablation.json").resolve()),
        },
    }
    profile_report_path = out_dir / f"profile_report_seed{args.seed}.json"
    profile_report_path.write_text(json.dumps(profile_report, indent=2), encoding="utf-8")

    # Profile samples (human inspection).
    sample_dims = ["length_bucket", "fragmentation", "template_type"]
    prof_rng = random.Random(args.seed)
    for dim in sample_dims:
        values = sorted(profile_df[dim].dropna().unique().tolist())
        for v in values:
            sub = profile_df[profile_df[dim] == v]
            rows = sub.to_dict(orient="records")
            prof_rng.shuffle(rows)
            rows = rows[: args.sample_per_bucket]
            _write_jsonl(samples_dir / f"{dim}_{v}.jsonl", rows)

    print(f"Wrote {parquet_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {routing_map_path}")
    print(f"Wrote {profile_report_path}")
    print(f"Wrote {bucket_csv}")
    print(f"Wrote {reports_dir / 'routing_ablation.json'}")


if __name__ == "__main__":
    main()

