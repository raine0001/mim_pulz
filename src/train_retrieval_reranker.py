from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from sklearn.linear_model import LogisticRegression


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.domain_intent import infer_dialog_domain_with_confidence
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator, RetrievalConfig
from utils_manifest import write_json


FEATURE_NAMES = [
    "cosine",
    "base_score",
    "stage2_norm",
    "rank_inv",
    "is_external",
    "domain_match",
    "len_ratio",
    "digits_overlap",
    "pn_overlap",
]


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


def _digits_overlap(a: str, b: str) -> float:
    da = set(re.findall(r"\d+", a or ""))
    db = set(re.findall(r"\d+", b or ""))
    if not da and not db:
        return 0.0
    inter = len(da.intersection(db))
    union = len(da.union(db))
    return float(inter / union) if union else 0.0


def _pn_tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z-]{1,}", text or "")
    out = set()
    for t in toks:
        if any(ch.isupper() for ch in t):
            out.add(t.lower())
    return out


def _pn_overlap(a: str, b: str) -> float:
    pa = _pn_tokens(a)
    pb = _pn_tokens(b)
    if not pa and not pb:
        return 0.0
    inter = len(pa.intersection(pb))
    union = len(pa.union(pb))
    return float(inter / union) if union else 0.0


def _profile_oracc_best(repo_root: Path) -> RetrievalConfig:
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


def _build_feature_matrix_for_query(
    *,
    model: CanonicalRetrievalTranslator,
    query_text_raw: str,
    row: np.ndarray,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    q_norm = model._norm(query_text_raw)
    q_len = len(q_norm)
    q_domain = infer_dialog_domain_with_confidence(q_norm)[0]
    bm25_row = (
        model.stage2_bm25.score_one(q_norm)
        if (model.config.stage2_type == "bm25" and model.stage2_bm25 is not None)
        else None
    )
    _, base_scores, stage2_norm = model._rerank_components(
        row=row,
        query=q_norm,
        q_len=q_len,
        candidates=candidate_idx,
        bm25_row=bm25_row,
    )

    feats = []
    n = len(candidate_idx)
    for rank, idx in enumerate(candidate_idx):
        idx_i = int(idx)
        len_ratio = float(model._len_score(q_len=q_len, d_len=int(model.train_lens[idx_i])))
        is_external = 1.0 if model.train_origins[idx_i] != "competition" else 0.0
        domain_match = 1.0 if model.train_domains[idx_i] == q_domain else 0.0
        digits_ov = _digits_overlap(query_text_raw, model.train_src_raw[idx_i])
        pn_ov = _pn_overlap(query_text_raw, model.train_src_raw[idx_i])
        rank_inv = 1.0 / float(rank + 1)
        feats.append(
            [
                float(row[idx_i]),
                float(base_scores[rank]),
                float(stage2_norm[rank]),
                rank_inv,
                is_external,
                domain_match,
                len_ratio,
                digits_ov,
                pn_ov,
            ]
        )
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a lightweight logistic reranker on retrieval candidate features."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--memory-profile", type=str, choices=["internal", "oracc_best"], default="oracc_best")
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--train-query-samples", type=int, default=400)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_reranker_eval_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    if args.memory_profile == "internal":
        base_cfg = CANONICAL_BASELINE_V2
    else:
        base_cfg = _profile_oracc_best(PATHS.root)
    model = CanonicalRetrievalTranslator(config=base_cfg)
    model.fit(train_src=train_src, train_tgt=train_tgt)

    # Baseline retrieval score on val.
    baseline_preds = model.predict(val_src)
    baseline_score = _score(baseline_preds, val_tgt)

    # Build training data for reranker from train-side queries (exclude self candidate).
    q_norm_train = [model._norm(x) for x in train_src]
    q_mat_train = model.vectorizer.transform(q_norm_train)
    sims_train = np.asarray((q_mat_train @ model.train_matrix.T).todense(), dtype=np.float32)

    candidate_pool = int(max(5, args.candidate_pool))
    n_train = len(train_src)
    all_train_indices = list(range(n_train))
    random.shuffle(all_train_indices)
    train_query_indices = all_train_indices[: min(int(args.train_query_samples), n_train)]

    x_rows = []
    y_rows = []
    meta_rows = 0
    for qi in train_query_indices:
        row = sims_train[qi]
        cands = model._candidate_indices(row, min(candidate_pool + 10, len(row)))
        # Exclude the exact same training row from candidates.
        cands = np.asarray([int(c) for c in cands if int(c) != int(qi)], dtype=np.int32)
        if cands.size == 0:
            continue
        cands = cands[:candidate_pool]
        feats = _build_feature_matrix_for_query(
            model=model,
            query_text_raw=train_src[qi],
            row=row,
            candidate_idx=cands,
        )
        if feats.shape[0] == 0:
            continue
        gold = train_tgt[qi]
        sent_scores = [
            float(sacrebleu.sentence_chrf(model.train_tgt[int(idx)], [gold], word_order=2).score)
            for idx in cands
        ]
        best_pos = int(np.argmax(np.asarray(sent_scores, dtype=np.float32)))
        labels = np.zeros(feats.shape[0], dtype=np.int32)
        labels[best_pos] = 1
        x_rows.append(feats)
        y_rows.append(labels)
        meta_rows += int(feats.shape[0])

    if not x_rows:
        raise RuntimeError("No reranker training rows generated.")
    x_train = np.vstack(x_rows)
    y_train = np.concatenate(y_rows)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=args.seed)
    clf.fit(x_train, y_train)

    # Evaluate reranker on val.
    q_norm_val = [model._norm(x) for x in val_src]
    q_mat_val = model.vectorizer.transform(q_norm_val)
    sims_val = np.asarray((q_mat_val @ model.train_matrix.T).todense(), dtype=np.float32)

    rerank_preds = []
    external_pick_count = 0
    for i, row in enumerate(sims_val):
        cands = model._candidate_indices(row, min(candidate_pool, len(row)))
        feats = _build_feature_matrix_for_query(
            model=model,
            query_text_raw=val_src[i],
            row=row,
            candidate_idx=cands,
        )
        if feats.shape[0] == 0:
            cidx = int(cands[0])
        else:
            proba = clf.predict_proba(feats)[:, 1]
            best_pos = int(np.argmax(proba))
            cidx = int(cands[best_pos])
        if model.train_origins[cidx] != "competition":
            external_pick_count += 1
        rerank_preds.append(model.train_tgt[cidx])
    rerank_score = _score(rerank_preds, val_tgt)

    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "memory_profile": args.memory_profile,
        "base_retrieval_config": asdict(base_cfg),
        "candidate_pool": int(candidate_pool),
        "train_query_samples": len(train_query_indices),
        "reranker_train_rows": int(meta_rows),
        "reranker_positive_rate": float(float(y_train.mean())),
        "coefficients": {
            "intercept": float(clf.intercept_[0]),
            "feature_names": FEATURE_NAMES,
            "weights": [float(w) for w in clf.coef_[0].tolist()],
        },
        "baseline": baseline_score,
        "reranker": rerank_score,
        "delta_chrF2": float(rerank_score["chrf2"] - baseline_score["chrf2"]),
        "reranker_external_pick_rate": float(external_pick_count / len(val_src)) if val_src else 0.0,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print(f"Baseline chrF2\t{baseline_score['chrf2']:.4f}")
    print(f"Reranker chrF2\t{rerank_score['chrf2']:.4f}")
    print(f"Delta\t{summary['delta_chrF2']:.4f}")
    print(f"Train rows\t{summary['reranker_train_rows']}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()

