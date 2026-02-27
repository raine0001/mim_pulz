from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import sacrebleu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.eval_metrics import corpus_metrics
from mim_pulz.routed_reranker import FEATURE_NAMES, LinearRerankerModel
from utils_manifest import write_json


BASE_SCORE_IDX = FEATURE_NAMES.index("base_score")
COSINE_IDX = FEATURE_NAMES.index("cosine")
DIGITS_OVERLAP_IDX = FEATURE_NAMES.index("digits_overlap")


def _score(preds: list[str], gold: list[str]) -> dict:
    s = corpus_metrics(preds, gold)
    s_ref = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return {
        "bleu": float(s["bleu"]),
        "chrf2": float(s["chrf2"]),
        "combined": float(s["combined"]),
        "chrf2_reference": s_ref,
        "abs_diff": float(abs(float(s["chrf2"]) - s_ref)),
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _feature_vec(feat: dict) -> list[float]:
    return [float(feat.get(name, 0.0)) for name in FEATURE_NAMES]


def _parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [float(v) for v in vals]


def _augment_features(raw_x: np.ndarray) -> np.ndarray:
    mat = np.asarray(raw_x, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.shape}")
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-6
    z = (mat - mu) / sd
    top_idx = int(np.argmax(mat[:, BASE_SCORE_IDX]))
    rel = mat - mat[top_idx : top_idx + 1, :]
    return np.concatenate([mat, z, rel], axis=1)


def _candidate_score(cand: dict, optimize_target: str) -> float:
    if optimize_target == "bleu":
        return float(cand.get("sentence_bleu", 0.0))
    if optimize_target == "chrf2":
        return float(cand.get("sentence_chrf2", 0.0))
    if "sentence_combined" in cand:
        return float(cand.get("sentence_combined", 0.0))
    bleu = float(cand.get("sentence_bleu", 0.0))
    chrf = float(cand.get("sentence_chrf2", 0.0))
    return float(np.sqrt(max(0.0, bleu) * max(0.0, chrf)))


def _baseline_pos(row_obj: dict, candidates: list[dict], raw_x: np.ndarray) -> int:
    target_idx = row_obj.get("current_baseline_memory_idx", None)
    if target_idx is not None:
        for i, cand in enumerate(candidates):
            if int(cand.get("memory_idx", -1)) == int(target_idx):
                return int(i)
    for i, cand in enumerate(candidates):
        if bool(cand.get("is_current_choice", False)):
            return int(i)
    return int(np.argmax(raw_x[:, BASE_SCORE_IDX])) if raw_x.size else 0


def _is_hard_query(
    raw_x: np.ndarray,
    *,
    hard_gap_threshold: float,
    hard_top_cos_threshold: float,
    hard_close_band: float,
    hard_close_min: int,
) -> bool:
    if raw_x.shape[0] < 2:
        return False
    base = np.asarray(raw_x[:, BASE_SCORE_IDX], dtype=np.float32)
    top = float(np.max(base))
    sorted_base = np.sort(base)[::-1]
    gap = float(sorted_base[0] - sorted_base[1]) if len(sorted_base) > 1 else 1.0
    close_count = int(np.sum(base >= (top - float(hard_close_band))))
    top_cos = float(np.max(raw_x[:, COSINE_IDX]))
    return bool(
        gap <= float(hard_gap_threshold)
        or top_cos <= float(hard_top_cos_threshold)
        or close_count >= int(max(2, hard_close_min))
    )


def _build_pairwise_examples(
    aug_x: np.ndarray,
    scores: np.ndarray,
    *,
    baseline_pos: int,
    min_delta: float,
    top_winners: int,
    bottom_losers: int,
) -> tuple[list[np.ndarray], list[int]]:
    n = len(scores)
    if n < 2:
        return [], []
    winners_n = int(max(1, min(top_winners, n)))
    losers_n = int(max(1, min(bottom_losers, n)))
    order = np.argsort(scores)[::-1]
    winners = [int(x) for x in order[:winners_n]]
    losers = [int(x) for x in order[-losers_n:]]
    if int(baseline_pos) not in losers:
        losers.append(int(baseline_pos))

    diffs: list[np.ndarray] = []
    labels: list[int] = []

    for wi in winners:
        for li in losers:
            if wi == li:
                continue
            if float(scores[wi]) <= float(scores[li]) + float(min_delta):
                continue
            d = np.asarray(aug_x[wi] - aug_x[li], dtype=np.float32)
            diffs.append(d)
            labels.append(1)
            diffs.append(-d)
            labels.append(0)

    for i in range(n):
        if i == int(baseline_pos):
            continue
        delta = float(scores[i] - scores[int(baseline_pos)])
        if abs(delta) < float(min_delta):
            continue
        if delta > 0:
            d = np.asarray(aug_x[i] - aug_x[int(baseline_pos)], dtype=np.float32)
        else:
            d = np.asarray(aug_x[int(baseline_pos)] - aug_x[i], dtype=np.float32)
        diffs.append(d)
        labels.append(1)
        diffs.append(-d)
        labels.append(0)

    return diffs, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate pairwise linear reranker from routed candidate exports."
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "routed_candidates_seed42",
    )
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=PATHS.root / "artifacts" / "models" / "routed_linear_reranker_seed42.json",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "routed_linear_reranker_eval_seed42.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-target", type=str, choices=["combined", "chrf2", "bleu"], default="combined")
    parser.add_argument("--target-baseline", type=float, default=25.7786)
    parser.add_argument("--logistic-c", type=float, default=0.25)
    parser.add_argument("--pairwise-min-delta", type=float, default=0.05)
    parser.add_argument("--pairwise-top-winners", type=int, default=4)
    parser.add_argument("--pairwise-bottom-losers", type=int, default=6)
    parser.add_argument("--hard-gap-threshold", type=float, default=0.02)
    parser.add_argument("--hard-top-cos-threshold", type=float, default=0.52)
    parser.add_argument("--hard-close-band", type=float, default=0.03)
    parser.add_argument("--hard-close-min", type=int, default=3)
    parser.add_argument("--hard-query-weight", type=float, default=2.0)
    parser.add_argument("--easy-keep-prob", type=float, default=0.35)
    parser.add_argument("--gate-prob-margins", type=str, default="0.02,0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--gate-base-drops", type=str, default="0.02,0.03,0.05,0.08,0.10")
    parser.add_argument("--gate-pred-delta-thresholds", type=str, default="0.000,0.005,0.010,0.015")
    parser.add_argument("--gate-digit-overlap-tols", type=str, default="0.00,0.02,0.05")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cdir = args.candidates_dir if args.candidates_dir.is_absolute() else (PATHS.root / args.candidates_dir)
    train_path = args.train_jsonl if args.train_jsonl is not None else (cdir / "train_candidates.jsonl")
    val_path = args.val_jsonl if args.val_jsonl is not None else (cdir / "val_candidates.jsonl")
    train_path = train_path if train_path.is_absolute() else (PATHS.root / train_path)
    val_path = val_path if val_path.is_absolute() else (PATHS.root / val_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train candidates JSONL: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val candidates JSONL: {val_path}")

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    rng = np.random.default_rng(int(args.seed))

    x_pairs: list[np.ndarray] = []
    y_pairs: list[int] = []
    w_pairs: list[float] = []
    query_count = 0
    hard_query_count = 0
    kept_easy_count = 0

    for row in train_rows:
        candidates = list(row.get("candidates", []))
        if len(candidates) < 2:
            continue
        raw_x = np.asarray([_feature_vec(dict(c.get("features", {}))) for c in candidates], dtype=np.float32)
        if raw_x.shape[0] < 2:
            continue
        aug_x = _augment_features(raw_x)
        cand_scores = np.asarray([_candidate_score(c, str(args.optimize_target)) for c in candidates], dtype=np.float32)
        base_pos = _baseline_pos(row, candidates, raw_x)
        hard = _is_hard_query(
            raw_x,
            hard_gap_threshold=float(args.hard_gap_threshold),
            hard_top_cos_threshold=float(args.hard_top_cos_threshold),
            hard_close_band=float(args.hard_close_band),
            hard_close_min=int(args.hard_close_min),
        )
        if hard:
            hard_query_count += 1
        else:
            if float(args.easy_keep_prob) <= 0.0 or rng.random() > float(args.easy_keep_prob):
                continue
            kept_easy_count += 1

        pair_x, pair_y = _build_pairwise_examples(
            aug_x,
            cand_scores,
            baseline_pos=int(base_pos),
            min_delta=float(args.pairwise_min_delta),
            top_winners=int(args.pairwise_top_winners),
            bottom_losers=int(args.pairwise_bottom_losers),
        )
        if not pair_x:
            continue

        weight = float(args.hard_query_weight) if hard else 1.0
        x_pairs.extend(pair_x)
        y_pairs.extend(pair_y)
        w_pairs.extend([weight] * len(pair_y))
        query_count += 1

    if not x_pairs:
        raise RuntimeError("No pairwise training examples produced from train candidate rows.")

    x_train = np.vstack(x_pairs).astype(np.float32)
    y_train = np.asarray(y_pairs, dtype=np.int32)
    sample_weight = np.asarray(w_pairs, dtype=np.float32)

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
    clf.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)

    baseline_preds: list[str] = []
    reranked_preds: list[str] = []
    oracle_preds: list[str] = []
    val_gold: list[str] = []
    reranker_pick_external = 0
    total_val = 0

    val_cache: list[dict] = []
    for row in val_rows:
        candidates = list(row.get("candidates", []))
        if len(candidates) < 2:
            continue
        gold = str(row.get("gold", ""))
        val_gold.append(gold)

        raw_x = np.asarray([_feature_vec(dict(c.get("features", {}))) for c in candidates], dtype=np.float32)
        if raw_x.shape[0] < 2:
            continue
        aug_x = _augment_features(raw_x)
        proba = clf.predict_proba(aug_x)[:, 1]
        cand_scores = np.asarray([_candidate_score(c, str(args.optimize_target)) for c in candidates], dtype=np.float32)

        base_pos = _baseline_pos(row, candidates, raw_x)
        baseline_text = str(candidates[base_pos].get("target_text", ""))
        baseline_preds.append(baseline_text)

        best_pos = int(np.argmax(proba))
        best = candidates[best_pos]
        reranked_preds.append(str(best.get("target_text", "")))
        if str(best.get("origin", "")) != "competition":
            reranker_pick_external += 1

        oracle_pos = int(np.argmax(cand_scores))
        oracle_preds.append(str(candidates[oracle_pos].get("target_text", "")))

        labels_obj = dict(row.get("labels", {}))
        q_text = str(row.get("query", ""))
        numeric_heavy = bool(
            str(labels_obj.get("numeric_density", "low")) == "high"
            or any(ch.isdigit() for ch in q_text)
        )

        val_cache.append(
            {
                "candidates": candidates,
                "raw_x": raw_x,
                "proba": proba,
                "base_pos": int(base_pos),
                "best_pos_raw": int(best_pos),
                "cand_scores": cand_scores,
                "numeric_heavy": bool(numeric_heavy),
            }
        )
        total_val += 1

    if not val_gold:
        raise RuntimeError("No validation rows available for reranker evaluation.")

    baseline_score = _score(baseline_preds, val_gold)
    reranked_score = _score(reranked_preds, val_gold)
    oracle_score = _score(oracle_preds, val_gold)

    cal_x: list[float] = []
    cal_y: list[float] = []
    for row in val_cache:
        base_pos = int(row["base_pos"])
        best_pos = int(row["best_pos_raw"])
        proba = np.asarray(row["proba"], dtype=np.float32)
        score_arr = np.asarray(row["cand_scores"], dtype=np.float32)
        cal_x.append(float(proba[best_pos] - proba[base_pos]))
        cal_y.append(float(score_arr[best_pos] - score_arr[base_pos]))
    if len(cal_x) >= 2 and float(np.std(np.asarray(cal_x, dtype=np.float32))) > 1e-6:
        slope, intercept = np.polyfit(np.asarray(cal_x, dtype=np.float32), np.asarray(cal_y, dtype=np.float32), deg=1)
        delta_cal_slope = float(np.clip(slope, -20.0, 20.0))
        delta_cal_intercept = float(np.clip(intercept, -5.0, 5.0))
    else:
        delta_cal_slope = 1.0
        delta_cal_intercept = 0.0

    gate_prob_grid = _parse_float_list(args.gate_prob_margins)
    gate_base_grid = _parse_float_list(args.gate_base_drops)
    gate_delta_grid = _parse_float_list(args.gate_pred_delta_thresholds)
    gate_digit_tol_grid = _parse_float_list(args.gate_digit_overlap_tols)

    best_gate_score = reranked_score
    best_gate_preds = reranked_preds
    best_gate = {
        "prob_margin": 0.0,
        "base_score_drop": -1.0,
        "min_pred_delta": 0.0,
        "digit_overlap_tol": 0.0,
        "changed_count": int(sum(1 for row in val_cache if int(np.argmax(row["proba"])) != int(row["base_pos"]))),
        "changed_rate": float(
            sum(1 for row in val_cache if int(np.argmax(row["proba"])) != int(row["base_pos"])) / max(1, len(val_cache))
        ),
        "digit_veto_count": 0,
    }

    for pm in gate_prob_grid:
        for bd in gate_base_grid:
            for dthr in gate_delta_grid:
                for dtol in gate_digit_tol_grid:
                    gated_preds: list[str] = []
                    changed = 0
                    digit_veto_count = 0
                    for row in val_cache:
                        cands = row["candidates"]
                        raw_x = row["raw_x"]
                        proba = row["proba"]
                        base_pos = int(row["base_pos"])
                        best_pos = int(np.argmax(proba))

                        prob_gap = float(proba[best_pos] - proba[base_pos])
                        best_base = float(raw_x[best_pos, BASE_SCORE_IDX])
                        base_base = float(raw_x[base_pos, BASE_SCORE_IDX])
                        pred_delta = float(delta_cal_intercept + delta_cal_slope * prob_gap)

                        allow_prob = prob_gap >= float(pm)
                        allow_base = best_base >= (base_base - float(bd))
                        allow_delta = pred_delta >= float(dthr)
                        allow_digits = True
                        if bool(row.get("numeric_heavy", False)) and best_pos != base_pos:
                            best_digits = float(raw_x[best_pos, DIGITS_OVERLAP_IDX])
                            base_digits = float(raw_x[base_pos, DIGITS_OVERLAP_IDX])
                            if best_digits + float(dtol) < base_digits:
                                allow_digits = False
                                digit_veto_count += 1

                        if not (allow_prob and allow_base and allow_delta and allow_digits):
                            best_pos = base_pos
                        if best_pos != base_pos:
                            changed += 1
                        gated_preds.append(str(cands[best_pos].get("target_text", "")))

                    gated_score = _score(gated_preds, val_gold)
                    if gated_score[str(args.optimize_target)] > best_gate_score[str(args.optimize_target)]:
                        best_gate_score = gated_score
                        best_gate_preds = gated_preds
                        best_gate = {
                            "prob_margin": float(pm),
                            "base_score_drop": float(bd),
                            "min_pred_delta": float(dthr),
                            "digit_overlap_tol": float(dtol),
                            "changed_count": int(changed),
                            "changed_rate": float(changed / max(1, len(val_cache))),
                            "digit_veto_count": int(digit_veto_count),
                        }

    model = LinearRerankerModel(
        feature_names=FEATURE_NAMES,
        weights=tuple(float(w) for w in clf.named_steps["logisticregression"].coef_[0].tolist()),
        intercept=float(clf.named_steps["logisticregression"].intercept_[0]),
        model_id="linear_pairwise_reranker_v3",
        feature_transform="raw_z_rel_to_top_base",
        gate_prob_margin=float(best_gate["prob_margin"]),
        gate_base_score_drop=float(best_gate["base_score_drop"]),
        gate_min_pred_delta=float(best_gate["min_pred_delta"]),
        delta_calibration_slope=float(delta_cal_slope),
        delta_calibration_intercept=float(delta_cal_intercept),
        gate_numeric_digit_overlap_tol=float(best_gate["digit_overlap_tol"]),
    )

    model_payload = {
        **model.to_payload(),
        "seed": int(args.seed),
        "optimize_target": str(args.optimize_target),
        "train_queries": int(query_count),
        "hard_query_count": int(hard_query_count),
        "kept_easy_count": int(kept_easy_count),
        "train_rows": int(len(train_rows)),
        "train_examples": int(len(y_train)),
        "positive_rate": float(float(y_train.mean())),
        "sample_weight_mean": float(sample_weight.mean()),
        "sample_weight_max": float(sample_weight.max()),
        "pairwise_min_delta": float(args.pairwise_min_delta),
        "pairwise_top_winners": int(args.pairwise_top_winners),
        "pairwise_bottom_losers": int(args.pairwise_bottom_losers),
        "hard_gap_threshold": float(args.hard_gap_threshold),
        "hard_top_cos_threshold": float(args.hard_top_cos_threshold),
        "hard_close_band": float(args.hard_close_band),
        "hard_close_min": int(args.hard_close_min),
        "hard_query_weight": float(args.hard_query_weight),
        "easy_keep_prob": float(args.easy_keep_prob),
        "logistic_c": float(args.logistic_c),
        "delta_calibration_slope": float(delta_cal_slope),
        "delta_calibration_intercept": float(delta_cal_intercept),
        "augmented_feature_dim": int(x_train.shape[1]),
        "input_feature_names": list(FEATURE_NAMES),
        "source_train_jsonl": str(train_path.resolve()),
        "source_val_jsonl": str(val_path.resolve()),
    }

    model_out = args.model_out if args.model_out.is_absolute() else (PATHS.root / args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    report = {
        "seed": int(args.seed),
        "feature_names": list(FEATURE_NAMES),
        "feature_transform": "raw_z_rel_to_top_base",
        "weights": [float(w) for w in model.weights],
        "intercept": float(model.intercept),
        "top_weights": model.top_weight_items(top_n=8),
        "optimize_target": str(args.optimize_target),
        "train_queries": int(query_count),
        "hard_query_count": int(hard_query_count),
        "kept_easy_count": int(kept_easy_count),
        "train_rows": int(len(train_rows)),
        "val_rows": int(total_val),
        "train_examples": int(len(y_train)),
        "positive_rate": float(float(y_train.mean())),
        "sample_weight_mean": float(sample_weight.mean()),
        "sample_weight_max": float(sample_weight.max()),
        "pairwise_min_delta": float(args.pairwise_min_delta),
        "pairwise_top_winners": int(args.pairwise_top_winners),
        "pairwise_bottom_losers": int(args.pairwise_bottom_losers),
        "hard_gap_threshold": float(args.hard_gap_threshold),
        "hard_top_cos_threshold": float(args.hard_top_cos_threshold),
        "hard_close_band": float(args.hard_close_band),
        "hard_close_min": int(args.hard_close_min),
        "hard_query_weight": float(args.hard_query_weight),
        "easy_keep_prob": float(args.easy_keep_prob),
        "logistic_c": float(args.logistic_c),
        "delta_calibration_slope": float(delta_cal_slope),
        "delta_calibration_intercept": float(delta_cal_intercept),
        "baseline_routed": baseline_score,
        "reranked_raw": reranked_score,
        "reranked_gated": best_gate_score,
        "selected_gate": best_gate,
        "oracle_from_candidate_pool": oracle_score,
        "delta_vs_baseline_raw_combined": float(reranked_score["combined"] - baseline_score["combined"]),
        "delta_vs_baseline_gated_combined": float(best_gate_score["combined"] - baseline_score["combined"]),
        "delta_vs_baseline_raw": float(reranked_score["chrf2"] - baseline_score["chrf2"]),
        "delta_vs_baseline_gated": float(best_gate_score["chrf2"] - baseline_score["chrf2"]),
        "delta_vs_target_baseline": float(best_gate_score[str(args.optimize_target)] - float(args.target_baseline)),
        "reranker_external_pick_rate_raw": float(reranker_pick_external / total_val) if total_val else 0.0,
        "target_baseline": float(args.target_baseline),
        "model_out": str(model_out.resolve()),
        "train_jsonl": str(train_path.resolve()),
        "val_jsonl": str(val_path.resolve()),
    }

    report_path = args.report_json if args.report_json.is_absolute() else (PATHS.root / args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)

    print(
        "Baseline routed\t"
        f"BLEU={baseline_score['bleu']:.4f},"
        f"chrF2={baseline_score['chrf2']:.4f},"
        f"combined={baseline_score['combined']:.4f}"
    )
    print(
        "Reranked raw\t"
        f"BLEU={reranked_score['bleu']:.4f},"
        f"chrF2={reranked_score['chrf2']:.4f},"
        f"combined={reranked_score['combined']:.4f}"
    )
    print(
        "Reranked gated\t"
        f"BLEU={best_gate_score['bleu']:.4f},"
        f"chrF2={best_gate_score['chrf2']:.4f},"
        f"combined={best_gate_score['combined']:.4f}"
    )
    print(
        "Gate\t"
        f"prob_margin={best_gate['prob_margin']},"
        f" base_score_drop={best_gate['base_score_drop']},"
        f" min_pred_delta={best_gate['min_pred_delta']},"
        f" digit_overlap_tol={best_gate['digit_overlap_tol']}"
    )
    print(f"Delta vs routed (gated, combined)\t{report['delta_vs_baseline_gated_combined']:.4f}")
    print(
        "Oracle pool\t"
        f"BLEU={oracle_score['bleu']:.4f},"
        f"chrF2={oracle_score['chrf2']:.4f},"
        f"combined={oracle_score['combined']:.4f}"
    )
    print(f"Model\t{model_out}")
    print(f"Report\t{report_path}")


if __name__ == "__main__":
    main()
