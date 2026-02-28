from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from mim_pulz.domain_intent import infer_dialog_domain_with_confidence


FEATURE_NAMES: tuple[str, ...] = (
    "cosine",
    "base_score",
    "stage2_norm",
    "rank_inv",
    "is_external",
    "domain_match",
    "len_ratio",
    "digits_overlap",
    "pn_overlap",
    "section_match",
    "digit_ratio_proximity",
    "bracket_ratio_proximity",
    "unknown_ratio_proximity",
    "repeat_trigram_proximity",
    "marker_density_proximity",
)

_BRACKET_CHARS = set("[](){}<>")
_UNKNOWN_TOKENS = {"x", "xx", "xxx", "xxxx", "?", "...", "[...]", "x-x", "x.x"}
_CLOSING_MARKER_RE = re.compile(
    r"\b("
    r"igi|witness(?:es)?|seal(?:ed)?|colophon|date(?:d)?|month|year|day|"
    r"limmu|itu|mu|ki[šs]ib|kunuk|total|sum|subtotal|balance|"
    r"šu\.?nigin2?|nigin2?|debet|credit|obv|rev|edge"
    r")\b",
    flags=re.IGNORECASE,
)


def _repeat_trigram_rate(tokens: list[str]) -> float:
    if len(tokens) < 3:
        return 0.0
    grams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not grams:
        return 0.0
    counts: dict[tuple[str, str, str], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeated = sum(1 for g in grams if counts[g] > 1)
    return float(repeated / len(grams))


def _section_from_text(text: str) -> str:
    s = str(text or "")
    if not s:
        return "body"
    tokens = re.findall(r"\S+", s)
    if not tokens:
        return "body"
    tail_n = int(max(1, round(len(tokens) * 0.2)))
    tail_text = " ".join(tokens[-tail_n:])
    tail_digit_ratio = float(sum(ch.isdigit() for ch in tail_text) / max(1, len(tail_text)))
    marker_hits_tail = len(_CLOSING_MARKER_RE.findall(tail_text))
    score = 0.0
    if marker_hits_tail > 0:
        score += 1.0 + min(1.0, 0.4 * marker_hits_tail)
    if tail_digit_ratio >= 0.05:
        score += 1.0
    elif tail_digit_ratio >= 0.03:
        score += 0.5
    return "closing" if score >= 1.5 else "body"


def _source_profile(text: str) -> dict[str, float | str]:
    s = str(text or "")
    tokens = re.findall(r"\S+", s)
    n_chars = max(1, len(s))
    bracket_ratio = float(sum(1 for ch in s if ch in _BRACKET_CHARS) / n_chars)
    digit_ratio = float(sum(ch.isdigit() for ch in s) / n_chars)
    unknown_ratio = (
        float(sum(1 for tok in tokens if tok.lower() in _UNKNOWN_TOKENS) / len(tokens))
        if tokens
        else 0.0
    )
    repeat_rate = _repeat_trigram_rate([tok.lower() for tok in tokens])
    marker_density = float(len(_CLOSING_MARKER_RE.findall(s)) / max(1, len(tokens)))
    section = _section_from_text(s)
    return {
        "section": section,
        "bracket_ratio": bracket_ratio,
        "digit_ratio": digit_ratio,
        "unknown_ratio": unknown_ratio,
        "repeat_trigram_rate": repeat_rate,
        "marker_density": marker_density,
    }


def _ensure_profile_cache(model: Any) -> list[dict[str, float | str]]:
    cache = getattr(model, "_reranker_src_profiles", None)
    if cache is not None:
        return cache
    profiles = [_source_profile(txt) for txt in model.train_src_raw]
    setattr(model, "_reranker_src_profiles", profiles)
    return profiles


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
    out: set[str] = set()
    for tok in toks:
        if any(ch.isupper() for ch in tok):
            out.add(tok.lower())
    return out


def _pn_overlap(a: str, b: str) -> float:
    pa = _pn_tokens(a)
    pb = _pn_tokens(b)
    if not pa and not pb:
        return 0.0
    inter = len(pa.intersection(pb))
    union = len(pa.union(pb))
    return float(inter / union) if union else 0.0


@dataclass(frozen=True)
class LinearRerankerModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    model_id: str = "linear_logistic_reranker_v1"
    feature_transform: str = "identity"
    gate_prob_margin: float = 0.0
    gate_base_score_drop: float = -1.0
    gate_min_pred_delta: float = 0.0
    delta_calibration_slope: float = 1.0
    delta_calibration_intercept: float = 0.0
    gate_numeric_digit_overlap_tol: float = 0.0

    def score_matrix(self, x: np.ndarray) -> np.ndarray:
        mat = np.asarray(x, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {mat.shape}")
        if mat.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: got {mat.shape[1]}, expected {len(self.feature_names)}"
            )
        if self.feature_transform == "raw_z_rel_to_top_base":
            mu = mat.mean(axis=0, keepdims=True)
            sd = mat.std(axis=0, keepdims=True) + 1e-6
            z = (mat - mu) / sd
            top_idx = int(np.argmax(mat[:, FEATURE_NAMES.index("base_score")]))
            rel = mat - mat[top_idx : top_idx + 1, :]
            mat = np.concatenate([mat, z, rel], axis=1)
        elif self.feature_transform != "identity":
            raise ValueError(f"Unsupported feature_transform: {self.feature_transform}")

        if mat.shape[1] != len(self.weights):
            raise ValueError(
                f"Transformed feature dimension mismatch: got {mat.shape[1]}, expected {len(self.weights)}"
            )
        w = np.asarray(self.weights, dtype=np.float32)
        logits = mat @ w + float(self.intercept)
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def top_weight_items(self, top_n: int = 5) -> list[dict[str, float | str]]:
        top_n = int(max(1, top_n))
        ranked = sorted(
            zip(self.feature_names, self.weights),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_n]
        return [
            {
                "feature": str(name),
                "weight": float(weight),
                "abs_weight": float(abs(weight)),
            }
            for name, weight in ranked
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_id,
            "feature_names": list(self.feature_names),
            "weights": [float(w) for w in self.weights],
            "intercept": float(self.intercept),
            "feature_transform": str(self.feature_transform),
            "gate_prob_margin": float(self.gate_prob_margin),
            "gate_base_score_drop": float(self.gate_base_score_drop),
            "gate_min_pred_delta": float(self.gate_min_pred_delta),
            "delta_calibration_slope": float(self.delta_calibration_slope),
            "delta_calibration_intercept": float(self.delta_calibration_intercept),
            "gate_numeric_digit_overlap_tol": float(self.gate_numeric_digit_overlap_tol),
        }

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "LinearRerankerModel":
        if "feature_names" in payload and "weights" in payload and "intercept" in payload:
            names = tuple(str(x) for x in payload["feature_names"])
            weights = tuple(float(x) for x in payload["weights"])
            intercept = float(payload["intercept"])
            model_id = str(payload.get("model_type", "linear_logistic_reranker_v1"))
            feature_transform = str(payload.get("feature_transform", "identity"))
            gate_prob_margin = float(payload.get("gate_prob_margin", 0.0))
            gate_base_score_drop = float(payload.get("gate_base_score_drop", -1.0))
            gate_min_pred_delta = float(payload.get("gate_min_pred_delta", 0.0))
            delta_calibration_slope = float(payload.get("delta_calibration_slope", 1.0))
            delta_calibration_intercept = float(payload.get("delta_calibration_intercept", 0.0))
            gate_numeric_digit_overlap_tol = float(payload.get("gate_numeric_digit_overlap_tol", 0.0))
            return LinearRerankerModel(
                feature_names=names,
                weights=weights,
                intercept=intercept,
                model_id=model_id,
                feature_transform=feature_transform,
                gate_prob_margin=gate_prob_margin,
                gate_base_score_drop=gate_base_score_drop,
                gate_min_pred_delta=gate_min_pred_delta,
                delta_calibration_slope=delta_calibration_slope,
                delta_calibration_intercept=delta_calibration_intercept,
                gate_numeric_digit_overlap_tol=gate_numeric_digit_overlap_tol,
            )

        # Backward compatibility with evaluation JSON payloads.
        coeffs = payload.get("coefficients")
        if isinstance(coeffs, dict):
            names = tuple(str(x) for x in coeffs.get("feature_names", []))
            weights = tuple(float(x) for x in coeffs.get("weights", []))
            intercept = float(coeffs.get("intercept", 0.0))
            if names and weights and len(names) == len(weights):
                return LinearRerankerModel(
                    feature_names=names,
                    weights=weights,
                    intercept=intercept,
                    model_id="linear_logistic_reranker_v1",
                    feature_transform=str(payload.get("feature_transform", "identity")),
                    gate_prob_margin=float(payload.get("gate_prob_margin", 0.0)),
                    gate_base_score_drop=float(payload.get("gate_base_score_drop", -1.0)),
                    gate_min_pred_delta=float(payload.get("gate_min_pred_delta", 0.0)),
                    delta_calibration_slope=float(payload.get("delta_calibration_slope", 1.0)),
                    delta_calibration_intercept=float(payload.get("delta_calibration_intercept", 0.0)),
                    gate_numeric_digit_overlap_tol=float(payload.get("gate_numeric_digit_overlap_tol", 0.0)),
                )

        raise ValueError("Unsupported reranker payload format: missing coefficients")


def load_linear_reranker(path: Path) -> LinearRerankerModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reranker model JSON must be an object: {path}")
    model = LinearRerankerModel.from_payload(payload)
    if tuple(model.feature_names) != FEATURE_NAMES:
        raise ValueError(
            f"Unsupported feature set in reranker model {path}: {model.feature_names}; expected {FEATURE_NAMES}"
        )
    return model


def build_candidate_pool_top_internal_oracc(
    *,
    model: Any,
    row: np.ndarray,
    internal_top_k: int,
    oracc_cap: int,
    required_indices: tuple[int, ...] = (),
    exclude_indices: tuple[int, ...] = (),
) -> np.ndarray:
    exclude = {int(i) for i in exclude_indices}

    internal_fetch = int(max(1, internal_top_k + len(exclude) + 8))
    internal_cands = model._candidate_indices_from_subset(
        row,
        model.internal_indices,
        min(internal_fetch, len(model.internal_indices)),
    )
    internal_list: list[int] = []
    for idx in internal_cands.tolist():
        idx_i = int(idx)
        if idx_i in exclude:
            continue
        internal_list.append(idx_i)
        if len(internal_list) >= int(max(1, internal_top_k)):
            break

    external_list: list[int] = []
    if getattr(model, "external_indices", np.asarray([], dtype=np.int32)).size > 0 and int(oracc_cap) > 0:
        ext_fetch = int(max(1, oracc_cap + len(exclude) + 4))
        external_cands = model._candidate_indices_from_subset(
            row,
            model.external_indices,
            min(ext_fetch, len(model.external_indices)),
        )
        for idx in external_cands.tolist():
            idx_i = int(idx)
            if idx_i in exclude:
                continue
            external_list.append(idx_i)
            if len(external_list) >= int(max(1, oracc_cap)):
                break

    ordered: list[int] = []
    seen: set[int] = set()
    for idx in internal_list + external_list + [int(i) for i in required_indices]:
        idx_i = int(idx)
        if idx_i in exclude:
            continue
        if idx_i in seen:
            continue
        seen.add(idx_i)
        ordered.append(idx_i)

    if not ordered:
        fallback = model._candidate_indices(row, 1)
        ordered = [int(fallback[0])]

    return np.asarray(ordered, dtype=np.int32)


def build_feature_matrix_for_candidates(
    *,
    model: Any,
    query_text_raw: str,
    row: np.ndarray,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    q_norm = model._norm(query_text_raw)
    q_len = len(q_norm)
    q_domain = infer_dialog_domain_with_confidence(q_norm)[0]
    q_prof = _source_profile(query_text_raw)
    prof_cache = _ensure_profile_cache(model)
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

    features: list[list[float]] = []
    for rank, idx in enumerate(candidate_idx):
        idx_i = int(idx)
        d_prof = prof_cache[idx_i]
        len_ratio = float(model._len_score(q_len=q_len, d_len=int(model.train_lens[idx_i])))
        is_external = 1.0 if model.train_origins[idx_i] != "competition" else 0.0
        domain_match = 1.0 if model.train_domains[idx_i] == q_domain else 0.0
        digits_ov = _digits_overlap(query_text_raw, model.train_src_raw[idx_i])
        pn_ov = _pn_overlap(query_text_raw, model.train_src_raw[idx_i])
        rank_inv = 1.0 / float(rank + 1)
        section_match = 1.0 if str(d_prof["section"]) == str(q_prof["section"]) else 0.0
        digit_ratio_prox = 1.0 - min(
            1.0,
            abs(float(q_prof["digit_ratio"]) - float(d_prof["digit_ratio"])) * 12.0,
        )
        bracket_ratio_prox = 1.0 - min(
            1.0,
            abs(float(q_prof["bracket_ratio"]) - float(d_prof["bracket_ratio"])) * 16.0,
        )
        unknown_ratio_prox = 1.0 - min(
            1.0,
            abs(float(q_prof["unknown_ratio"]) - float(d_prof["unknown_ratio"])) * 10.0,
        )
        repeat_trigram_prox = 1.0 - min(
            1.0,
            abs(float(q_prof["repeat_trigram_rate"]) - float(d_prof["repeat_trigram_rate"])) * 8.0,
        )
        marker_density_prox = 1.0 - min(
            1.0,
            abs(float(q_prof["marker_density"]) - float(d_prof["marker_density"])) * 10.0,
        )
        features.append(
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
                section_match,
                digit_ratio_prox,
                bracket_ratio_prox,
                unknown_ratio_prox,
                repeat_trigram_prox,
                marker_density_prox,
            ]
        )

    if not features:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def feature_dict_from_row(values: np.ndarray) -> dict[str, float]:
    return {
        str(name): float(values[i])
        for i, name in enumerate(FEATURE_NAMES)
    }
