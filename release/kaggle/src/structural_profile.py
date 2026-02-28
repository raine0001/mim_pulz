from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np


TOKEN_RE = re.compile(r"\S+")
DIGIT_RE = re.compile(r"\d+")
ELLIPSIS_RE = re.compile(r"\.\.\.|\u2026")
UNKNOWN_TOKEN_RE = re.compile(r"^(x+|\?+|\.\.\.)$", re.IGNORECASE)
PN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{1,}")
PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or ""))


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def build_token_frequency(texts: list[str]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            key = tok.lower()
            freq[key] = freq.get(key, 0) + 1
    return freq


def extract_source_features(text: str, token_freq: dict[str, int], rare_freq_max: int = 2) -> dict[str, float]:
    s = str(text or "")
    toks = tokenize(s)
    n_tok = len(toks)
    n_chars = len(s)
    lines = s.splitlines() if s else [""]
    n_lines = max(1, len(lines))

    bracket_chars = sum(s.count(ch) for ch in "[](){}<>\u2e22\u2e23")
    bracket_ratio = _safe_ratio(bracket_chars, n_chars)
    ellipsis_count = len(ELLIPSIS_RE.findall(s))
    unknown_token_count = sum(1 for t in toks if UNKNOWN_TOKEN_RE.match(t) is not None)
    unknown_token_ratio = _safe_ratio(unknown_token_count, n_tok)

    # Formula density proxies
    tri = [tuple(toks[i : i + 3]) for i in range(max(0, n_tok - 2))]
    if tri:
        tri_unique = len(set(tri))
        repeat_trigram_rate = 1.0 - _safe_ratio(tri_unique, len(tri))
    else:
        repeat_trigram_rate = 0.0
    prefixes = []
    for ln in lines:
        p = tokenize(ln)[:3]
        if p:
            prefixes.append(tuple(p))
    if prefixes:
        pref_unique = len(set(prefixes))
        line_prefix_repeat_rate = 1.0 - _safe_ratio(pref_unique, len(prefixes))
    else:
        line_prefix_repeat_rate = 0.0

    digit_token_count = sum(1 for t in toks if DIGIT_RE.search(t) is not None)
    digit_chars = sum(len(m.group(0)) for m in DIGIT_RE.finditer(s))
    digit_ratio = _safe_ratio(digit_chars, n_chars)

    hyphen_token_count = sum(1 for t in toks if "-" in t)
    hyphen_token_ratio = _safe_ratio(hyphen_token_count, n_tok)
    rare_token_count = sum(1 for t in toks if token_freq.get(t.lower(), 0) <= rare_freq_max)
    rare_token_ratio = _safe_ratio(rare_token_count, n_tok)

    return {
        "src_chars": float(n_chars),
        "src_tokens": float(n_tok),
        "src_lines": float(n_lines),
        "bracket_ratio": bracket_ratio,
        "ellipsis_count": float(ellipsis_count),
        "unknown_token_ratio": unknown_token_ratio,
        "repeat_trigram_rate": repeat_trigram_rate,
        "line_prefix_repeat_rate": line_prefix_repeat_rate,
        "digit_ratio": digit_ratio,
        "digit_token_count": float(digit_token_count),
        "hyphen_token_ratio": hyphen_token_ratio,
        "rare_token_ratio": rare_token_ratio,
    }


def extract_target_features(text: str) -> dict[str, float]:
    s = str(text or "")
    toks = tokenize(s)
    n_tok = max(1, len(toks))
    n_chars = max(1, len(s))
    cap_tok = sum(1 for t in toks if t and t[0].isupper())
    punct_count = len(PUNCT_RE.findall(s))
    digit_chars = sum(len(m.group(0)) for m in DIGIT_RE.finditer(s))
    return {
        "tgt_capitalized_ratio": _safe_ratio(cap_tok, n_tok),
        "tgt_digit_ratio": _safe_ratio(digit_chars, n_chars),
        "tgt_punctuation_ratio": _safe_ratio(punct_count, n_chars),
    }


@dataclass(frozen=True)
class ProfileThresholds:
    fragment_bracket_p90: float
    fragment_unknown_p90: float
    formula_high_p85: float
    formula_mid_p60: float
    numeric_high_p85: float
    pn_high_p85: float

    @staticmethod
    def from_feature_rows(rows: list[dict[str, float]]) -> "ProfileThresholds":
        arr = lambda k: np.asarray([float(r.get(k, 0.0)) for r in rows], dtype=np.float32)
        return ProfileThresholds(
            fragment_bracket_p90=float(np.percentile(arr("bracket_ratio"), 90)),
            fragment_unknown_p90=float(np.percentile(arr("unknown_token_ratio"), 90)),
            formula_high_p85=float(np.percentile(arr("repeat_trigram_rate"), 85)),
            formula_mid_p60=float(np.percentile(arr("repeat_trigram_rate"), 60)),
            numeric_high_p85=float(np.percentile(arr("digit_ratio"), 85)),
            pn_high_p85=float(np.percentile(arr("rare_token_ratio"), 85)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "fragment_bracket_p90": self.fragment_bracket_p90,
            "fragment_unknown_p90": self.fragment_unknown_p90,
            "formula_high_p85": self.formula_high_p85,
            "formula_mid_p60": self.formula_mid_p60,
            "numeric_high_p85": self.numeric_high_p85,
            "pn_high_p85": self.pn_high_p85,
        }


def _conf_distance(value: float, threshold: float, scale_floor: float = 1e-4) -> float:
    scale = max(abs(threshold), scale_floor)
    z = abs(value - threshold) / scale
    # 0.5..1.0
    return float(0.5 + 0.5 * min(1.0, z))


def label_features(features: dict[str, float], thr: ProfileThresholds) -> dict[str, Any]:
    src_tokens = float(features.get("src_tokens", 0.0))
    bracket_ratio = float(features.get("bracket_ratio", 0.0))
    unk_ratio = float(features.get("unknown_token_ratio", 0.0))
    formula = float(features.get("repeat_trigram_rate", 0.0))
    line_rep = float(features.get("line_prefix_repeat_rate", 0.0))
    digit_ratio = float(features.get("digit_ratio", 0.0))
    rare_ratio = float(features.get("rare_token_ratio", 0.0))
    hyphen_ratio = float(features.get("hyphen_token_ratio", 0.0))
    src_lines = int(features.get("src_lines", 1.0))

    if src_tokens <= 80:
        length_bucket = "short"
    elif src_tokens <= 200:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    frag_score = max(
        _safe_ratio(bracket_ratio, max(thr.fragment_bracket_p90, 1e-6)),
        _safe_ratio(unk_ratio, max(thr.fragment_unknown_p90, 1e-6)),
    )
    if frag_score >= 1.0:
        fragmentation = "fragmentary"
    elif frag_score >= 0.45:
        fragmentation = "partial"
    else:
        fragmentation = "complete"

    if formula >= thr.formula_high_p85:
        formula_density = "high"
    elif formula >= thr.formula_mid_p60:
        formula_density = "medium"
    else:
        formula_density = "low"

    numeric_density = "high" if digit_ratio >= thr.numeric_high_p85 else "low"
    pn_score = 0.65 * rare_ratio + 0.35 * hyphen_ratio
    proper_name_density = "high" if pn_score >= thr.pn_high_p85 else "low"

    slot_like = (numeric_density == "high") or (formula_density == "high")
    if slot_like and length_bucket != "long":
        template_type = "slot_structured"
    elif (formula_density == "low") and (length_bucket == "long"):
        template_type = "narrative"
    else:
        template_type = "hybrid"

    if src_lines <= 1:
        layout_hint = "single_column"
    elif src_lines >= 3 and line_rep >= 0.2:
        layout_hint = "multi_column"
    else:
        layout_hint = "unknown"

    role_hint = "unknown"

    conf = {
        "length_bucket": 0.9,
        "fragmentation": _conf_distance(max(bracket_ratio, unk_ratio), max(thr.fragment_bracket_p90, thr.fragment_unknown_p90)),
        "formula_density": _conf_distance(formula, thr.formula_high_p85),
        "numeric_density": _conf_distance(digit_ratio, thr.numeric_high_p85),
        "proper_name_density": _conf_distance(pn_score, thr.pn_high_p85),
        "template_type": 0.75,
        "layout_hint": 0.7,
        "role_hint": 0.6,
    }
    overall_conf = float(np.mean(np.asarray(list(conf.values()), dtype=np.float32)))

    return {
        "length_bucket": length_bucket,
        "fragmentation": fragmentation,
        "formula_density": formula_density,
        "numeric_density": numeric_density,
        "proper_name_density": proper_name_density,
        "template_type": template_type,
        "layout_hint": layout_hint,
        "role_hint": role_hint,
        "confidence": conf,
        "overall_confidence": overall_conf,
    }


def top_flags(features: dict[str, float], labels: dict[str, Any]) -> list[str]:
    out = []
    if labels.get("fragmentation") == "fragmentary":
        out.append("high_fragmentation")
    if labels.get("numeric_density") == "high":
        out.append("high_numeric_density")
    if labels.get("formula_density") == "high":
        out.append("high_formula_density")
    if labels.get("template_type") == "slot_structured":
        out.append("slot_structured")
    if labels.get("length_bucket") == "long":
        out.append("long_text")
    if labels.get("proper_name_density") == "high":
        out.append("high_proper_name_density")
    return out


def default_routing_map(internal_score_stats: dict[str, float]) -> dict[str, Any]:
    t_low = float(internal_score_stats.get("p25", 0.12))
    t_high = float(internal_score_stats.get("p75", 0.20))
    return {
        "thresholds": {
            "internal_top_low": t_low,
            "internal_top_high": t_high,
        },
        "routes": {
            "P0": {"route": "RETRIEVE_INTERNAL", "policy_params": {"top_k": 120}},
            "P1": {"route": "RETRIEVE_HYBRID", "policy_params": {"oracc_cap": 25}},
            "P2": {"route": "RETRIEVE_ORACC_FALLBACK", "policy_params": {"oracc_cap": 10, "gate": "low_conf"}},
            "P3": {"route": "RERANK_STRONG", "policy_params": {"stage2_pool": 120, "stage2_weight": 0.35}},
        },
    }


def route_decision(
    labels: dict[str, Any],
    *,
    internal_top_score: float | None,
    internal_gap: float | None,
    routing_map: dict[str, Any],
) -> dict[str, Any]:
    t_low = float(routing_map["thresholds"]["internal_top_low"])
    t_high = float(routing_map["thresholds"]["internal_top_high"])

    rationale = []

    # Confidence gating layer.
    if internal_top_score is not None and internal_top_score >= t_high:
        rationale.append("internal_top_score>=T_high")
        return {
            "route": "RETRIEVE_INTERNAL",
            "policy_params": routing_map["routes"]["P0"]["policy_params"],
            "rationale": "; ".join(rationale),
        }
    if internal_top_score is not None and internal_top_score < t_low:
        rationale.append("internal_top_score<T_low")
        return {
            "route": "RETRIEVE_HYBRID",
            "policy_params": routing_map["routes"]["P1"]["policy_params"],
            "rationale": "; ".join(rationale),
        }

    # Structural routing.
    if labels.get("template_type") == "slot_structured" and labels.get("numeric_density") == "high":
        rationale.append("slot_structured+high_numeric")
        return {
            "route": "RERANK_STRONG",
            "policy_params": routing_map["routes"]["P3"]["policy_params"],
            "rationale": "; ".join(rationale),
        }
    if labels.get("fragmentation") == "fragmentary" or labels.get("length_bucket") == "long":
        rationale.append("fragmentary_or_long")
        return {
            "route": "RETRIEVE_HYBRID",
            "policy_params": routing_map["routes"]["P1"]["policy_params"],
            "rationale": "; ".join(rationale),
        }
    if (
        labels.get("template_type") == "narrative"
        and labels.get("formula_density") == "low"
        and labels.get("length_bucket") == "long"
    ):
        rationale.append("narrative_low_formula_long")
        return {
            "route": "RETRIEVE_ORACC_FALLBACK",
            "policy_params": routing_map["routes"]["P2"]["policy_params"],
            "rationale": "; ".join(rationale),
        }
    if labels.get("formula_density") == "high" and labels.get("length_bucket") == "short":
        rationale.append("high_formula_short")
        return {
            "route": "RETRIEVE_INTERNAL",
            "policy_params": routing_map["routes"]["P0"]["policy_params"],
            "rationale": "; ".join(rationale),
        }

    if internal_gap is not None and internal_gap < 0.02:
        rationale.append("ambiguous_internal_gap")
        return {
            "route": "RETRIEVE_ORACC_FALLBACK",
            "policy_params": routing_map["routes"]["P2"]["policy_params"],
            "rationale": "; ".join(rationale),
        }

    rationale.append("default_internal")
    return {
        "route": "RETRIEVE_INTERNAL",
        "policy_params": routing_map["routes"]["P0"]["policy_params"],
        "rationale": "; ".join(rationale),
    }


