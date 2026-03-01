from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.data import load_deep_past_competition
from mim_pulz.domain_intent import infer_dialog_domain_with_confidence
from mim_pulz.routed_reranker import (
    FEATURE_NAMES,
    build_candidate_pool_top_internal_oracc,
    build_feature_matrix_for_candidates,
    feature_dict_from_row,
    load_linear_reranker,
)


def _normalize_text(text: str, *, strip_punct: bool, lowercase: bool, collapse_whitespace: bool) -> str:
    s = str(text or "")
    if strip_punct:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if lowercase:
        s = s.lower()
    if collapse_whitespace:
        s = " ".join(s.split())
    return s


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return float(inter / union) if union else 0.0


def _short(text: str, max_chars: int = 320) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _canonical_context_label(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return "unknown"
    if re.search(r"\b(letter|epist|message|correspond)\b", s):
        return "letter"
    if re.search(r"\b(legal|contract|lawsuit|witness)\b", s):
        return "legal"
    if re.search(r"\b(economic|account|ration|receipt|loan|silver|tin)\b", s):
        return "economic"
    if re.search(r"\b(ritual|prayer|incant|hymn|omen)\b", s):
        return "ritual"
    if re.search(r"\b(admin|administrative|official|governor)\b", s):
        return "administrative"
    if re.search(r"\b(scholarly|lexical|commentary)\b", s):
        return "scholarly"
    if s in {
        "letter",
        "legal",
        "economic",
        "ritual",
        "administrative",
        "scholarly",
        "unknown",
    }:
        return s
    return "unknown"


_CLOSING_MARKER_RE = re.compile(
    r"\b("
    r"igi|witness(?:es)?|seal(?:ed)?|colophon|date(?:d)?|month|year|day|"
    r"limmu|itu|mu|ki[šs]ib|kunuk|total|sum|subtotal|balance|"
    r"šu\.?nigin2?|nigin2?|debet|credit|obv|rev|edge"
    r")\b",
    flags=re.IGNORECASE,
)

_PROFILE_TOKEN_RE = re.compile(r"\S+")
_PROFILE_UNKNOWN_TOKEN_RE = re.compile(r"^(x+|\?+|\.{2,}|\[\.\.\.\])$", flags=re.IGNORECASE)
_DIGIT_TOKEN_RE = re.compile(r"\d+")
_SKELETON_BRACKET_SPAN_RE = re.compile(r"\[[^\]]*\]")
_SLOT_TOKEN_RE = re.compile(r"[0-9A-Za-z_?.\-]+")

_MEASURE_TOKENS = {
    "gin",
    "gin2",
    "gín",
    "ma-na",
    "mana",
    "mina",
    "minas",
    "shekel",
    "shekels",
    "talent",
    "talents",
    "tug",
    "gur",
    "qa",
    "sila",
}

_FORMULA_ANCHORS = {
    "um-ma",
    "qibi-ma",
    "igi",
    "kunuk",
    "kišib",
    "kilib",
    "seal",
    "witness",
    "date",
    "month",
    "year",
    "total",
    "sum",
    "obv",
    "rev",
}

_VARIANT_SKIP_TOKENS = {"<unk>", "<num>", "<measure>", "<formula>"}


def _infer_section_type(text: str, *, tail_ratio: float = 0.15) -> tuple[str, dict[str, float]]:
    raw = str(text or "").strip()
    if not raw:
        return "body", {"score": 0.0, "digit_ratio": 0.0, "tail_digit_ratio": 0.0, "tail_markers": 0.0}

    lines = [ln.strip() for ln in re.split(r"[\r\n]+", raw) if ln.strip()]
    if not lines:
        lines = [raw]
    full_text = " ".join(lines)
    tokens = re.findall(r"\S+", full_text)
    if not tokens:
        return "body", {"score": 0.0, "digit_ratio": 0.0, "tail_digit_ratio": 0.0, "tail_markers": 0.0}

    tail_ratio = float(min(0.5, max(0.05, tail_ratio)))
    tail_tok_n = int(max(1, round(len(tokens) * tail_ratio)))
    tail_tokens = tokens[-tail_tok_n:]
    tail_text = " ".join(tail_tokens)
    tail_line_n = int(max(1, round(len(lines) * tail_ratio)))
    tail_lines_text = " ".join(lines[-tail_line_n:])

    digit_ratio = float(sum(ch.isdigit() for ch in full_text) / max(1, len(full_text)))
    tail_digit_ratio = float(sum(ch.isdigit() for ch in tail_text) / max(1, len(tail_text)))
    marker_hits_whole = int(len(_CLOSING_MARKER_RE.findall(full_text)))
    marker_hits_tail = int(len(_CLOSING_MARKER_RE.findall(tail_lines_text)))

    score = 0.0
    if marker_hits_tail > 0:
        score += 1.0 + min(1.0, 0.4 * marker_hits_tail)
    if marker_hits_whole >= 2:
        score += 0.5
    if tail_digit_ratio >= 0.05:
        score += 1.0
    elif tail_digit_ratio >= 0.03:
        score += 0.5
    if digit_ratio >= 0.04:
        score += 0.5

    label = "closing" if score >= 1.5 else "body"
    meta = {
        "score": float(score),
        "digit_ratio": float(digit_ratio),
        "tail_digit_ratio": float(tail_digit_ratio),
        "tail_markers": float(marker_hits_tail),
    }
    return label, meta


def _set_overlap(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / union) if union else 0.0


def _tokenize_slot(text: str) -> list[str]:
    return [t for t in _SLOT_TOKEN_RE.findall(str(text or "")) if t]


def _token_to_slot(tok: str) -> str:
    t = str(tok or "").strip().lower()
    if not t:
        return "OTHER"
    if _PROFILE_UNKNOWN_TOKEN_RE.match(t) is not None or "[" in t or "]" in t:
        return "OTHER"
    if _DIGIT_TOKEN_RE.search(t) is not None:
        return "NUM"
    if t in _MEASURE_TOKENS:
        return "MEASURE"
    if t in _FORMULA_ANCHORS:
        return "FORMULA"
    if "-" in t and len(t) >= 4 and not t.startswith("<"):
        return "PN"
    return "OTHER"


def _slot_signature_text(text: str) -> str:
    toks = _tokenize_slot(text)
    if not toks:
        return ""
    tags = [_token_to_slot(tok) for tok in toks]
    return " ".join(tags)


def _slot_similarity(sig_a: str, sig_b: str) -> float:
    if not sig_a and not sig_b:
        return 0.0
    if not sig_a or not sig_b:
        return 0.0
    return float(SequenceMatcher(None, sig_a, sig_b).ratio())


def _extract_digit_tokens(text: str) -> set[str]:
    return {m.group(0) for m in _DIGIT_TOKEN_RE.finditer(str(text or ""))}


def _extract_formula_tokens(text: str) -> set[str]:
    low = str(text or "").lower()
    toks = set(_PROFILE_TOKEN_RE.findall(low))
    out = {tok for tok in toks if tok in _FORMULA_ANCHORS}
    for marker in _FORMULA_ANCHORS:
        if marker in low:
            out.add(marker)
    return out


def _skeletonize_text(text: str) -> str:
    s = str(text or "")
    s = _SKELETON_BRACKET_SPAN_RE.sub(" <UNK> ", s)
    out_toks: list[str] = []
    last_was_unk = False
    for tok in _PROFILE_TOKEN_RE.findall(s):
        low = tok.lower()
        if _PROFILE_UNKNOWN_TOKEN_RE.match(low) is not None or low in {"[", "]"}:
            mapped = "<UNK>"
        elif _DIGIT_TOKEN_RE.search(low) is not None:
            mapped = "<NUM>"
        elif low in _MEASURE_TOKENS:
            mapped = "<MEASURE>"
        elif low in _FORMULA_ANCHORS:
            mapped = "<FORMULA>"
        else:
            mapped = low
        if mapped == "<UNK>":
            if last_was_unk:
                continue
            last_was_unk = True
        else:
            last_was_unk = False
        out_toks.append(mapped)
    return " ".join(out_toks)


def _normalize_variant_token(tok: str) -> str:
    t = str(tok or "").strip().lower()
    if not t:
        return ""
    if t in _VARIANT_SKIP_TOKENS:
        return ""
    if _PROFILE_UNKNOWN_TOKEN_RE.match(t) is not None:
        return ""
    if _DIGIT_TOKEN_RE.search(t) is not None:
        return ""
    if t.startswith("<") and t.endswith(">"):
        return ""
    if len(t) < 2:
        return ""
    return t


def _variant_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _SLOT_TOKEN_RE.findall(str(text or "")):
        norm = _normalize_variant_token(tok)
        if norm:
            out.append(norm)
    return out


def _collect_context_equivalence_counts(tokens_a: list[str], tokens_b: list[str], counts: dict[tuple[str, str], int]) -> None:
    n = min(len(tokens_a), len(tokens_b))
    if n < 3:
        return
    for i in range(1, n - 1):
        if tokens_a[i - 1] != tokens_b[i - 1] or tokens_a[i + 1] != tokens_b[i + 1]:
            continue
        a = tokens_a[i]
        b = tokens_b[i]
        if a == b:
            continue
        counts[(a, b)] = counts.get((a, b), 0) + 1
        counts[(b, a)] = counts.get((b, a), 0) + 1


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported external memory file format: {path}")


def _resolve_col(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    return None


def _load_external_memory(config: "RetrievalConfig") -> tuple[list[str], list[str], list[str], list[str], list[dict]]:
    if not config.external_memory_paths:
        return [], [], [], [], []

    src_all: list[str] = []
    tgt_all: list[str] = []
    ctx_all: list[str] = []
    origin_all: list[str] = []
    stats: list[dict] = []
    total_limit = int(config.external_memory_limit)
    allow_ctx = {_canonical_context_label(x) for x in config.external_context_allowlist if str(x).strip()}

    for raw_path in config.external_memory_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"External retrieval memory path does not exist: {path}")

        frame = _read_table(path)
        src_col = _resolve_col(
            frame,
            preferred=config.external_source_col,
            fallbacks=["source", "src", "transliteration", "input", "text"],
        )
        tgt_col = _resolve_col(
            frame,
            preferred=config.external_target_col,
            fallbacks=["target", "tgt", "translation", "output"],
        )
        if src_col is None or tgt_col is None:
            raise ValueError(
                f"External memory {path} is missing source/target columns. "
                f"Available columns: {list(frame.columns)}"
            )
        ctx_col = _resolve_col(
            frame,
            preferred=config.external_context_col,
            fallbacks=["context", "domain", "genre", "text_type"],
        )
        origin_col = _resolve_col(
            frame,
            preferred=config.external_origin_col,
            fallbacks=["origin", "source_name", "dataset"],
        )

        frame = frame.copy()
        frame[src_col] = frame[src_col].astype(str).fillna("")
        frame[tgt_col] = frame[tgt_col].astype(str).fillna("")

        kept = 0
        dropped_empty = 0
        dropped_context = 0
        for i, row in frame.iterrows():
            s = str(row[src_col]).strip()
            t = str(row[tgt_col]).strip()
            if not s or not t:
                dropped_empty += 1
                continue
            if ctx_col is not None:
                c = _canonical_context_label(row.get(ctx_col, ""))
            else:
                c = _canonical_context_label(infer_dialog_domain_with_confidence(s)[0])
            if allow_ctx and c not in allow_ctx:
                dropped_context += 1
                continue
            if origin_col is not None:
                o = str(row.get(origin_col, "") or path.stem)
            else:
                o = path.stem
            src_all.append(s)
            tgt_all.append(t)
            ctx_all.append(c)
            origin_all.append(o)
            kept += 1

            if total_limit > 0 and len(src_all) >= total_limit:
                break
        stats.append(
            {
                "path": str(path.resolve()),
                "rows_total": int(len(frame)),
                "rows_kept": int(kept),
                "rows_dropped_empty": int(dropped_empty),
                "rows_dropped_context": int(dropped_context),
                "source_col": src_col,
                "target_col": tgt_col,
                "context_col": ctx_col,
                "origin_col": origin_col,
                "context_allowlist": sorted(list(allow_ctx)),
            }
        )
        if total_limit > 0 and len(src_all) >= total_limit:
            break

    return src_all, tgt_all, ctx_all, origin_all, stats


@dataclass(frozen=True)
class RetrievalConfig:
    analyzer: str = "char_wb"
    ngram_min: int = 3
    ngram_max: int = 5
    max_features: int = 300_000
    top_k: int = 120
    len_weight: float = 0.4
    len_mode: str = "ratio"

    # normalization controls
    strip_punct: bool = False
    lowercase: bool = False
    collapse_whitespace: bool = True

    # stage-2 reranker
    stage2_type: str = "bm25"  # none|bm25|seq_ratio|token_overlap
    stage2_pool: int = 80
    stage2_weight: float = 0.2
    stage2_bm25_k1: float = 1.2
    stage2_bm25_b: float = 0.5

    # confidence-gated context override
    enable_domain_override: bool = True
    domain_candidate_top_k: int = 120
    domain_bonus: float = 0.02
    domain_conf_threshold: float = 0.45
    domain_margin: float = 0.02

    # border-aware section constraints (MVP)
    enable_section_border: bool = False
    section_force_closing: bool = True
    section_force_min_score: float = 2.4
    section_closing_tail_ratio: float = 0.15
    section_min_pool: int = 10
    section_match_bonus: float = 0.04

    # text confidence adaptation (scribal noise / reconstruction-aware)
    enable_uncertainty_adaptation: bool = False
    uncertainty_high_threshold: float = 0.03
    uncertainty_closing_threshold: float = 0.02
    uncertainty_bracket_percentile: float = 90.0
    uncertainty_internal_top_threshold: float = 0.60
    uncertainty_internal_gap_threshold: float = 0.02
    uncertainty_topk_add: int = 20
    uncertainty_external_cap_add: int = 2
    uncertainty_topk_boost: float = 0.0
    uncertainty_external_bonus: float = 0.0
    uncertainty_stage2_discount: float = 0.0
    uncertainty_len_discount: float = 0.0
    uncertainty_triggered_len_ratio_min: float = 0.0
    uncertainty_skeleton_blend: float = 0.05
    uncertainty_number_bonus: float = 0.005
    uncertainty_formula_bonus: float = 0.005
    uncertainty_slot_bonus: float = 0.0
    uncertainty_variant_bonus: float = 0.0
    uncertainty_candidate_uncertain_penalty: float = 0.0
    uncertainty_rare_cutoff: int = 2
    enable_variant_equivalence: bool = True
    variant_neighbor_k: int = 3
    variant_similarity_threshold: float = 0.72
    variant_min_support: int = 2
    variant_max_pairs: int = 256

    # inference policy hint (set by routed builder)
    policy_name: str = ""

    # uncertainty-triggered skeleton retrieval (candidate union)
    enable_skeleton_retrieval_path: bool = True
    skeleton_candidate_top_k: int = 120

    # external memory
    external_memory_paths: tuple[str, ...] = ()
    external_source_col: str = "source"
    external_target_col: str = "target"
    external_context_col: str = "context"
    external_origin_col: str = "origin"
    external_memory_limit: int = 0
    external_context_allowlist: tuple[str, ...] = ()
    external_candidate_cap: int = 25
    external_enable_fallback: bool = False
    external_internal_top_threshold: float = -1.0
    external_internal_gap_threshold: float = -1.0
    external_force_contexts: tuple[str, ...] = ()
    external_gate_bonus: float = 0.0
    external_gate_margin: float = 0.0

    # origin calibration (useful when external memory is much larger)
    competition_origin_bonus: float = 0.0
    external_origin_bonus: float = 0.0

    # evidence reporting
    evidence_k: int = 3

    def model_id(self) -> str:
        base = (
            "CanonicalRetrievalV2("
            f"analyzer={self.analyzer},"
            f"ngram=({self.ngram_min},{self.ngram_max}),"
            f"top_k={self.top_k},"
            f"len_weight={self.len_weight},"
            f"len_mode={self.len_mode},"
            f"norm(strip_punct={self.strip_punct},lower={self.lowercase},collapse_ws={self.collapse_whitespace}),"
            f"stage2(type={self.stage2_type},pool={self.stage2_pool},weight={self.stage2_weight},"
            f"k1={self.stage2_bm25_k1},b={self.stage2_bm25_b}),"
            f"domain_override={self.enable_domain_override},"
            f"domain_candidate_top_k={self.domain_candidate_top_k},"
            f"domain_bonus={self.domain_bonus},"
            f"domain_conf_threshold={self.domain_conf_threshold},"
            f"domain_margin={self.domain_margin})"
        )
        has_external_overrides = (
            self.external_memory_paths
            or self.external_memory_limit > 0
            or self.external_context_allowlist
            or self.external_candidate_cap != 25
            or self.external_enable_fallback
            or self.external_internal_top_threshold >= 0.0
            or self.external_internal_gap_threshold >= 0.0
            or self.external_force_contexts
            or self.external_gate_bonus != 0.0
            or self.external_gate_margin != 0.0
            or self.evidence_k != 3
            or self.competition_origin_bonus != 0.0
            or self.external_origin_bonus != 0.0
        )
        has_section_overrides = (
            self.enable_section_border
            or not self.section_force_closing
            or self.section_force_min_score != 2.4
            or self.section_closing_tail_ratio != 0.15
            or self.section_min_pool != 10
            or self.section_match_bonus != 0.04
        )
        has_uncertainty_overrides = (
            self.enable_uncertainty_adaptation
            or self.uncertainty_high_threshold != 0.03
            or self.uncertainty_closing_threshold != 0.02
            or self.uncertainty_bracket_percentile != 90.0
            or self.uncertainty_internal_top_threshold != 0.60
            or self.uncertainty_internal_gap_threshold != 0.02
            or self.uncertainty_topk_add != 20
            or self.uncertainty_external_cap_add != 2
            or self.uncertainty_topk_boost != 0.0
            or self.uncertainty_external_bonus != 0.0
            or self.uncertainty_stage2_discount != 0.0
            or self.uncertainty_len_discount != 0.0
            or self.uncertainty_triggered_len_ratio_min != 0.0
            or self.uncertainty_skeleton_blend != 0.05
            or self.uncertainty_number_bonus != 0.005
            or self.uncertainty_formula_bonus != 0.005
            or self.uncertainty_slot_bonus != 0.0
            or self.uncertainty_variant_bonus != 0.0
            or self.uncertainty_candidate_uncertain_penalty != 0.0
            or not self.enable_variant_equivalence
            or self.variant_neighbor_k != 3
            or self.variant_similarity_threshold != 0.72
            or self.variant_min_support != 2
            or self.variant_max_pairs != 256
            or not self.enable_skeleton_retrieval_path
            or self.skeleton_candidate_top_k != 120
            or self.uncertainty_rare_cutoff != 2
        )
        parts: list[str] = []
        if has_external_overrides:
            parts.extend(
                [
                    f"ext_mem_paths={len(self.external_memory_paths)}",
                    f"ext_mem_limit={self.external_memory_limit}",
                    f"ext_mem_ctx_allow={len(self.external_context_allowlist)}",
                    (
                        "ext_gate("
                        f"enabled={self.external_enable_fallback},"
                        f"cap={self.external_candidate_cap},"
                        f"top_thr={self.external_internal_top_threshold},"
                        f"gap_thr={self.external_internal_gap_threshold},"
                        f"force_ctx={len(self.external_force_contexts)},"
                        f"bonus={self.external_gate_bonus},"
                        f"margin={self.external_gate_margin})"
                    ),
                    f"origin_bonus(comp={self.competition_origin_bonus},ext={self.external_origin_bonus})",
                    f"evidence_k={self.evidence_k}",
                ]
            )
        if has_section_overrides:
            parts.append(
                "section_border("
                f"enabled={self.enable_section_border},"
                f"force_closing={self.section_force_closing},"
                f"force_min_score={self.section_force_min_score},"
                f"tail={self.section_closing_tail_ratio},"
                f"min_pool={self.section_min_pool},"
                f"bonus={self.section_match_bonus})"
            )
        if has_uncertainty_overrides:
            parts.append(
                "uncertainty("
                f"enabled={self.enable_uncertainty_adaptation},"
                f"thr={self.uncertainty_high_threshold},"
                f"closing_thr={self.uncertainty_closing_threshold},"
                f"br_pct={self.uncertainty_bracket_percentile},"
                f"itop_thr={self.uncertainty_internal_top_threshold},"
                f"igap_thr={self.uncertainty_internal_gap_threshold},"
                f"topk_add={self.uncertainty_topk_add},"
                f"ext_add={self.uncertainty_external_cap_add},"
                f"kboost={self.uncertainty_topk_boost},"
                f"ext_bonus={self.uncertainty_external_bonus},"
                f"stg2_disc={self.uncertainty_stage2_discount},"
                f"len_disc={self.uncertainty_len_discount},"
                f"len_min={self.uncertainty_triggered_len_ratio_min},"
                f"skel_blend={self.uncertainty_skeleton_blend},"
                f"num_bonus={self.uncertainty_number_bonus},"
                f"form_bonus={self.uncertainty_formula_bonus},"
                f"slot_bonus={self.uncertainty_slot_bonus},"
                f"var_bonus={self.uncertainty_variant_bonus},"
                f"unc_pen={self.uncertainty_candidate_uncertain_penalty},"
                f"var_eq={self.enable_variant_equivalence},"
                f"var_k={self.variant_neighbor_k},"
                f"var_thr={self.variant_similarity_threshold},"
                f"var_min_sup={self.variant_min_support},"
                f"var_max={self.variant_max_pairs},"
                f"skel_path={self.enable_skeleton_retrieval_path},"
                f"skel_k={self.skeleton_candidate_top_k},"
                f"rare_cut={self.uncertainty_rare_cutoff})"
            )
        if self.policy_name:
            parts.append(f"policy={self.policy_name}")
        if parts:
            return base[:-1] + "," + ",".join(parts) + ")"
        return base


CANONICAL_BASELINE_V2 = RetrievalConfig()


def _config_with_overrides(config: RetrievalConfig, **overrides: Any) -> RetrievalConfig:
    payload = asdict(config)
    payload.update(overrides)
    for key in ("external_memory_paths", "external_context_allowlist", "external_force_contexts"):
        value = payload.get(key)
        if isinstance(value, list):
            payload[key] = tuple(value)
    return RetrievalConfig(**payload)


def build_policy_config(
    base_config: RetrievalConfig,
    *,
    policy_name: str,
    policy_params: dict[str, Any] | None = None,
    routing_thresholds: dict[str, Any] | None = None,
) -> RetrievalConfig:
    params = dict(policy_params or {})
    thresholds = dict(routing_thresholds or {})
    if policy_name == "internal_only":
        overrides: dict[str, Any] = {
            "external_memory_paths": (),
            "external_memory_limit": 0,
            "external_context_allowlist": (),
            "external_enable_fallback": False,
            "external_force_contexts": (),
            "policy_name": "internal_only",
        }
        if "top_k" in params:
            overrides["top_k"] = int(params["top_k"])
        return _config_with_overrides(base_config, **overrides)

    if policy_name == "hybrid":
        overrides = {"external_enable_fallback": False, "policy_name": "hybrid"}
        if "top_k" in params:
            overrides["top_k"] = int(params["top_k"])
        if "oracc_cap" in params:
            overrides["external_candidate_cap"] = int(params["oracc_cap"])
        return _config_with_overrides(base_config, **overrides)

    if policy_name == "fallback":
        overrides = {"external_enable_fallback": True, "policy_name": "fallback"}
        if "top_k" in params:
            overrides["top_k"] = int(params["top_k"])
        if "oracc_cap" in params:
            overrides["external_candidate_cap"] = int(params["oracc_cap"])
        gate = str(params.get("gate", "")).strip().lower()
        if gate == "low_conf":
            low = float(thresholds.get("internal_top_low", -1.0))
            if low >= 0.0:
                overrides["external_internal_top_threshold"] = low
            elif float(base_config.external_internal_top_threshold) < 0.0:
                overrides["external_internal_top_threshold"] = 0.14
            if float(base_config.external_internal_gap_threshold) < 0.0:
                overrides["external_internal_gap_threshold"] = -1.0
        return _config_with_overrides(base_config, **overrides)

    if policy_name == "strong_rerank":
        overrides = {
            "external_memory_paths": (),
            "external_memory_limit": 0,
            "external_context_allowlist": (),
            "external_enable_fallback": False,
            "external_force_contexts": (),
            "policy_name": "strong_rerank",
            "stage2_pool": int(params.get("stage2_pool", max(int(base_config.stage2_pool), 120))),
            "stage2_weight": float(params.get("stage2_weight", max(float(base_config.stage2_weight), 0.35))),
        }
        if "top_k" in params:
            overrides["top_k"] = int(params["top_k"])
        return _config_with_overrides(base_config, **overrides)

    raise ValueError(f"Unknown policy_name: {policy_name}")


class _CharBM25:
    def __init__(self, ngram_range: tuple[int, int], k1: float, b: float):
        self.ngram_range = ngram_range
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1)
        self.doc_term_csr: sparse.csr_matrix | None = None
        self.doc_term_csc: sparse.csc_matrix | None = None
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0

    def fit(self, docs: list[str]) -> None:
        x = self.vectorizer.fit_transform(docs).astype(np.float32).tocsr()
        self.doc_term_csr = x
        self.doc_term_csc = x.tocsc()
        n_docs = x.shape[0]
        df = np.diff(self.doc_term_csc.indptr).astype(np.float32)
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        self.doc_len = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if n_docs else 0.0

    def score_one(self, query: str) -> np.ndarray:
        if self.doc_term_csr is None or self.doc_term_csc is None or self.idf is None or self.doc_len is None:
            raise RuntimeError("BM25 index is not fitted")
        qv = self.vectorizer.transform([query]).tocsr()
        scores = np.zeros(self.doc_term_csr.shape[0], dtype=np.float32)
        if qv.nnz == 0:
            return scores
        for term, qtf in zip(qv.indices, qv.data):
            start = self.doc_term_csc.indptr[term]
            end = self.doc_term_csc.indptr[term + 1]
            if start == end:
                continue
            docs = self.doc_term_csc.indices[start:end]
            tfs = self.doc_term_csc.data[start:end]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * (self.doc_len[docs] / max(self.avgdl, 1e-6)))
            term_scores = self.idf[term] * (tfs * (self.k1 + 1.0) / denom)
            q_weight = 1.0 + np.log1p(float(qtf))
            scores[docs] += term_scores * q_weight
        return scores


class CanonicalRetrievalTranslator:
    def __init__(self, config: RetrievalConfig = CANONICAL_BASELINE_V2):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=(config.ngram_min, config.ngram_max),
            min_df=1,
            max_features=config.max_features,
        )
        self.skeleton_vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=(config.ngram_min, config.ngram_max),
            min_df=1,
            max_features=config.max_features,
        )
        self.train_matrix = None
        self.train_skeleton_matrix = None
        self.train_src_raw: list[str] = []
        self.train_src: list[str] = []
        self.train_skeleton: list[str] = []
        self.train_tgt: list[str] = []
        self.train_lens: np.ndarray | None = None
        self.train_domains: list[str] = []
        self.train_sections: list[str] = []
        self.train_origins: list[str] = []
        self.internal_indices: np.ndarray = np.asarray([], dtype=np.int32)
        self.external_indices: np.ndarray = np.asarray([], dtype=np.int32)
        self.stage2_bm25: _CharBM25 | None = None
        self.external_memory_stats: list[dict] = []
        self.external_rows_loaded: int = 0
        self.train_token_freq: dict[str, int] = {}
        self.train_bracket_ratio_p80: float = 0.0
        self.train_bracket_ratio_p85: float = 0.0
        self.train_bracket_ratio_p90: float = 0.0
        self.train_unknown_ratio_p90: float = 0.0
        self.train_uncertainty_ratio: np.ndarray | None = None
        self.train_digit_sets: list[set[str]] = []
        self.train_formula_sets: list[set[str]] = []
        self.train_slot_signatures: list[str] = []
        self.train_variant_token_sets: list[set[str]] = []
        self.variant_equiv_map: dict[str, set[str]] = {}
        self.variant_equiv_pairs: list[tuple[str, str, int]] = []

    def _norm(self, text: str) -> str:
        return _normalize_text(
            text,
            strip_punct=self.config.strip_punct,
            lowercase=self.config.lowercase,
            collapse_whitespace=self.config.collapse_whitespace,
        )

    def fit(
        self,
        train_texts: list[str] | None = None,
        train_targets: list[str] | None = None,
        train_src: list[str] | None = None,
        train_tgt: list[str] | None = None,
    ) -> None:
        src = train_texts if train_texts is not None else train_src
        tgt = train_targets if train_targets is not None else train_tgt
        if src is None or tgt is None:
            raise ValueError("fit() expects train_texts/train_targets or train_src/train_tgt")

        base_src = [str(x or "") for x in src]
        base_tgt = [str(x or "") for x in tgt]
        base_ctx = [_canonical_context_label(infer_dialog_domain_with_confidence(x)[0]) for x in base_src]
        base_origin = ["competition"] * len(base_src)

        ext_src, ext_tgt, ext_ctx, ext_origin, ext_stats = _load_external_memory(self.config)
        self.external_memory_stats = ext_stats
        self.external_rows_loaded = len(ext_src)

        all_src_raw = base_src + ext_src
        all_tgt = base_tgt + ext_tgt
        all_ctx = base_ctx + ext_ctx
        all_origin = base_origin + ext_origin

        if not all_src_raw:
            raise RuntimeError("No retrieval memory rows available after loading base + external memory.")

        self.train_src_raw = all_src_raw
        self.train_tgt = all_tgt
        self.train_domains = all_ctx
        self.train_sections = [
            _infer_section_type(
                txt,
                tail_ratio=float(self.config.section_closing_tail_ratio),
            )[0]
            for txt in all_src_raw
        ]
        self.train_origins = all_origin
        self.internal_indices = np.asarray(
            [i for i, o in enumerate(all_origin) if o == "competition"],
            dtype=np.int32,
        )
        self.external_indices = np.asarray(
            [i for i, o in enumerate(all_origin) if o != "competition"],
            dtype=np.int32,
        )
        base_norm = [self._norm(x) for x in base_src]
        ext_norm = [self._norm(x) for x in ext_src]
        self.train_src = base_norm + ext_norm
        base_skel = [_skeletonize_text(x) for x in base_norm]
        ext_skel = [_skeletonize_text(x) for x in ext_norm]
        self.train_skeleton = base_skel + ext_skel
        base_matrix = self.vectorizer.fit_transform(base_norm)
        if ext_norm:
            ext_matrix = self.vectorizer.transform(ext_norm)
            self.train_matrix = sparse.vstack([base_matrix, ext_matrix], format="csr")
        else:
            self.train_matrix = base_matrix
        base_skel_matrix = self.skeleton_vectorizer.fit_transform(base_skel)
        if ext_skel:
            ext_skel_matrix = self.skeleton_vectorizer.transform(ext_skel)
            self.train_skeleton_matrix = sparse.vstack([base_skel_matrix, ext_skel_matrix], format="csr")
        else:
            self.train_skeleton_matrix = base_skel_matrix
        self.train_lens = np.asarray([len(x) for x in self.train_src], dtype=np.int32)

        if self.config.stage2_type == "bm25":
            self.stage2_bm25 = _CharBM25(
                ngram_range=(self.config.ngram_min, self.config.ngram_max),
                k1=self.config.stage2_bm25_k1,
                b=self.config.stage2_bm25_b,
            )
            self.stage2_bm25.fit(self.train_src)
        else:
            self.stage2_bm25 = None
        token_freq: dict[str, int] = {}
        bracket_ratios: list[float] = []
        unknown_ratios: list[float] = []
        bracket_chars_set = "[](){}<>\u2e22\u2e23"
        for text in self.train_src:
            for tok in _PROFILE_TOKEN_RE.findall(text):
                key = tok.lower()
                token_freq[key] = token_freq.get(key, 0) + 1
        # Calibrate fragmentation thresholds on competition memory only to avoid
        # external corpus style shifts from ORACC changing what "fragmentary" means.
        calib_raw = base_src
        calib_norm = base_norm
        for raw, norm in zip(calib_raw, calib_norm):
            raw_s = str(raw or "")
            norm_s = str(norm or "")
            n_chars = max(1, len(raw_s))
            bracket_chars = sum(raw_s.count(ch) for ch in bracket_chars_set)
            bracket_ratios.append(float(bracket_chars / n_chars))
            toks = _PROFILE_TOKEN_RE.findall(norm_s)
            n_tok = max(1, len(toks))
            unknown_count = sum(1 for tok in toks if _PROFILE_UNKNOWN_TOKEN_RE.match(tok) is not None)
            unknown_ratios.append(float(unknown_count / n_tok))
        self.train_token_freq = token_freq
        self.train_bracket_ratio_p80 = float(np.percentile(np.asarray(bracket_ratios, dtype=np.float32), 80))
        self.train_bracket_ratio_p85 = float(np.percentile(np.asarray(bracket_ratios, dtype=np.float32), 85))
        self.train_bracket_ratio_p90 = float(np.percentile(np.asarray(bracket_ratios, dtype=np.float32), 90))
        self.train_unknown_ratio_p90 = float(np.percentile(np.asarray(unknown_ratios, dtype=np.float32), 90))
        uncertainty_vals = [
            float(self._uncertainty_profile(query_raw=raw, query_norm=norm)["uncertainty_ratio"])
            for raw, norm in zip(self.train_src_raw, self.train_src)
        ]
        self.train_uncertainty_ratio = np.asarray(uncertainty_vals, dtype=np.float32)
        self.train_digit_sets = [_extract_digit_tokens(txt) for txt in self.train_src_raw]
        self.train_formula_sets = [_extract_formula_tokens(txt) for txt in self.train_src_raw]
        self.train_slot_signatures = [_slot_signature_text(txt) for txt in self.train_src_raw]
        self.train_variant_token_sets = [set(_variant_tokens(txt)) for txt in self.train_src]
        self._build_variant_equivalences(base_norm=base_norm, base_matrix=base_matrix)

    def _uncertainty_profile(self, query_raw: str, query_norm: str) -> dict[str, float]:
        raw = str(query_raw or "")
        norm = str(query_norm or "")
        tokens = _PROFILE_TOKEN_RE.findall(norm)
        n_tokens = max(1, len(tokens))
        n_chars = max(1, len(raw))

        bracket_chars = sum(raw.count(ch) for ch in "[](){}<>\u2e22\u2e23")
        bracket_ratio = float(bracket_chars / n_chars)
        unknown_token_count = sum(1 for tok in tokens if _PROFILE_UNKNOWN_TOKEN_RE.match(tok) is not None)
        unknown_token_ratio = float(unknown_token_count / n_tokens)
        question_count = raw.count("?")
        question_ratio = float(question_count / n_chars)
        reconstruction_ratio = float(min(1.0, bracket_ratio + unknown_token_ratio))
        rare_cutoff = int(max(1, self.config.uncertainty_rare_cutoff))
        rare_token_count = sum(1 for tok in tokens if self.train_token_freq.get(tok.lower(), 0) <= rare_cutoff)
        rare_token_ratio = float(rare_token_count / n_tokens)

        uncertainty_ratio = float(
            min(
                1.0,
                0.55 * reconstruction_ratio
                + 0.30 * unknown_token_ratio
                + 0.10 * rare_token_ratio
                + 0.05 * question_ratio,
            )
        )

        return {
            "query_tokens": float(len(tokens)),
            "query_chars": float(len(raw)),
            "bracket_ratio": bracket_ratio,
            "unknown_token_ratio": unknown_token_ratio,
            "question_ratio": question_ratio,
            "reconstruction_ratio": reconstruction_ratio,
            "rare_token_ratio": rare_token_ratio,
            "uncertainty_ratio": uncertainty_ratio,
        }

    def _build_variant_equivalences(self, base_norm: list[str], base_matrix: sparse.spmatrix) -> None:
        self.variant_equiv_map = {}
        self.variant_equiv_pairs = []
        if not self.config.enable_variant_equivalence:
            return
        n = len(base_norm)
        if n < 2:
            return

        k = int(max(1, self.config.variant_neighbor_k))
        sim_thr = float(max(0.0, min(1.0, self.config.variant_similarity_threshold)))
        min_support = int(max(1, self.config.variant_min_support))
        max_pairs = int(max(0, self.config.variant_max_pairs))
        if max_pairs == 0:
            return

        base_toks = [_variant_tokens(txt) for txt in base_norm]
        sim = cosine_similarity(base_matrix, base_matrix)
        pair_counts: dict[tuple[str, str], int] = {}

        for i in range(n):
            row = sim[i]
            cand = np.where((row >= sim_thr) & (row < 0.999999))[0]
            if cand.size == 0:
                continue
            order = cand[np.argsort(row[cand])[::-1][:k]]
            toks_i = base_toks[i]
            for j in order:
                jj = int(j)
                if jj <= i:
                    continue
                _collect_context_equivalence_counts(toks_i, base_toks[jj], pair_counts)

        pairs: list[tuple[str, str, int]] = []
        for (a, b), c in pair_counts.items():
            if c < min_support:
                continue
            pairs.append((a, b, int(c)))
        pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        eq_map: dict[str, set[str]] = {}
        for a, b, _ in pairs:
            eq_map.setdefault(a, set()).add(b)
        self.variant_equiv_map = eq_map
        self.variant_equiv_pairs = pairs[: min(50, len(pairs))]

    def _variant_equiv_overlap(self, query_tokens: set[str], cand_tokens: set[str]) -> float:
        if not query_tokens or not cand_tokens or not self.variant_equiv_map:
            return 0.0
        possible = 0
        matched = 0
        for tok in query_tokens:
            equiv = self.variant_equiv_map.get(tok)
            if not equiv:
                continue
            possible += 1
            if tok in cand_tokens:
                continue
            if any(e in cand_tokens for e in equiv):
                matched += 1
        return float(matched / possible) if possible else 0.0

    def _candidate_indices(self, row: np.ndarray, k: int) -> np.ndarray:
        k = int(max(1, min(k, len(row))))
        idx = np.argpartition(row, -k)[-k:]
        return idx[np.argsort(row[idx])[::-1]]

    def _candidate_indices_from_subset(self, row: np.ndarray, subset_idx: np.ndarray, k: int) -> np.ndarray:
        if subset_idx.size == 0:
            return np.asarray([], dtype=np.int32)
        vals = row[subset_idx]
        k = int(max(1, min(k, len(subset_idx))))
        pos = np.argpartition(vals, -k)[-k:]
        chosen = subset_idx[pos]
        return chosen[np.argsort(row[chosen])[::-1]]

    def _candidate_indices_with_section(
        self,
        *,
        row: np.ndarray,
        subset_idx: np.ndarray,
        k: int,
        query_section: str,
        query_section_score: float,
    ) -> tuple[np.ndarray, bool, int]:
        if subset_idx.size == 0:
            return np.asarray([], dtype=np.int32), False, 0
        k = int(max(1, min(k, len(subset_idx))))
        if not self.config.enable_section_border:
            return self._candidate_indices_from_subset(row, subset_idx, k), False, 0
        if (
            not self.config.section_force_closing
            or query_section != "closing"
            or float(query_section_score) < float(self.config.section_force_min_score)
        ):
            return self._candidate_indices_from_subset(row, subset_idx, k), False, 0

        matched = np.asarray(
            [int(idx) for idx in subset_idx if self.train_sections[int(idx)] == query_section],
            dtype=np.int32,
        )
        min_pool = int(max(1, min(self.config.section_min_pool, k)))
        if matched.size >= min_pool:
            return self._candidate_indices_from_subset(row, matched, k), True, int(matched.size)
        return self._candidate_indices_from_subset(row, subset_idx, k), False, int(matched.size)

    def _len_score(self, q_len: int, d_len: int) -> float:
        q = max(1, int(q_len))
        d = max(1, int(d_len))
        if self.config.len_mode == "ratio":
            return float(min(q, d) / max(q, d))
        if self.config.len_mode == "logexp":
            return float(np.exp(-abs(np.log((d + 1.0) / (q + 1.0)))))
        raise ValueError(f"Unknown len_mode: {self.config.len_mode}")

    def _stage2_raw(self, query: str, train_idx: int, bm25_row: np.ndarray | None) -> float:
        st = self.config.stage2_type
        if st == "none":
            return 0.0
        if st == "bm25":
            if bm25_row is None:
                return 0.0
            return float(bm25_row[train_idx])
        if st == "seq_ratio":
            return float(SequenceMatcher(None, query, self.train_src[train_idx]).ratio())
        if st == "token_overlap":
            return float(_token_overlap(query, self.train_src[train_idx]))
        raise ValueError(f"Unknown stage2_type: {st}")

    def _rerank_components(
        self,
        row: np.ndarray,
        query: str,
        q_len: int,
        candidates: np.ndarray,
        bm25_row: np.ndarray | None,
        query_section: str | None = None,
        len_weight_override: float | None = None,
        stage2_weight_override: float | None = None,
        row_skeleton: np.ndarray | None = None,
        query_digits: set[str] | None = None,
        query_formula: set[str] | None = None,
        query_slot_signature: str | None = None,
        query_variant_tokens: set[str] | None = None,
        high_uncertainty: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        len_weight = float(self.config.len_weight if len_weight_override is None else len_weight_override)
        stage2_weight = float(self.config.stage2_weight if stage2_weight_override is None else stage2_weight_override)
        skel_blend = float(min(1.0, max(0.0, self.config.uncertainty_skeleton_blend)))
        num_bonus_w = float(self.config.uncertainty_number_bonus)
        form_bonus_w = float(self.config.uncertainty_formula_bonus)
        slot_bonus_w = float(self.config.uncertainty_slot_bonus)
        variant_bonus_w = float(self.config.uncertainty_variant_bonus)
        uncertain_pen_w = float(self.config.uncertainty_candidate_uncertain_penalty)
        query_digits = query_digits or set()
        query_formula = query_formula or set()
        query_slot_signature = str(query_slot_signature or "")
        query_variant_tokens = query_variant_tokens or set()

        base_vals: list[float] = []
        for idx in candidates:
            i = int(idx)
            cosine_val = float(row[i])
            if high_uncertainty and row_skeleton is not None:
                cosine_skel = float(row_skeleton[i])
                cosine_val = float((1.0 - skel_blend) * cosine_val + skel_blend * cosine_skel)

            score = cosine_val
            score += len_weight * self._len_score(q_len=q_len, d_len=int(self.train_lens[i]))
            score += (
                self.config.competition_origin_bonus
                if self.train_origins[i] == "competition"
                else self.config.external_origin_bonus
            )
            if (
                self.config.enable_section_border
                and query_section is not None
                and self.train_sections[i] == query_section
            ):
                score += float(self.config.section_match_bonus)

            if high_uncertainty:
                score += num_bonus_w * _set_overlap(query_digits, self.train_digit_sets[i])
                score += form_bonus_w * _set_overlap(query_formula, self.train_formula_sets[i])
                score += slot_bonus_w * _slot_similarity(query_slot_signature, self.train_slot_signatures[i])
                if (
                    variant_bonus_w > 0.0
                    and self.config.enable_variant_equivalence
                    and self.variant_equiv_map
                    and i < len(self.train_variant_token_sets)
                ):
                    score += variant_bonus_w * self._variant_equiv_overlap(
                        query_tokens=query_variant_tokens,
                        cand_tokens=self.train_variant_token_sets[i],
                    )
                if self.train_uncertainty_ratio is not None:
                    score -= uncertain_pen_w * float(self.train_uncertainty_ratio[i])

            base_vals.append(float(score))

        base = np.asarray(base_vals, dtype=np.float32)
        stage2_norm_full = np.zeros(len(candidates), dtype=np.float32)
        if self.config.stage2_type == "none" or stage2_weight <= 0.0:
            return base.copy(), base, stage2_norm_full

        pool_n = int(max(1, min(self.config.stage2_pool, len(candidates))))
        stage2_raw = np.asarray(
            [self._stage2_raw(query=query, train_idx=int(idx), bm25_row=bm25_row) for idx in candidates[:pool_n]],
            dtype=np.float32,
        )
        if stage2_raw.size > 0:
            mn = float(stage2_raw.min())
            mx = float(stage2_raw.max())
            if mx > mn:
                stage2_norm = (stage2_raw - mn) / (mx - mn)
            else:
                stage2_norm = np.zeros_like(stage2_raw)
            stage2_norm_full[:pool_n] = stage2_norm
        out = base.copy()
        out[:pool_n] += stage2_weight * stage2_norm_full[:pool_n]
        return out, base, stage2_norm_full

    def _build_evidence(
        self,
        *,
        query: str,
        query_context: str,
        query_section: str,
        candidates: np.ndarray,
        final_scores: np.ndarray,
        base_scores: np.ndarray,
        stage2_norm: np.ndarray,
        context_bonus: np.ndarray,
        row: np.ndarray,
        q_len: int,
        bm25_row: np.ndarray | None,
    ) -> list[dict]:
        k = int(max(1, self.config.evidence_k))
        order = np.argsort(final_scores)[::-1][:k]
        out: list[dict] = []
        for rank, local_pos in enumerate(order, start=1):
            idx = int(candidates[local_pos])
            len_term = float(self._len_score(q_len=q_len, d_len=int(self.train_lens[idx])))
            stage2_raw = float(self._stage2_raw(query=query, train_idx=idx, bm25_row=bm25_row))
            section = self.train_sections[idx]
            section_bonus = (
                float(self.config.section_match_bonus)
                if (self.config.enable_section_border and section == query_section)
                else 0.0
            )
            out.append(
                {
                    "rank": int(rank),
                    "memory_idx": int(idx),
                    "origin": self.train_origins[idx],
                    "context": self.train_domains[idx],
                    "section": section,
                    "context_match_query": bool(self.train_domains[idx] == query_context),
                    "section_match_query": bool(section == query_section),
                    "score_final": float(final_scores[local_pos]),
                    "score_base": float(base_scores[local_pos]),
                    "score_cosine": float(row[idx]),
                    "score_len_term": len_term,
                    "score_stage2_raw": stage2_raw,
                    "score_stage2_norm": float(stage2_norm[local_pos]),
                    "score_context_bonus": float(context_bonus[local_pos]),
                    "score_section_bonus": section_bonus,
                    "source_text": _short(self.train_src_raw[idx]),
                    "target_text": _short(self.train_tgt[idx]),
                }
            )
        return out

    def _predict_one(self, query: str, row: np.ndarray, row_skeleton: np.ndarray | None = None) -> tuple[int, dict]:
        q = self._norm(query)
        q_len = len(q)
        cfg = self.config
        query_digits = _extract_digit_tokens(query)
        query_formula = _extract_formula_tokens(query)
        query_slot_signature = _slot_signature_text(query)
        query_variant_tokens = set(_variant_tokens(q))

        bm25_row = self.stage2_bm25.score_one(q) if (cfg.stage2_type == "bm25" and self.stage2_bm25 is not None) else None

        query_context, query_conf, query_context_scores = infer_dialog_domain_with_confidence(q)
        query_context = _canonical_context_label(query_context)
        query_section, query_section_meta = _infer_section_type(
            query,
            tail_ratio=float(cfg.section_closing_tail_ratio),
        )
        query_section_score = float(query_section_meta.get("score", 0.0))

        uncertainty = self._uncertainty_profile(query_raw=query, query_norm=q)
        uncertainty_ratio = float(uncertainty.get("uncertainty_ratio", 0.0))
        bracket_ratio = float(uncertainty.get("bracket_ratio", 0.0))
        unknown_ratio = float(uncertainty.get("unknown_token_ratio", 0.0))
        bracket_pct = float(min(99.9, max(0.0, cfg.uncertainty_bracket_percentile)))
        if bracket_pct <= 80.0:
            bracket_thr = float(self.train_bracket_ratio_p80)
        elif bracket_pct <= 85.0:
            lo = float(self.train_bracket_ratio_p80)
            hi = float(self.train_bracket_ratio_p85)
            alpha = float((bracket_pct - 80.0) / 5.0)
            bracket_thr = float(lo + alpha * (hi - lo))
        elif bracket_pct <= 90.0:
            lo = float(self.train_bracket_ratio_p85)
            hi = float(self.train_bracket_ratio_p90)
            alpha = float((bracket_pct - 85.0) / 5.0)
            bracket_thr = float(lo + alpha * (hi - lo))
        else:
            bracket_thr = float(self.train_bracket_ratio_p90)
        if bracket_thr <= 0.0:
            bracket_thr = 0.01
        unknown_p90 = float(self.train_unknown_ratio_p90) if self.train_unknown_ratio_p90 > 0.0 else 0.01
        frag_score = max(
            float(bracket_ratio / max(bracket_thr, 1e-6)),
            float(unknown_ratio / max(unknown_p90, 1e-6)),
        )
        if frag_score >= 1.0:
            query_fragmentation = "fragmentary"
        elif frag_score >= 0.45:
            query_fragmentation = "partial"
        else:
            query_fragmentation = "complete"

        effective_top_k = int(cfg.top_k)
        effective_external_cap = int(max(1, cfg.external_candidate_cap))
        effective_len_weight = float(cfg.len_weight)
        effective_stage2_weight = float(cfg.stage2_weight)
        structural_trigger = False
        ambiguity_trigger = False
        structural_trigger_reasons: list[str] = []
        ambiguity_trigger_reasons: list[str] = []
        high_uncertainty = False
        uncertainty_external_bonus = 0.0
        skel_internal_added = 0
        skel_external_added = 0
        skel_global_added = 0

        internal_cands, internal_section_filtered, internal_section_pool = self._candidate_indices_with_section(
            row=row,
            subset_idx=self.internal_indices,
            k=cfg.top_k,
            query_section=query_section,
            query_section_score=query_section_score,
        )
        if internal_cands.size == 0:
            internal_cands = self._candidate_indices(row, cfg.top_k)
            internal_section_filtered = False
            internal_section_pool = 0
        internal_scores, internal_base, internal_stage2_norm = self._rerank_components(
            row=row,
            query=q,
            q_len=q_len,
            candidates=internal_cands,
            bm25_row=bm25_row,
            query_section=query_section,
            len_weight_override=cfg.len_weight,
            stage2_weight_override=cfg.stage2_weight,
            row_skeleton=row_skeleton,
            query_digits=query_digits,
            query_formula=query_formula,
            query_slot_signature=query_slot_signature,
            query_variant_tokens=query_variant_tokens,
            high_uncertainty=False,
        )
        internal_pos = int(np.argmax(internal_scores))
        internal_idx = int(internal_cands[internal_pos])
        internal_score = float(internal_scores[internal_pos])
        internal_top_cosine = float(row[internal_idx])
        if len(internal_cands) > 1:
            internal_second_idx = int(internal_cands[1])
            internal_second_cosine = float(row[internal_second_idx])
            internal_gap = float(internal_top_cosine - internal_second_cosine)
        else:
            internal_second_idx = None
            internal_second_cosine = None
            internal_gap = None

        if query_fragmentation == "fragmentary":
            structural_trigger_reasons.append("fragmentary")
        if bracket_ratio >= bracket_thr:
            structural_trigger_reasons.append("bracket_percentile")
        structural_trigger = bool(structural_trigger_reasons)

        if internal_top_cosine < float(cfg.uncertainty_internal_top_threshold):
            ambiguity_trigger_reasons.append("internal_top_low")
        if internal_gap is not None and internal_gap < float(cfg.uncertainty_internal_gap_threshold):
            ambiguity_trigger_reasons.append("internal_gap_low")
        ambiguity_trigger = bool(ambiguity_trigger_reasons)

        high_uncertainty = bool(cfg.enable_uncertainty_adaptation and structural_trigger and ambiguity_trigger)
        if high_uncertainty:
            k_boost = max(0.0, float(cfg.uncertainty_topk_boost))
            topk_add = int(max(0, cfg.uncertainty_topk_add)) + int(round(cfg.top_k * k_boost))
            ext_add = int(max(0, cfg.uncertainty_external_cap_add)) + int(
                round(max(1, cfg.external_candidate_cap) * k_boost)
            )
            effective_top_k = int(max(1, min(len(row), cfg.top_k + topk_add)))
            effective_external_cap = int(max(1, min(len(row), cfg.external_candidate_cap + ext_add)))
            stage2_discount = min(1.0, max(0.0, float(cfg.uncertainty_stage2_discount)))
            len_discount = min(1.0, max(0.0, float(cfg.uncertainty_len_discount)))
            effective_stage2_weight = float(cfg.stage2_weight * (1.0 - stage2_discount))
            effective_len_weight = float(cfg.len_weight * (1.0 - len_discount))
            uncertainty_external_bonus = float(cfg.uncertainty_external_bonus)

            # Re-evaluate internal neighborhood with the uncertainty-adapted pool.
            internal_cands, internal_section_filtered, internal_section_pool = self._candidate_indices_with_section(
                row=row,
                subset_idx=self.internal_indices,
                k=effective_top_k,
                query_section=query_section,
                query_section_score=query_section_score,
            )
            if internal_cands.size == 0:
                internal_cands = self._candidate_indices(row, effective_top_k)
                internal_section_filtered = False
                internal_section_pool = 0
            skel_internal_added = 0
            if cfg.enable_skeleton_retrieval_path and row_skeleton is not None:
                skel_k = int(max(1, min(len(row), max(effective_top_k, cfg.skeleton_candidate_top_k))))
                skel_internal, _, _ = self._candidate_indices_with_section(
                    row=row_skeleton,
                    subset_idx=self.internal_indices,
                    k=skel_k,
                    query_section=query_section,
                    query_section_score=query_section_score,
                )
                if skel_internal.size > 0:
                    merged = list(int(x) for x in internal_cands.tolist())
                    seen = set(merged)
                    for idx in skel_internal.tolist():
                        j = int(idx)
                        if j not in seen:
                            seen.add(j)
                            merged.append(j)
                    skel_internal_added = int(max(0, len(merged) - len(internal_cands)))
                    internal_cands = np.asarray(merged, dtype=np.int32)
            else:
                skel_internal_added = 0
            internal_scores, internal_base, internal_stage2_norm = self._rerank_components(
                row=row,
                query=q,
                q_len=q_len,
                candidates=internal_cands,
                bm25_row=bm25_row,
                query_section=query_section,
                len_weight_override=effective_len_weight,
                stage2_weight_override=effective_stage2_weight,
                row_skeleton=row_skeleton,
                query_digits=query_digits,
                query_formula=query_formula,
                query_slot_signature=query_slot_signature,
                query_variant_tokens=query_variant_tokens,
                high_uncertainty=True,
            )
            internal_pos = int(np.argmax(internal_scores))
            internal_idx = int(internal_cands[internal_pos])
            internal_score = float(internal_scores[internal_pos])
            internal_top_cosine = float(row[internal_idx])
            if len(internal_cands) > 1:
                internal_second_idx = int(internal_cands[1])
                internal_second_cosine = float(row[internal_second_idx])
                internal_gap = float(internal_top_cosine - internal_second_cosine)
            else:
                internal_second_idx = None
                internal_second_cosine = None
                internal_gap = None

        allow_external = True
        external_trigger_reasons: list[str] = []
        if cfg.external_enable_fallback and self.external_indices.size > 0:
            allow_external = False
            top_thr = float(cfg.external_internal_top_threshold)
            gap_thr = float(cfg.external_internal_gap_threshold)
            force_ctx = {_canonical_context_label(x) for x in cfg.external_force_contexts if str(x).strip()}
            if top_thr >= 0.0 and internal_top_cosine < top_thr:
                allow_external = True
                external_trigger_reasons.append("low_internal_top")
            if gap_thr >= 0.0 and internal_gap is not None and internal_gap < gap_thr:
                allow_external = True
                external_trigger_reasons.append("ambiguous_internal_gap")
            if force_ctx and query_context in force_ctx:
                allow_external = True
                external_trigger_reasons.append("forced_context")
            if high_uncertainty:
                allow_external = True
                external_trigger_reasons.append("high_uncertainty")
            if not external_trigger_reasons:
                external_trigger_reasons.append("fallback_blocked")
        elif not cfg.external_enable_fallback:
            external_trigger_reasons.append("fallback_disabled")

        external_section_filtered = False
        external_section_pool = 0
        if cfg.external_enable_fallback:
            pool_list = [int(i) for i in internal_cands.tolist()]
            if allow_external and self.external_indices.size > 0:
                ext_cap = int(max(1, effective_external_cap))
                external_cands, external_section_filtered, external_section_pool = self._candidate_indices_with_section(
                    row=row,
                    subset_idx=self.external_indices,
                    k=ext_cap,
                    query_section=query_section,
                    query_section_score=query_section_score,
                )
                pool_list.extend(int(i) for i in external_cands.tolist())
                if high_uncertainty and cfg.enable_skeleton_retrieval_path and row_skeleton is not None:
                    skel_ext_k = int(max(1, min(len(row), max(ext_cap, cfg.skeleton_candidate_top_k))))
                    skel_external, _, _ = self._candidate_indices_with_section(
                        row=row_skeleton,
                        subset_idx=self.external_indices,
                        k=skel_ext_k,
                        query_section=query_section,
                        query_section_score=query_section_score,
                    )
                    skel_external_added = int(len(skel_external))
                    pool_list.extend(int(i) for i in skel_external.tolist())
            # Keep unique order.
            seen = set()
            candidate_pool = []
            for idx in pool_list:
                if idx not in seen:
                    seen.add(idx)
                    candidate_pool.append(idx)
            if not candidate_pool:
                candidate_pool = [int(internal_idx)]
            candidate_pool_arr = np.asarray(candidate_pool, dtype=np.int32)
            global_section_filtered = bool(internal_section_filtered or external_section_filtered)
            global_section_pool = int(internal_section_pool + external_section_pool)
        else:
            all_indices = np.arange(len(row), dtype=np.int32)
            candidate_pool_arr, global_section_filtered, global_section_pool = self._candidate_indices_with_section(
                row=row,
                subset_idx=all_indices,
                k=effective_top_k,
                query_section=query_section,
                query_section_score=query_section_score,
            )
            if high_uncertainty and cfg.enable_skeleton_retrieval_path and row_skeleton is not None:
                skel_k = int(max(1, min(len(row), max(effective_top_k, cfg.skeleton_candidate_top_k))))
                skel_global, _, _ = self._candidate_indices_with_section(
                    row=row_skeleton,
                    subset_idx=all_indices,
                    k=skel_k,
                    query_section=query_section,
                    query_section_score=query_section_score,
                )
                if skel_global.size > 0:
                    merged = list(int(x) for x in candidate_pool_arr.tolist())
                    seen = set(merged)
                    for idx in skel_global.tolist():
                        j = int(idx)
                        if j not in seen:
                            seen.add(j)
                            merged.append(j)
                    skel_global_added = int(max(0, len(merged) - len(candidate_pool_arr)))
                    candidate_pool_arr = np.asarray(merged, dtype=np.int32)

        len_filter_applied = False
        len_filter_threshold = float(max(0.0, cfg.uncertainty_triggered_len_ratio_min))
        len_filter_kept = int(len(candidate_pool_arr))
        if high_uncertainty and len_filter_threshold > 0.0 and len(candidate_pool_arr) > 1:
            filtered = [
                int(idx)
                for idx in candidate_pool_arr
                if (
                    self._len_score(q_len=q_len, d_len=int(self.train_lens[int(idx)])) >= len_filter_threshold
                    or int(idx) == int(internal_idx)
                )
            ]
            if filtered and len(filtered) < len(candidate_pool_arr):
                candidate_pool_arr = np.asarray(filtered, dtype=np.int32)
                len_filter_applied = True
                len_filter_kept = int(len(filtered))

        global_scores, global_base, global_stage2_norm = self._rerank_components(
            row=row,
            query=q,
            q_len=q_len,
            candidates=candidate_pool_arr,
            bm25_row=bm25_row,
            query_section=query_section,
            len_weight_override=effective_len_weight,
            stage2_weight_override=effective_stage2_weight,
            row_skeleton=row_skeleton,
            query_digits=query_digits,
            query_formula=query_formula,
            query_slot_signature=query_slot_signature,
            query_variant_tokens=query_variant_tokens,
            high_uncertainty=high_uncertainty,
        )
        total_external_bonus = float(cfg.external_gate_bonus) + float(uncertainty_external_bonus)
        if allow_external and total_external_bonus != 0.0:
            ext_bonus = np.asarray(
                [
                    total_external_bonus
                    if self.train_origins[int(idx)] != "competition"
                    else 0.0
                    for idx in candidate_pool_arr
                ],
                dtype=np.float32,
            )
            global_scores = global_scores + ext_bonus
        else:
            ext_bonus = np.zeros(len(candidate_pool_arr), dtype=np.float32)

        global_pos = int(np.argmax(global_scores))
        global_idx = int(candidate_pool_arr[global_pos])
        global_score = float(global_scores[global_pos])

        if allow_external and self.train_origins[global_idx] != "competition":
            if (global_score - internal_score) < float(cfg.external_gate_margin):
                global_idx = internal_idx
                global_score = internal_score
                global_pos = int(np.where(candidate_pool_arr == internal_idx)[0][0])

        chosen_idx = global_idx
        chosen_score = global_score
        override = False

        # Context override constrained to the same candidate pool (no new external leakage).
        domain_rank = np.argsort(row[candidate_pool_arr])[::-1]
        domain_cands = candidate_pool_arr[domain_rank[: min(cfg.domain_candidate_top_k, len(candidate_pool_arr))]]
        chosen_cands = candidate_pool_arr
        chosen_scores = global_scores
        chosen_base = global_base
        chosen_stage2_norm = global_stage2_norm
        chosen_context_bonus = np.zeros(len(candidate_pool_arr), dtype=np.float32)

        if cfg.enable_domain_override and query_context != "unknown" and query_conf >= cfg.domain_conf_threshold:
            context_scores, context_base, context_stage2_norm = self._rerank_components(
                row=row,
                query=q,
                q_len=q_len,
                candidates=domain_cands,
                bm25_row=bm25_row,
                query_section=query_section,
                len_weight_override=effective_len_weight,
                stage2_weight_override=effective_stage2_weight,
                row_skeleton=row_skeleton,
                query_digits=query_digits,
                query_formula=query_formula,
                query_slot_signature=query_slot_signature,
                query_variant_tokens=query_variant_tokens,
                high_uncertainty=high_uncertainty,
            )
            context_bonus = np.asarray(
                [float(cfg.domain_bonus) if self.train_domains[int(idx)] == query_context else 0.0 for idx in domain_cands],
                dtype=np.float32,
            )
            context_scores_bonus = context_scores + context_bonus
            dom_pos = int(np.argmax(context_scores_bonus))
            dom_idx = int(domain_cands[dom_pos])
            dom_score = float(context_scores_bonus[dom_pos])

            if (
                self.train_domains[dom_idx] == query_context
                and dom_idx != global_idx
                and (dom_score - global_score) >= cfg.domain_margin
            ):
                chosen_idx = dom_idx
                chosen_score = dom_score
                override = True
                chosen_cands = domain_cands
                chosen_scores = context_scores_bonus
                chosen_base = context_base
                chosen_stage2_norm = context_stage2_norm
                chosen_context_bonus = context_bonus

        evidence = self._build_evidence(
            query=q,
            query_context=query_context,
            query_section=query_section,
            candidates=chosen_cands,
            final_scores=chosen_scores,
            base_scores=chosen_base,
            stage2_norm=chosen_stage2_norm,
            context_bonus=chosen_context_bonus,
            row=row,
            q_len=q_len,
            bm25_row=bm25_row,
        )

        chosen_variant_overlap = 0.0
        if (
            high_uncertainty
            and cfg.enable_variant_equivalence
            and cfg.uncertainty_variant_bonus > 0.0
            and chosen_idx < len(self.train_variant_token_sets)
            and query_variant_tokens
        ):
            chosen_variant_overlap = float(
                self._variant_equiv_overlap(
                    query_tokens=query_variant_tokens,
                    cand_tokens=self.train_variant_token_sets[chosen_idx],
                )
            )

        debug = {
            "query_context": query_context,
            "query_context_confidence": float(query_conf),
            "query_context_scores": query_context_scores,
            "query_section": query_section,
            "query_section_meta": query_section_meta,
            "query_section_score": query_section_score,
            "section_border_enabled": bool(cfg.enable_section_border),
            "section_force_closing": bool(cfg.section_force_closing),
            "section_force_min_score": float(cfg.section_force_min_score),
            "section_match_bonus": float(cfg.section_match_bonus),
            "section_closing_tail_ratio": float(cfg.section_closing_tail_ratio),
            "section_min_pool": int(cfg.section_min_pool),
            "internal_section_filtered": bool(internal_section_filtered),
            "internal_section_pool": int(internal_section_pool),
            "external_section_filtered": bool(external_section_filtered),
            "external_section_pool": int(external_section_pool),
            "global_section_filtered": bool(global_section_filtered),
            "global_section_pool": int(global_section_pool),
            "internal_top_idx": int(internal_idx),
            "internal_top_score": float(internal_score),
            "internal_top_cosine": float(internal_top_cosine),
            "internal_second_idx": int(internal_second_idx) if internal_second_idx is not None else None,
            "internal_second_cosine": float(internal_second_cosine) if internal_second_cosine is not None else None,
            "internal_gap": float(internal_gap) if internal_gap is not None else None,
            "external_fallback_enabled": bool(cfg.external_enable_fallback),
            "external_allowed": bool(allow_external),
            "external_trigger_reasons": external_trigger_reasons,
            "external_candidate_cap": int(cfg.external_candidate_cap),
            "global_idx": int(global_idx),
            "global_score": float(global_score),
            "global_origin": self.train_origins[global_idx],
            "global_context": self.train_domains[global_idx],
            "global_section": self.train_sections[global_idx],
            "chosen_idx": int(chosen_idx),
            "chosen_score": float(chosen_score),
            "chosen_origin": self.train_origins[chosen_idx],
            "chosen_context": self.train_domains[chosen_idx],
            "chosen_section": self.train_sections[chosen_idx],
            "section_match_chosen": bool(self.train_sections[chosen_idx] == query_section),
            "override_used": bool(override),
            "score_margin_vs_global": float(chosen_score - global_score),
            "stage2_type": cfg.stage2_type,
            "stage2_pool": cfg.stage2_pool,
            "stage2_weight": cfg.stage2_weight,
            "effective_top_k": int(effective_top_k),
            "effective_external_candidate_cap": int(effective_external_cap),
            "effective_stage2_weight": float(effective_stage2_weight),
            "effective_len_weight": float(effective_len_weight),
            "high_uncertainty": bool(high_uncertainty),
            "skeleton_path_enabled": bool(cfg.enable_skeleton_retrieval_path),
            "skeleton_candidate_top_k": int(cfg.skeleton_candidate_top_k),
            "skeleton_internal_added": int(skel_internal_added),
            "skeleton_external_added": int(skel_external_added),
            "skeleton_global_added": int(skel_global_added),
            "query_fragmentation": str(query_fragmentation),
            "query_frag_score": float(frag_score),
            "train_bracket_ratio_p80": float(self.train_bracket_ratio_p80),
            "train_bracket_ratio_p85": float(self.train_bracket_ratio_p85),
            "train_bracket_ratio_p90": float(self.train_bracket_ratio_p90),
            "uncertainty_bracket_percentile": float(bracket_pct),
            "uncertainty_bracket_threshold": float(bracket_thr),
            "train_unknown_ratio_p90": float(unknown_p90),
            "uncertainty_structural_trigger": bool(structural_trigger),
            "uncertainty_ambiguity_trigger": bool(ambiguity_trigger),
            "uncertainty_structural_trigger_reasons": structural_trigger_reasons,
            "uncertainty_ambiguity_trigger_reasons": ambiguity_trigger_reasons,
            "uncertainty_threshold": float(cfg.uncertainty_high_threshold),
            "uncertainty_closing_threshold": float(cfg.uncertainty_closing_threshold),
            "uncertainty_internal_top_threshold": float(cfg.uncertainty_internal_top_threshold),
            "uncertainty_internal_gap_threshold": float(cfg.uncertainty_internal_gap_threshold),
            "uncertainty_external_bonus": float(uncertainty_external_bonus),
            "uncertainty_triggered_len_ratio_min": float(len_filter_threshold),
            "uncertainty_len_filter_applied": bool(len_filter_applied),
            "uncertainty_len_filter_kept": int(len_filter_kept),
            "uncertainty_skeleton_blend": float(cfg.uncertainty_skeleton_blend),
            "uncertainty_number_bonus": float(cfg.uncertainty_number_bonus),
            "uncertainty_formula_bonus": float(cfg.uncertainty_formula_bonus),
            "uncertainty_slot_bonus": float(cfg.uncertainty_slot_bonus),
            "uncertainty_variant_bonus": float(cfg.uncertainty_variant_bonus),
            "uncertainty_candidate_uncertain_penalty": float(cfg.uncertainty_candidate_uncertain_penalty),
            "variant_equivalence_enabled": bool(cfg.enable_variant_equivalence),
            "variant_equivalence_map_size": int(len(self.variant_equiv_map)),
            "variant_equivalence_pairs": int(len(self.variant_equiv_pairs)),
            "chosen_variant_equiv_overlap": float(chosen_variant_overlap),
            "uncertainty_profile": uncertainty,
            "policy_name": str(cfg.policy_name),
            "memory_total_rows": int(len(self.train_src_raw)),
            "memory_external_rows": int(self.external_rows_loaded),
            "evidence": evidence,
        }
        return chosen_idx, debug

    def retrieve_translate(self, query: str, *, return_debug: bool = False):
        if self.train_matrix is None:
            raise RuntimeError("Model not fit() yet.")
        q_norm = self._norm(query)
        q_mat = self.vectorizer.transform([q_norm])
        row = cosine_similarity(q_mat, self.train_matrix)[0]
        if self.train_skeleton_matrix is not None:
            q_skel = _skeletonize_text(q_norm)
            q_skel_mat = self.skeleton_vectorizer.transform([q_skel])
            row_skel = cosine_similarity(q_skel_mat, self.train_skeleton_matrix)[0]
        else:
            row_skel = None
        idx, dbg = self._predict_one(query=query, row=row, row_skeleton=row_skel)
        pred = self.train_tgt[idx]
        if return_debug:
            return pred, dbg
        return pred

    def predict(self, test_texts: list[str], return_debug: bool = False):
        if self.train_matrix is None:
            raise RuntimeError("Model not fit() yet.")
        q_norm = [self._norm(x) for x in test_texts]
        q_mat = self.vectorizer.transform(q_norm)
        sims = cosine_similarity(q_mat, self.train_matrix)
        if self.train_skeleton_matrix is not None:
            q_skel = [_skeletonize_text(x) for x in q_norm]
            q_skel_mat = self.skeleton_vectorizer.transform(q_skel)
            sims_skel = cosine_similarity(q_skel_mat, self.train_skeleton_matrix)
        else:
            sims_skel = None

        preds: list[str] = []
        debug_rows: list[dict] = []
        for i, q in enumerate(test_texts):
            row_skel = None if sims_skel is None else sims_skel[i]
            idx, dbg = self._predict_one(query=q, row=sims[i], row_skeleton=row_skel)
            preds.append(self.train_tgt[idx])
            debug_rows.append(dbg)
        if return_debug:
            return preds, debug_rows
        return preds


@dataclass(frozen=True)
class RetrievalSubmissionResult:
    output_csv: Path
    metadata_json: Path
    output_sha256: str
    rows: int
    model_id: str
    evidence_json: Path | None = None
    routing_json: Path | None = None
    routing_map_path: Path | None = None
    routing_map_sha256: str | None = None
    route_counts: dict[str, int] | None = None
    policy_model_ids: dict[str, str] | None = None
    reranker_model_path: Path | None = None
    reranker_model_sha256: str | None = None


def make_retrieval_submission(
    output_csv: Path,
    competition_dir: Path,
    schema_path: Path,
    config: RetrievalConfig = CANONICAL_BASELINE_V2,
    verify_determinism: bool = False,
) -> RetrievalSubmissionResult:
    data = load_deep_past_competition(competition_dir, schema_path)
    train_src = data.train[data.schema.train_text_col].astype(str).fillna("").tolist()
    train_tgt = data.train[data.schema.train_target_col].astype(str).fillna("").tolist()
    test_src = data.test[data.schema.test_text_col].astype(str).fillna("").tolist()

    model = CanonicalRetrievalTranslator(config=config)
    model.fit(train_src=train_src, train_tgt=train_tgt)
    preds, debug_rows = model.predict(test_src, return_debug=True)

    if verify_determinism:
        verify_preds = model.predict(test_src, return_debug=False)
        if verify_preds != preds:
            raise RuntimeError("Determinism check failed for retrieval submission.")

    sub = data.sample_submission.copy()
    sub[data.schema.submission_target_col] = preds
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_csv, index=False)

    digest = sha256(output_csv.read_bytes()).hexdigest()
    override_count = int(sum(1 for r in debug_rows if r["override_used"]))
    section_closing_count = int(sum(1 for r in debug_rows if str(r.get("query_section", "")) == "closing"))
    section_match_count = int(sum(1 for r in debug_rows if bool(r.get("section_match_chosen", False))))

    evidence_rows = []
    optional_cols = [data.schema.test_id_col, "text_id", "line_start", "line_end"]
    for i, dbg in enumerate(debug_rows):
        item = {
            "row_index": int(i),
            "query": test_src[i],
            "prediction": preds[i],
            "debug": dbg,
        }
        row = data.test.iloc[i]
        for col in optional_cols:
            if col in data.test.columns:
                value = row[col]
                item[col] = value.item() if hasattr(value, "item") else value
        evidence_rows.append(item)

    evidence_payload = {
        "model_id": config.model_id(),
        "rows": evidence_rows,
    }
    evidence_path = output_csv.with_suffix(output_csv.suffix + ".evidence.json")
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "output_csv": str(output_csv.resolve()),
        "output_sha256": digest,
        "rows": len(sub),
        "model_id": config.model_id(),
        "retrieval_config": asdict(config),
        "domain_override_count": override_count,
        "domain_override_rate": float((override_count / len(debug_rows)) if debug_rows else 0.0),
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(debug_rows)) if debug_rows else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(debug_rows)) if debug_rows else 0.0),
        "determinism_verified": bool(verify_determinism),
        "memory_total_rows": int(len(model.train_src_raw)),
        "memory_external_rows": int(model.external_rows_loaded),
        "external_memory_stats": model.external_memory_stats,
        "evidence_json": str(evidence_path.resolve()),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    meta_path = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return RetrievalSubmissionResult(
        output_csv=output_csv,
        metadata_json=meta_path,
        output_sha256=digest,
        rows=len(sub),
        model_id=config.model_id(),
        evidence_json=evidence_path,
    )


def _source_sha256(text: str) -> str:
    return sha256(str(text or "").encode("utf-8")).hexdigest()


def _default_routing_telemetry_path(output_csv: Path) -> Path:
    return output_csv.with_name(output_csv.stem + ".routing.json")


def _route_policy_params(routing_map: dict[str, Any], route_name: str) -> dict[str, Any]:
    routes = routing_map.get("routes", {})
    for route_def in routes.values():
        if str(route_def.get("route", "")) == route_name:
            return dict(route_def.get("policy_params", {}))
    return {}


def _load_profiles_cache(path: Path, test_src: list[str], routing_map_sha256: str) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("routing_map_sha256", "")) != routing_map_sha256:
        return None
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or len(rows) != len(test_src):
        return None
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            return None
        if str(row.get("source_sha256", "")) != _source_sha256(test_src[i]):
            return None
    return rows


def _write_profiles_cache(path: Path, rows: list[dict], routing_map_sha256: str) -> None:
    payload = {
        "version": 1,
        "routing_map_sha256": routing_map_sha256,
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_routed_retrieval_submission(
    output_csv: Path,
    competition_dir: Path,
    schema_path: Path,
    base_config: RetrievalConfig,
    routing_map_path: Path,
    profiles_cache_path: Path | None = None,
    routing_telemetry_path: Path | None = None,
    verify_determinism: bool = False,
) -> RetrievalSubmissionResult:
    from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, profile_source
    from structural_profile import build_token_frequency

    loaded = load_routing_map(routing_map_path)
    routing_thresholds = dict(loaded.routing_map.get("thresholds", {}))
    routed_model_id = f"RoutedRetrieval(base={base_config.model_id()},routing_sha={loaded.sha256[:12]})"

    data = load_deep_past_competition(competition_dir, schema_path)
    train_src = data.train[data.schema.train_text_col].astype(str).fillna("").tolist()
    train_tgt = data.train[data.schema.train_target_col].astype(str).fillna("").tolist()
    test_src = data.test[data.schema.test_text_col].astype(str).fillna("").tolist()
    token_freq = build_token_frequency(train_src)

    route_to_params = {
        route: _route_policy_params(loaded.routing_map, route)
        for route in ROUTE_TO_POLICY
    }
    policy_configs = {
        policy_name: build_policy_config(
            base_config,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_thresholds,
        )
        for route_name, policy_name in ROUTE_TO_POLICY.items()
    }

    policy_model_ids: dict[str, str] = {}
    policy_preds: dict[str, list[str]] = {}
    policy_debug: dict[str, list[dict]] = {}
    for policy_name, pcfg in policy_configs.items():
        model = CanonicalRetrievalTranslator(config=pcfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, dbg = model.predict(test_src, return_debug=True)
        if verify_determinism:
            verify = model.predict(test_src, return_debug=False)
            if verify != preds:
                raise RuntimeError(f"Determinism check failed for routed policy: {policy_name}")
        policy_model_ids[policy_name] = pcfg.model_id()
        policy_preds[policy_name] = preds
        policy_debug[policy_name] = dbg

    internal_debug = policy_debug["internal_only"]
    computed_profile_rows: list[dict] = []
    for i, src in enumerate(test_src):
        features, labels = profile_source(
            src,
            token_freq=token_freq,
            profile_thresholds=loaded.profile_thresholds,
        )
        dbg = internal_debug[i]
        internal_top_score = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
        raw_gap = dbg.get("internal_gap", None)
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=labels,
            internal_top_score=internal_top_score,
            internal_gap=internal_gap,
            routing_map=loaded.routing_map,
        )
        computed_profile_rows.append(
            {
                "row_index": int(i),
                "source_sha256": _source_sha256(src),
                "features": features,
                "labels": labels,
                "route": decision["route"],
                "policy_name": decision["policy_name"],
                "policy_params": decision["policy_params"],
                "rationale": decision["rationale"],
                "internal_top_score": internal_top_score,
                "internal_gap": internal_gap,
            }
        )

    profile_rows = computed_profile_rows
    if profiles_cache_path is not None:
        if profiles_cache_path.exists():
            cached = _load_profiles_cache(profiles_cache_path, test_src=test_src, routing_map_sha256=loaded.sha256)
            if cached is not None:
                profile_rows = cached
        if profile_rows is computed_profile_rows:
            _write_profiles_cache(profiles_cache_path, rows=profile_rows, routing_map_sha256=loaded.sha256)

    final_preds: list[str] = []
    final_debug: list[dict] = []
    route_counts: dict[str, int] = {}
    unknown_routes = set()
    selected_by_text_id: dict[str, set[int]] = {}
    selected_by_text_id: dict[str, set[int]] = {}
    for i, prow in enumerate(profile_rows):
        policy_name = str(prow.get("policy_name", ""))
        route_name = str(prow.get("route", ""))
        if route_name not in ROUTE_TO_POLICY:
            unknown_routes.add(route_name)
            continue
        if policy_name not in policy_preds:
            unknown_routes.add(policy_name)
            continue
        final_preds.append(policy_preds[policy_name][i])
        dbg = dict(policy_debug[policy_name][i])
        dbg["routed_policy_name"] = policy_name
        dbg["routed_route_name"] = route_name
        dbg["routed_policy_params"] = dict(prow.get("policy_params", {}))
        dbg["routed_rationale"] = str(prow.get("rationale", ""))
        dbg["routed_labels"] = dict(prow.get("labels", {}))
        dbg["routed_features"] = dict(prow.get("features", {}))
        final_debug.append(dbg)
        route_counts[route_name] = route_counts.get(route_name, 0) + 1
    if unknown_routes:
        unknown_list = ", ".join(sorted(unknown_routes))
        raise RuntimeError(f"Unknown routed policies in routing decisions: {unknown_list}")
    if len(final_preds) != len(test_src):
        raise RuntimeError("Routed submission failed preflight: some rows did not produce outputs.")

    if verify_determinism:
        verify_profile_rows: list[dict] = []
        for i, src in enumerate(test_src):
            features, labels = profile_source(
                src,
                token_freq=token_freq,
                profile_thresholds=loaded.profile_thresholds,
            )
            dbg = internal_debug[i]
            top_score = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
            raw_gap = dbg.get("internal_gap", None)
            gap = None if raw_gap is None else float(raw_gap)
            decision = choose_policy(
                labels=labels,
                internal_top_score=top_score,
                internal_gap=gap,
                routing_map=loaded.routing_map,
            )
            verify_profile_rows.append(decision)
        verify_preds = [
            policy_preds[str(verify_profile_rows[i]["policy_name"])][i]
            for i in range(len(test_src))
        ]
        if verify_preds != final_preds:
            raise RuntimeError("Determinism check failed for retrieval_routed submission.")

    sub = data.sample_submission.copy()
    sub[data.schema.submission_target_col] = final_preds
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_csv, index=False)

    digest = sha256(output_csv.read_bytes()).hexdigest()
    section_closing_count = int(sum(1 for dbg in final_debug if str(dbg.get("query_section", "")) == "closing"))
    section_match_count = int(sum(1 for dbg in final_debug if bool(dbg.get("section_match_chosen", False))))

    evidence_rows = []
    telemetry_rows = []
    optional_cols = [data.schema.test_id_col, "text_id", "line_start", "line_end"]
    for i, dbg in enumerate(final_debug):
        prow = profile_rows[i]
        item = {
            "row_index": int(i),
            "query": test_src[i],
            "prediction": final_preds[i],
            "route": str(prow.get("route", "")),
            "policy_name": str(prow.get("policy_name", "")),
            "policy_params": dict(prow.get("policy_params", {})),
            "labels": dict(prow.get("labels", {})),
            "features": dict(prow.get("features", {})),
            "rationale": str(prow.get("rationale", "")),
            "debug": dbg,
        }
        row = data.test.iloc[i]
        for col in optional_cols:
            if col in data.test.columns:
                value = row[col]
                item[col] = value.item() if hasattr(value, "item") else value
        evidence_rows.append(item)
        telemetry_rows.append(
            {
                "row_index": int(i),
                "route": str(prow.get("route", "")),
                "policy_name": str(prow.get("policy_name", "")),
                "policy_params": dict(prow.get("policy_params", {})),
                "labels": dict(prow.get("labels", {})),
                "rationale": str(prow.get("rationale", "")),
                "internal_top_score": float(prow.get("internal_top_score", 0.0)),
                "internal_gap": prow.get("internal_gap", None),
                "chosen_origin": dbg.get("chosen_origin"),
                "chosen_context": dbg.get("chosen_context"),
            }
        )

    evidence_payload = {
        "model_id": routed_model_id,
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "rows": evidence_rows,
    }
    evidence_path = output_csv.with_suffix(output_csv.suffix + ".evidence.json")
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    telemetry_path = routing_telemetry_path or _default_routing_telemetry_path(output_csv)
    telemetry_payload = {
        "model_id": routed_model_id,
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "policy_model_ids": policy_model_ids,
        "rows": telemetry_rows,
    }
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(json.dumps(telemetry_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "output_csv": str(output_csv.resolve()),
        "output_sha256": digest,
        "rows": len(sub),
        "model_id": routed_model_id,
        "retrieval_base_config": asdict(base_config),
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "policy_model_ids": policy_model_ids,
        "determinism_verified": bool(verify_determinism),
        "evidence_json": str(evidence_path.resolve()),
        "routing_telemetry_json": str(telemetry_path.resolve()),
        "profiles_cache_path": str(profiles_cache_path.resolve()) if profiles_cache_path is not None else None,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    meta_path = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return RetrievalSubmissionResult(
        output_csv=output_csv,
        metadata_json=meta_path,
        output_sha256=digest,
        rows=len(sub),
        model_id=routed_model_id,
        evidence_json=evidence_path,
        routing_json=telemetry_path,
        routing_map_path=loaded.path,
        routing_map_sha256=loaded.sha256,
        route_counts=route_counts,
        policy_model_ids=policy_model_ids,
    )


def make_routed_reranked_retrieval_submission(
    output_csv: Path,
    competition_dir: Path,
    schema_path: Path,
    base_config: RetrievalConfig,
    routing_map_path: Path,
    reranker_model_path: Path,
    reranker_internal_top_k: int = 120,
    reranker_oracc_cap: int = 25,
    profiles_cache_path: Path | None = None,
    routing_telemetry_path: Path | None = None,
    verify_determinism: bool = False,
) -> RetrievalSubmissionResult:
    from routing_engine import ROUTE_TO_POLICY, choose_policy, load_routing_map, profile_source
    from structural_profile import build_token_frequency

    loaded = load_routing_map(routing_map_path)
    reranker_path = reranker_model_path.resolve()
    reranker = load_linear_reranker(reranker_path)
    reranker_sha = sha256(reranker_path.read_bytes()).hexdigest()

    routing_thresholds = dict(loaded.routing_map.get("thresholds", {}))
    routed_model_id = (
        f"RoutedRetrievalReranked(base={base_config.model_id()},"
        f"routing_sha={loaded.sha256[:12]},reranker_sha={reranker_sha[:12]})"
    )

    data = load_deep_past_competition(competition_dir, schema_path)
    train_src = data.train[data.schema.train_text_col].astype(str).fillna("").tolist()
    train_tgt = data.train[data.schema.train_target_col].astype(str).fillna("").tolist()
    test_src = data.test[data.schema.test_text_col].astype(str).fillna("").tolist()
    token_freq = build_token_frequency(train_src)

    route_to_params = {
        route: _route_policy_params(loaded.routing_map, route)
        for route in ROUTE_TO_POLICY
    }
    policy_configs = {
        policy_name: build_policy_config(
            base_config,
            policy_name=policy_name,
            policy_params=route_to_params[route_name],
            routing_thresholds=routing_thresholds,
        )
        for route_name, policy_name in ROUTE_TO_POLICY.items()
    }

    policy_model_ids: dict[str, str] = {}
    policy_preds: dict[str, list[str]] = {}
    policy_debug: dict[str, list[dict]] = {}
    for policy_name, pcfg in policy_configs.items():
        model = CanonicalRetrievalTranslator(config=pcfg)
        model.fit(train_src=train_src, train_tgt=train_tgt)
        preds, dbg = model.predict(test_src, return_debug=True)
        if verify_determinism:
            verify = model.predict(test_src, return_debug=False)
            if verify != preds:
                raise RuntimeError(f"Determinism check failed for reranked routed policy: {policy_name}")
        policy_model_ids[policy_name] = pcfg.model_id()
        policy_preds[policy_name] = preds
        policy_debug[policy_name] = dbg

    internal_debug = policy_debug["internal_only"]
    computed_profile_rows: list[dict] = []
    for i, src in enumerate(test_src):
        features, labels = profile_source(
            src,
            token_freq=token_freq,
            profile_thresholds=loaded.profile_thresholds,
        )
        dbg = internal_debug[i]
        internal_top_score = float(dbg.get("internal_top_cosine", dbg.get("global_score", 0.0)))
        raw_gap = dbg.get("internal_gap", None)
        internal_gap = None if raw_gap is None else float(raw_gap)
        decision = choose_policy(
            labels=labels,
            internal_top_score=internal_top_score,
            internal_gap=internal_gap,
            routing_map=loaded.routing_map,
        )
        computed_profile_rows.append(
            {
                "row_index": int(i),
                "source_sha256": _source_sha256(src),
                "features": features,
                "labels": labels,
                "route": decision["route"],
                "policy_name": decision["policy_name"],
                "policy_params": decision["policy_params"],
                "rationale": decision["rationale"],
                "internal_top_score": internal_top_score,
                "internal_gap": internal_gap,
            }
        )

    profile_rows = computed_profile_rows
    if profiles_cache_path is not None:
        if profiles_cache_path.exists():
            cached = _load_profiles_cache(profiles_cache_path, test_src=test_src, routing_map_sha256=loaded.sha256)
            if cached is not None:
                profile_rows = cached
        if profile_rows is computed_profile_rows:
            _write_profiles_cache(profiles_cache_path, rows=profile_rows, routing_map_sha256=loaded.sha256)

    candidate_config = build_policy_config(
        base_config,
        policy_name="hybrid",
        policy_params={
            "top_k": int(max(1, reranker_internal_top_k)),
            "oracc_cap": int(max(0, reranker_oracc_cap)),
        },
        routing_thresholds=routing_thresholds,
    )
    candidate_model = CanonicalRetrievalTranslator(config=candidate_config)
    candidate_model.fit(train_src=train_src, train_tgt=train_tgt)

    q_norm = [candidate_model._norm(x) for x in test_src]
    q_mat = candidate_model.vectorizer.transform(q_norm)
    sims = cosine_similarity(q_mat, candidate_model.train_matrix)

    final_preds: list[str] = []
    final_debug: list[dict] = []
    route_counts: dict[str, int] = {}
    unknown_routes = set()
    selected_by_text_id: dict[str, set[int]] = {}

    for i, prow in enumerate(profile_rows):
        policy_name = str(prow.get("policy_name", ""))
        route_name = str(prow.get("route", ""))
        if route_name not in ROUTE_TO_POLICY:
            unknown_routes.add(route_name)
            continue
        if policy_name not in policy_preds:
            unknown_routes.add(policy_name)
            continue

        routed_dbg = dict(policy_debug[policy_name][i])
        routed_idx = int(routed_dbg.get("chosen_idx", 0))
        internal_idx = int(policy_debug["internal_only"][i].get("chosen_idx", routed_idx))
        test_row = data.test.iloc[i]
        test_id = test_row.get(data.schema.test_id_col, None) if data.schema.test_id_col in data.test.columns else None
        text_id = test_row.get("text_id", None) if "text_id" in data.test.columns else None
        line_start = test_row.get("line_start", None) if "line_start" in data.test.columns else None
        line_end = test_row.get("line_end", None) if "line_end" in data.test.columns else None
        span_len = None
        try:
            if line_start is not None and line_end is not None:
                span_len = int(line_end) - int(line_start)
        except Exception:
            span_len = None

        row = sims[i]
        external_top_cosine = None
        if candidate_model.external_indices.size > 0:
            external_top_cosine = float(np.max(row[candidate_model.external_indices]))
        candidate_pool = build_candidate_pool_top_internal_oracc(
            model=candidate_model,
            row=row,
            internal_top_k=int(max(1, reranker_internal_top_k)),
            oracc_cap=int(max(0, reranker_oracc_cap)),
            required_indices=(internal_idx, routed_idx),
            exclude_indices=(),
        )
        feat_mat = build_feature_matrix_for_candidates(
            model=candidate_model,
            query_text_raw=test_src[i],
            row=row,
            candidate_idx=candidate_pool,
        )
        rerank_scores = reranker.score_matrix(feat_mat)
        best_pos = int(np.argmax(rerank_scores))
        routed_pos_candidates = np.where(candidate_pool == routed_idx)[0]
        routed_pos = int(routed_pos_candidates[0]) if routed_pos_candidates.size > 0 else 0

        base_score_idx = FEATURE_NAMES.index("base_score")
        digits_overlap_idx = FEATURE_NAMES.index("digits_overlap")
        use_gate = bool(
            reranker.gate_prob_margin > 0.0
            or reranker.gate_base_score_drop >= 0.0
            or reranker.gate_min_pred_delta > 0.0
            or reranker.gate_numeric_digit_overlap_tol >= 0.0
        )
        gated_to_baseline = False
        gate_prob_gap = float(rerank_scores[best_pos] - rerank_scores[routed_pos])
        gate_pred_delta = float(
            reranker.delta_calibration_intercept + reranker.delta_calibration_slope * gate_prob_gap
        )
        allow_prob = True
        allow_base = True
        allow_delta = True
        allow_digits = True
        numeric_heavy = bool(
            str(dict(prow.get("labels", {})).get("numeric_density", "low")) == "high"
            or any(ch.isdigit() for ch in str(test_src[i] or ""))
        )
        best_digits = float(feat_mat[best_pos, digits_overlap_idx])
        routed_digits = float(feat_mat[routed_pos, digits_overlap_idx])
        if use_gate:
            allow_prob = gate_prob_gap >= float(max(0.0, reranker.gate_prob_margin))
            if reranker.gate_base_score_drop >= 0.0:
                best_base = float(feat_mat[best_pos, base_score_idx])
                routed_base = float(feat_mat[routed_pos, base_score_idx])
                allow_base = best_base >= (routed_base - float(reranker.gate_base_score_drop))
            if reranker.gate_min_pred_delta > 0.0:
                allow_delta = gate_pred_delta >= float(reranker.gate_min_pred_delta)
            if numeric_heavy and reranker.gate_numeric_digit_overlap_tol >= 0.0:
                allow_digits = best_digits + float(reranker.gate_numeric_digit_overlap_tol) >= routed_digits
            if not (allow_prob and allow_base and allow_delta and allow_digits):
                best_pos = routed_pos
                gated_to_baseline = True

        chosen_idx = int(candidate_pool[best_pos])

        diversity_applied = False
        diversity_prev_idx = chosen_idx
        diversity_replaced_idx = None
        text_id_key = str(text_id).strip() if text_id is not None and str(text_id).strip() else ""
        if text_id_key:
            used = selected_by_text_id.setdefault(text_id_key, set())
            if chosen_idx in used and len(candidate_pool) > len(used):
                ranked_pos = np.argsort(rerank_scores)[::-1]
                for pos in ranked_pos:
                    alt_idx = int(candidate_pool[int(pos)])
                    if alt_idx not in used:
                        best_pos = int(pos)
                        chosen_idx = alt_idx
                        diversity_applied = True
                        diversity_replaced_idx = alt_idx
                        break
            used.add(chosen_idx)

        pred = candidate_model.train_tgt[chosen_idx]
        final_preds.append(pred)

        tfidf_score = float(row[internal_idx]) if internal_idx < len(row) else 0.0
        bm25_score = float(routed_dbg.get("chosen_score", routed_dbg.get("global_score", 0.0)))
        rerank_score = float(rerank_scores[best_pos])
        baseline_rerank_score = float(rerank_scores[routed_pos])
        ranked_all_pos = np.argsort(rerank_scores)[::-1]
        selected_rank = int(np.where(ranked_all_pos == best_pos)[0][0] + 1)
        baseline_rank = int(np.where(ranked_all_pos == routed_pos)[0][0] + 1)

        top_pos = np.argsort(rerank_scores)[::-1][: min(5, len(candidate_pool))]
        top_candidates = []
        for rank, pos in enumerate(top_pos, start=1):
            cidx = int(candidate_pool[pos])
            top_candidates.append(
                {
                    "rank": int(rank),
                    "memory_idx": cidx,
                    "origin": candidate_model.train_origins[cidx],
                    "context": candidate_model.train_domains[cidx],
                    "reranker_score": float(rerank_scores[pos]),
                    "target_text": _short(candidate_model.train_tgt[cidx]),
                    "features": feature_dict_from_row(feat_mat[pos]),
                }
            )

        scorecard = {
            "candidate_a": {
                "label": "selected by TF-IDF",
                "memory_idx": int(internal_idx),
                "score": tfidf_score,
                "origin": candidate_model.train_origins[internal_idx],
                "context": candidate_model.train_domains[internal_idx],
                "target_text": _short(candidate_model.train_tgt[internal_idx]),
            },
            "candidate_b": {
                "label": "selected by BM25",
                "memory_idx": int(routed_idx),
                "score": bm25_score,
                "origin": candidate_model.train_origins[routed_idx],
                "context": candidate_model.train_domains[routed_idx],
                "target_text": _short(candidate_model.train_tgt[routed_idx]),
            },
            "candidate_c": {
                "label": "selected by Reranker (final)",
                "memory_idx": int(chosen_idx),
                "score": rerank_score,
                "origin": candidate_model.train_origins[chosen_idx],
                "context": candidate_model.train_domains[chosen_idx],
                "target_text": _short(candidate_model.train_tgt[chosen_idx]),
            },
            "top_weights": reranker.top_weight_items(top_n=5),
        }

        routed_dbg["routed_policy_name"] = policy_name
        routed_dbg["routed_route_name"] = route_name
        routed_dbg["routed_policy_params"] = dict(prow.get("policy_params", {}))
        routed_dbg["routed_rationale"] = str(prow.get("rationale", ""))
        routed_dbg["routed_labels"] = dict(prow.get("labels", {}))
        routed_dbg["routed_features"] = dict(prow.get("features", {}))
        routed_dbg["reranker_model_path"] = str(reranker_path)
        routed_dbg["reranker_model_sha256"] = reranker_sha
        routed_dbg["reranker_feature_names"] = list(reranker.feature_names)
        routed_dbg["reranker_selected_idx"] = int(chosen_idx)
        routed_dbg["reranker_selected_score"] = rerank_score
        routed_dbg["reranker_selected_rank"] = int(selected_rank)
        routed_dbg["reranker_baseline_idx"] = int(routed_idx)
        routed_dbg["reranker_baseline_score"] = float(baseline_rerank_score)
        routed_dbg["reranker_baseline_rank"] = int(baseline_rank)
        routed_dbg["reranker_gated_to_baseline"] = bool(gated_to_baseline)
        routed_dbg["reranker_gate_prob_gap"] = float(gate_prob_gap)
        routed_dbg["reranker_gate_pred_delta"] = float(gate_pred_delta)
        routed_dbg["reranker_gate_allow_prob"] = bool(allow_prob)
        routed_dbg["reranker_gate_allow_base"] = bool(allow_base)
        routed_dbg["reranker_gate_allow_pred_delta"] = bool(allow_delta)
        routed_dbg["reranker_gate_allow_digits"] = bool(allow_digits)
        routed_dbg["reranker_gate_numeric_heavy"] = bool(numeric_heavy)
        routed_dbg["reranker_gate_best_digits_overlap"] = float(best_digits)
        routed_dbg["reranker_gate_routed_digits_overlap"] = float(routed_digits)
        routed_dbg["reranker_gate_prob_margin"] = float(reranker.gate_prob_margin)
        routed_dbg["reranker_gate_base_score_drop"] = float(reranker.gate_base_score_drop)
        routed_dbg["reranker_gate_min_pred_delta"] = float(reranker.gate_min_pred_delta)
        routed_dbg["reranker_delta_calibration_slope"] = float(reranker.delta_calibration_slope)
        routed_dbg["reranker_delta_calibration_intercept"] = float(reranker.delta_calibration_intercept)
        routed_dbg["reranker_gate_digit_overlap_tol"] = float(reranker.gate_numeric_digit_overlap_tol)
        routed_dbg["reranker_pool_size"] = int(len(candidate_pool))
        routed_dbg["reranker_internal_top_k"] = int(reranker_internal_top_k)
        routed_dbg["reranker_oracc_cap"] = int(reranker_oracc_cap)
        routed_dbg["reranker_candidates"] = top_candidates
        routed_dbg["selection_scorecard"] = scorecard
        routed_dbg["query_preview"] = _short(test_src[i], max_chars=80)
        routed_dbg["test_id"] = test_id.item() if hasattr(test_id, "item") else test_id
        routed_dbg["text_id"] = text_id.item() if hasattr(text_id, "item") else text_id
        routed_dbg["line_start"] = line_start.item() if hasattr(line_start, "item") else line_start
        routed_dbg["line_end"] = line_end.item() if hasattr(line_end, "item") else line_end
        routed_dbg["line_span_len"] = span_len
        routed_dbg["oracc_top_cosine"] = external_top_cosine
        routed_dbg["internal_oracc_margin"] = (
            float(prow.get("internal_top_score", 0.0) - external_top_cosine)
            if external_top_cosine is not None
            else None
        )
        routed_dbg["diversity_text_id"] = text_id_key if text_id_key else None
        routed_dbg["diversity_applied"] = bool(diversity_applied)
        routed_dbg["diversity_prev_idx"] = int(diversity_prev_idx)
        routed_dbg["diversity_replaced_idx"] = (
            int(diversity_replaced_idx) if diversity_replaced_idx is not None else None
        )

        final_debug.append(routed_dbg)
        route_counts[route_name] = route_counts.get(route_name, 0) + 1

    if unknown_routes:
        unknown_list = ", ".join(sorted(unknown_routes))
        raise RuntimeError(f"Unknown reranked routed policies in routing decisions: {unknown_list}")
    if len(final_preds) != len(test_src):
        raise RuntimeError("Reranked routed submission failed preflight: some rows did not produce outputs.")

    if verify_determinism:
        verify_preds: list[str] = []
        verify_selected_by_text_id: dict[str, set[int]] = {}
        for i, src in enumerate(test_src):
            prow = profile_rows[i]
            policy_name = str(prow.get("policy_name", ""))
            routed_idx = int(policy_debug[policy_name][i].get("chosen_idx", 0))
            internal_idx = int(policy_debug["internal_only"][i].get("chosen_idx", routed_idx))
            row = sims[i]
            candidate_pool = build_candidate_pool_top_internal_oracc(
                model=candidate_model,
                row=row,
                internal_top_k=int(max(1, reranker_internal_top_k)),
                oracc_cap=int(max(0, reranker_oracc_cap)),
                required_indices=(internal_idx, routed_idx),
                exclude_indices=(),
            )
            feat_mat = build_feature_matrix_for_candidates(
                model=candidate_model,
                query_text_raw=src,
                row=row,
                candidate_idx=candidate_pool,
            )
            scores = reranker.score_matrix(feat_mat)
            best_pos = int(np.argmax(scores))
            routed_pos_candidates = np.where(candidate_pool == routed_idx)[0]
            routed_pos = int(routed_pos_candidates[0]) if routed_pos_candidates.size > 0 else 0
            use_gate = bool(
                reranker.gate_prob_margin > 0.0
                or reranker.gate_base_score_drop >= 0.0
                or reranker.gate_min_pred_delta > 0.0
                or reranker.gate_numeric_digit_overlap_tol >= 0.0
            )
            if use_gate:
                base_score_idx = FEATURE_NAMES.index("base_score")
                digits_overlap_idx = FEATURE_NAMES.index("digits_overlap")
                prob_gap = float(scores[best_pos] - scores[routed_pos])
                pred_delta = float(
                    reranker.delta_calibration_intercept + reranker.delta_calibration_slope * prob_gap
                )
                allow_prob = prob_gap >= float(max(0.0, reranker.gate_prob_margin))
                if reranker.gate_base_score_drop >= 0.0:
                    best_base = float(feat_mat[best_pos, base_score_idx])
                    routed_base = float(feat_mat[routed_pos, base_score_idx])
                    allow_base = best_base >= (routed_base - float(reranker.gate_base_score_drop))
                else:
                    allow_base = True
                if reranker.gate_min_pred_delta > 0.0:
                    allow_delta = pred_delta >= float(reranker.gate_min_pred_delta)
                else:
                    allow_delta = True
                labels_obj = dict(prow.get("labels", {}))
                numeric_heavy = bool(
                    str(labels_obj.get("numeric_density", "low")) == "high"
                    or any(ch.isdigit() for ch in str(src or ""))
                )
                if numeric_heavy and reranker.gate_numeric_digit_overlap_tol >= 0.0:
                    best_digits = float(feat_mat[best_pos, digits_overlap_idx])
                    routed_digits = float(feat_mat[routed_pos, digits_overlap_idx])
                    allow_digits = best_digits + float(reranker.gate_numeric_digit_overlap_tol) >= routed_digits
                else:
                    allow_digits = True
                if not (allow_prob and allow_base and allow_delta and allow_digits):
                    best_pos = routed_pos
            verify_idx = int(candidate_pool[best_pos])
            if "text_id" in data.test.columns:
                verify_text_id = data.test.iloc[i].get("text_id", None)
                verify_text_key = (
                    str(verify_text_id).strip() if verify_text_id is not None and str(verify_text_id).strip() else ""
                )
            else:
                verify_text_key = ""
            if verify_text_key:
                used_verify = verify_selected_by_text_id.setdefault(verify_text_key, set())
                if verify_idx in used_verify and len(candidate_pool) > len(used_verify):
                    ranked_pos = np.argsort(scores)[::-1]
                    for pos in ranked_pos:
                        alt_idx = int(candidate_pool[int(pos)])
                        if alt_idx not in used_verify:
                            verify_idx = alt_idx
                            break
                used_verify.add(verify_idx)
            verify_preds.append(candidate_model.train_tgt[verify_idx])
        if verify_preds != final_preds:
            raise RuntimeError("Determinism check failed for retrieval_routed_reranked submission.")

    sub = data.sample_submission.copy()
    sub[data.schema.submission_target_col] = final_preds
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_csv, index=False)

    digest = sha256(output_csv.read_bytes()).hexdigest()
    section_closing_count = int(sum(1 for dbg in final_debug if str(dbg.get("query_section", "")) == "closing"))
    section_match_count = int(sum(1 for dbg in final_debug if bool(dbg.get("section_match_chosen", False))))

    evidence_rows = []
    telemetry_rows = []
    optional_cols = [data.schema.test_id_col, "text_id", "line_start", "line_end"]
    for i, dbg in enumerate(final_debug):
        prow = profile_rows[i]
        item = {
            "row_index": int(i),
            "query": test_src[i],
            "prediction": final_preds[i],
            "route": str(prow.get("route", "")),
            "policy_name": str(prow.get("policy_name", "")),
            "policy_params": dict(prow.get("policy_params", {})),
            "labels": dict(prow.get("labels", {})),
            "features": dict(prow.get("features", {})),
            "rationale": str(prow.get("rationale", "")),
            "debug": dbg,
        }
        row = data.test.iloc[i]
        for col in optional_cols:
            if col in data.test.columns:
                value = row[col]
                item[col] = value.item() if hasattr(value, "item") else value
        evidence_rows.append(item)
        selected_idx = dbg.get("reranker_selected_idx")
        baseline_idx = dbg.get("reranker_baseline_idx")
        selected_idx_int = int(selected_idx) if selected_idx is not None else None
        baseline_idx_int = int(baseline_idx) if baseline_idx is not None else None
        telemetry_item: dict[str, Any] = {
            "row_index": int(i),
            "query_preview": _short(test_src[i], max_chars=80),
            "route": str(prow.get("route", "")),
            "policy_name": str(prow.get("policy_name", "")),
            "policy_params": dict(prow.get("policy_params", {})),
            "labels": dict(prow.get("labels", {})),
            "rationale": str(prow.get("rationale", "")),
            "internal_top_score": float(prow.get("internal_top_score", 0.0)),
            "oracc_top_score": dbg.get("oracc_top_cosine"),
            "internal_oracc_margin": dbg.get("internal_oracc_margin"),
            "selected_candidate": {
                "memory_idx": selected_idx_int,
                "candidate_rank": dbg.get("reranker_selected_rank"),
                "candidate_score": dbg.get("reranker_selected_score"),
                "origin": dbg.get("chosen_origin"),
                "context": dbg.get("chosen_context"),
                "target_preview": _short(candidate_model.train_tgt[selected_idx_int]) if selected_idx_int is not None else None,
            },
            "baseline_candidate": {
                "memory_idx": baseline_idx_int,
                "candidate_rank": dbg.get("reranker_baseline_rank"),
                "candidate_score": dbg.get("reranker_baseline_score"),
                "origin": candidate_model.train_origins[baseline_idx_int] if baseline_idx_int is not None else None,
                "context": candidate_model.train_domains[baseline_idx_int] if baseline_idx_int is not None else None,
                "target_preview": _short(candidate_model.train_tgt[baseline_idx_int]) if baseline_idx_int is not None else None,
            },
            "reranker": {
                "selected_idx": selected_idx_int,
                "baseline_idx": baseline_idx_int,
                "gated_to_baseline": dbg.get("reranker_gated_to_baseline"),
                "predicted_delta": dbg.get("reranker_gate_pred_delta"),
                "diversity_applied": dbg.get("diversity_applied"),
                "diversity_prev_idx": dbg.get("diversity_prev_idx"),
                "diversity_replaced_idx": dbg.get("diversity_replaced_idx"),
            },
        }
        for col in optional_cols:
            if col in data.test.columns:
                value = row[col]
                telemetry_item[col] = value.item() if hasattr(value, "item") else value
        telemetry_rows.append(
            telemetry_item
        )

    evidence_payload = {
        "model_id": routed_model_id,
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "reranker_model_path": str(reranker_path),
        "reranker_model_sha256": reranker_sha,
        "reranker_internal_top_k": int(reranker_internal_top_k),
        "reranker_oracc_cap": int(reranker_oracc_cap),
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "rows": evidence_rows,
    }
    evidence_path = output_csv.with_suffix(output_csv.suffix + ".evidence.json")
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    telemetry_path = routing_telemetry_path or _default_routing_telemetry_path(output_csv)
    telemetry_payload = {
        "model_id": routed_model_id,
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "reranker_model_path": str(reranker_path),
        "reranker_model_sha256": reranker_sha,
        "reranker_internal_top_k": int(reranker_internal_top_k),
        "reranker_oracc_cap": int(reranker_oracc_cap),
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "policy_model_ids": policy_model_ids,
        "rows": telemetry_rows,
    }
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(json.dumps(telemetry_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "output_csv": str(output_csv.resolve()),
        "output_sha256": digest,
        "rows": len(sub),
        "model_id": routed_model_id,
        "retrieval_base_config": asdict(base_config),
        "reranker_candidate_config": asdict(candidate_config),
        "routing_map_path": str(loaded.path.resolve()),
        "routing_map_sha256": loaded.sha256,
        "reranker_model_path": str(reranker_path),
        "reranker_model_sha256": reranker_sha,
        "reranker_internal_top_k": int(reranker_internal_top_k),
        "reranker_oracc_cap": int(reranker_oracc_cap),
        "route_counts": route_counts,
        "section_closing_count": section_closing_count,
        "section_closing_rate": float((section_closing_count / len(final_debug)) if final_debug else 0.0),
        "section_match_count": section_match_count,
        "section_match_rate": float((section_match_count / len(final_debug)) if final_debug else 0.0),
        "policy_model_ids": policy_model_ids,
        "determinism_verified": bool(verify_determinism),
        "evidence_json": str(evidence_path.resolve()),
        "routing_telemetry_json": str(telemetry_path.resolve()),
        "profiles_cache_path": str(profiles_cache_path.resolve()) if profiles_cache_path is not None else None,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    meta_path = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return RetrievalSubmissionResult(
        output_csv=output_csv,
        metadata_json=meta_path,
        output_sha256=digest,
        rows=len(sub),
        model_id=routed_model_id,
        evidence_json=evidence_path,
        routing_json=telemetry_path,
        routing_map_path=loaded.path,
        routing_map_sha256=loaded.sha256,
        route_counts=route_counts,
        policy_model_ids=policy_model_ids,
        reranker_model_path=reranker_path,
        reranker_model_sha256=reranker_sha,
    )
