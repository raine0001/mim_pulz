from __future__ import annotations

import html
import json
import re
import sqlite3
import sys
import unicodedata
import csv
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

from flask import Blueprint, Response, jsonify, request, send_file
from functools import wraps
try:  # optional; app still runs without CORS
    from flask_cors import CORS
except ImportError:
    CORS = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "data" / "corpus" / "corpus.db"
PUBLISHED_TEXTS_CSV = PROJECT_ROOT / "data" / "raw" / "competition" / "published_texts.csv"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from mim_pulz.domain_intent import infer_dialog_domain_with_confidence
    from structural_profile import (
        ProfileThresholds,
        build_token_frequency,
        extract_source_features,
        label_features,
        route_decision,
    )

    STRUCTURAL_ROUTING_AVAILABLE = True
except Exception:
    STRUCTURAL_ROUTING_AVAILABLE = False

corpus_bp = Blueprint("corpus", __name__, url_prefix="/corpus")


def q(sql: str, params: Iterable = ()):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    con.close()
    return rows


def scalar(sql: str, params: Iterable = ()):
    rows = q(sql, params)
    return int(rows[0][0]) if rows else 0


def pct(n: int, d: int) -> float:
    return round((100.0 * n / d), 1) if d else 0.0


def safe_json_loads(raw: str | None, default: dict | None = None) -> dict:
    if not raw:
        return default or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default or {}


def normalize_lookup_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    text = text.replace(".pdf", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def json_errors(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return wrapper


_STRUCTURAL_TOKEN_FREQ: dict[str, int] | None = None
_ROUTING_MAP_CACHE: dict | None = None
_PROFILE_THRESHOLDS_CACHE: "ProfileThresholds | None" = None
_VISUAL_CANDIDATE_ROWS: list[dict] | None = None
_VISUAL_MATCH_CACHE: dict[tuple[str, str], list[dict]] = {}


def _method_policy_label(route_name: str) -> str:
    mapping = {
        "RETRIEVE_INTERNAL": "INTERNAL",
        "RETRIEVE_HYBRID": "HYBRID",
        "RETRIEVE_ORACC_FALLBACK": "FALLBACK",
        "RERANK_STRONG": "STRONG_RERANK",
    }
    return mapping.get(route_name, "INTERNAL")


def _titleish_label(key: str, value: str) -> str:
    value = str(value or "").strip().lower()
    if key == "template_type":
        table = {
            "slot_structured": "Slot-Structured",
            "narrative": "Narrative",
            "hybrid": "Hybrid",
        }
        return table.get(value, value.replace("_", " ").title())
    if key == "domain_intent":
        table = {
            "letter": "Letter",
            "legal": "Legal",
            "economic": "Economic",
            "administrative": "Administrative",
            "ritual": "Ritual",
            "unknown": "Unknown",
        }
        return table.get(value, "Unknown")
    return value.replace("_", " ").title()


def _default_profile_thresholds() -> "ProfileThresholds":
    return ProfileThresholds(
        fragment_bracket_p90=0.012,
        fragment_unknown_p90=0.087,
        formula_high_p85=0.045,
        formula_mid_p60=0.0,
        numeric_high_p85=0.044,
        pn_high_p85=0.203,
    )


def _default_routing_map() -> dict:
    return {
        "thresholds": {
            "internal_top_low": 0.5,
            "internal_top_high": 0.6,
        },
        "routes": {
            "P0": {"route": "RETRIEVE_INTERNAL", "policy_params": {"top_k": 120}},
            "P1": {"route": "RETRIEVE_HYBRID", "policy_params": {"top_k": 80}},
            "P2": {"route": "RETRIEVE_ORACC_FALLBACK", "policy_params": {"oracc_cap": 5, "gate": "low_conf"}},
            "P3": {"route": "RERANK_STRONG", "policy_params": {"stage2_pool": 80, "stage2_weight": 0.2}},
        },
    }


def _load_methodology_assets() -> tuple["ProfileThresholds", dict]:
    global _ROUTING_MAP_CACHE, _PROFILE_THRESHOLDS_CACHE
    if _ROUTING_MAP_CACHE is not None and _PROFILE_THRESHOLDS_CACHE is not None:
        return _PROFILE_THRESHOLDS_CACHE, _ROUTING_MAP_CACHE

    profile_thr = _default_profile_thresholds()
    routing_map = _default_routing_map()
    routing_path = PROJECT_ROOT / "artifacts" / "profiles" / "routing_map.json"
    if routing_path.exists():
        try:
            payload = json.loads(routing_path.read_text(encoding="utf-8"))
            candidate_map = payload.get("routing_map", payload)
            if isinstance(candidate_map, dict) and "thresholds" in candidate_map and "routes" in candidate_map:
                routing_map = candidate_map
            raw_thr = payload.get("thresholds", {})
            if all(
                k in raw_thr
                for k in (
                    "fragment_bracket_p90",
                    "fragment_unknown_p90",
                    "formula_high_p85",
                    "formula_mid_p60",
                    "numeric_high_p85",
                    "pn_high_p85",
                )
            ):
                profile_thr = ProfileThresholds(
                    fragment_bracket_p90=float(raw_thr["fragment_bracket_p90"]),
                    fragment_unknown_p90=float(raw_thr["fragment_unknown_p90"]),
                    formula_high_p85=float(raw_thr["formula_high_p85"]),
                    formula_mid_p60=float(raw_thr["formula_mid_p60"]),
                    numeric_high_p85=float(raw_thr["numeric_high_p85"]),
                    pn_high_p85=float(raw_thr["pn_high_p85"]),
                )
        except Exception:
            pass
    _PROFILE_THRESHOLDS_CACHE = profile_thr
    _ROUTING_MAP_CACHE = routing_map
    return profile_thr, routing_map


def _get_structural_token_freq() -> dict[str, int]:
    global _STRUCTURAL_TOKEN_FREQ
    if _STRUCTURAL_TOKEN_FREQ is not None:
        return _STRUCTURAL_TOKEN_FREQ
    rows = q(
        """
        SELECT text_norm
        FROM page_registry
        WHERE text_norm IS NOT NULL AND length(text_norm) > 0
        LIMIT 5000
        """
    )
    texts = [str(r["text_norm"] or "") for r in rows]
    _STRUCTURAL_TOKEN_FREQ = build_token_frequency(texts) if texts else {}
    return _STRUCTURAL_TOKEN_FREQ


def _oracc_proxy_score(internal_score: float, source_role: str, profile_labels: dict) -> float:
    score = float(internal_score)
    if profile_labels.get("fragmentation") == "fragmentary":
        score += 0.05
    if profile_labels.get("length_bucket") == "long":
        score += 0.04
    if source_role == "primary_text":
        score -= 0.06
    elif source_role == "scholarly_commentary":
        score += 0.03
    return max(0.0, min(1.5, score))


def compute_structural_intelligence(
    text: str | None,
    *,
    source_role: str,
    internal_score: float,
    internal_gap: float | None = None,
) -> dict:
    clean_text = str(text or "")
    if not STRUCTURAL_ROUTING_AVAILABLE:
        return {
            "profile": {
                "fragmentation": "Unknown",
                "formula_density": "Unknown",
                "numeric_density": "Unknown",
                "template_type": "Unknown",
                "length_bucket": "Unknown",
                "domain_intent": "Unknown",
            },
            "routing": {
                "selected_policy": "INTERNAL",
                "selected_policy_route": "RETRIEVE_INTERNAL",
                "internal_score": round(float(internal_score), 3),
                "oracc_score": round(float(internal_score), 3),
                "thresholds_applied": {"internal_low": 0.5, "internal_high": 0.6},
                "rationale": "Structural profiler unavailable; defaulting to internal policy.",
            },
            "method_explanation": "Structural profiling unavailable in this runtime.",
        }

    profile_thr, routing_map = _load_methodology_assets()
    token_freq = _get_structural_token_freq()
    features = extract_source_features(clean_text, token_freq=token_freq)
    labels = label_features(features, profile_thr)
    domain_intent, domain_confidence, _ = infer_dialog_domain_with_confidence(clean_text)
    labels["domain_intent"] = domain_intent
    labels["domain_intent_confidence"] = float(domain_confidence)

    decision = route_decision(
        labels=labels,
        internal_top_score=float(internal_score),
        internal_gap=None if internal_gap is None else float(internal_gap),
        routing_map=routing_map,
    )
    route_name = str(decision.get("route", "RETRIEVE_INTERNAL"))
    policy = _method_policy_label(route_name)
    oracc_score = _oracc_proxy_score(float(internal_score), source_role, labels)
    thresholds = routing_map.get("thresholds", {})
    internal_low = float(thresholds.get("internal_top_low", 0.5))
    internal_high = float(thresholds.get("internal_top_high", 0.6))

    profile = {
        "fragmentation": _titleish_label("fragmentation", labels.get("fragmentation", "unknown")),
        "formula_density": _titleish_label("formula_density", labels.get("formula_density", "unknown")),
        "numeric_density": _titleish_label("numeric_density", labels.get("numeric_density", "unknown")),
        "template_type": _titleish_label("template_type", labels.get("template_type", "unknown")),
        "length_bucket": _titleish_label("length_bucket", labels.get("length_bucket", "unknown")),
        "domain_intent": _titleish_label("domain_intent", domain_intent),
    }

    method_explanation = (
        f"This text is {profile['fragmentation'].lower()} with {profile['formula_density'].lower()} formula density. "
        f"Internal confidence was {float(internal_score):.2f} against thresholds {internal_low:.2f}/{internal_high:.2f}, "
        f"so policy {policy} was selected."
    )

    return {
        "profile": profile,
        "raw_features": {
            "src_tokens": round(float(features.get("src_tokens", 0.0)), 2),
            "digit_ratio": round(float(features.get("digit_ratio", 0.0)), 4),
            "bracket_ratio": round(float(features.get("bracket_ratio", 0.0)), 4),
            "repeat_trigram_rate": round(float(features.get("repeat_trigram_rate", 0.0)), 4),
        },
        "routing": {
            "selected_policy": policy,
            "selected_policy_route": route_name,
            "internal_score": round(float(internal_score), 3),
            "oracc_score": round(float(oracc_score), 3),
            "thresholds_applied": {
                "internal_low": round(internal_low, 3),
                "internal_high": round(internal_high, 3),
            },
            "rationale": str(decision.get("rationale", "")),
            "policy_params": dict(decision.get("policy_params", {})),
            "domain_intent_confidence": round(float(domain_confidence), 3),
        },
        "method_explanation": method_explanation,
    }


def snippet(text: str, term: str, window: int = 120) -> str:
    if not text:
        return ""
    tl = text.lower()
    idx = tl.find(term.lower())
    if idx < 0:
        return (text[:window] + "...") if len(text) > window else text
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    raw = prefix + text[start:end] + suffix
    return highlight(raw, term)


def highlight(snippet_text: str, term: str) -> str:
    if not snippet_text or not term:
        return snippet_text
    return re.sub(re.escape(term), lambda m: f"<<{m.group(0)}>>", snippet_text, flags=re.IGNORECASE)


def normalize_marker(marker: str) -> str:
    # Fold accents so li-tù-la and li-tu-la both work
    return unicodedata.normalize("NFKD", marker).encode("ascii", "ignore").decode("ascii").lower()


def normalize_ref(ref: str) -> str:
    ref = " ".join((ref or "").strip().split()).lower()
    # OCR tails: 2g -> 29, 3g -> 39
    ref = re.sub(r"\b([0-9])g\b", r"\g<1>9", ref)
    return ref


def normalize_anchor(anchor: str) -> str:
    anchor = strip_markers(anchor or "")
    anchor = re.sub(r"\s+", " ", anchor).strip()
    return anchor


def best_window_excerpt(text: str, anchor: str, window_chars: int = 1200) -> tuple[str, float]:
    """
    Return (excerpt, score). Uses token overlap to find the best matching region.
    Works when OCR/newlines/punctuation differ.
    """
    if not text:
        return ("", 0.0)
    if not anchor or len(anchor.strip()) < 12:
        excerpt = text[:900] + ("..." if len(text) > 900 else "")
        return (excerpt, 0.0)

    def tokset(s: str) -> set[str]:
        s = strip_markers(s).replace("\\n", " ").replace("\n", " ")
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        s = " ".join(s.split())
        toks = [t for t in s.split() if len(t) >= 3]
        return set(toks[:80])

    anchor_set = tokset(anchor)
    if not anchor_set:
        excerpt = text[:900] + ("..." if len(text) > 900 else "")
        return (excerpt, 0.0)

    step = max(200, window_chars // 3)
    best_score = 0.0
    best_start = 0
    best_end = min(len(text), window_chars)

    for start in range(0, len(text), step):
        end = min(len(text), start + window_chars)
        chunk = text[start:end]
        chunk_set = tokset(chunk)
        if not chunk_set:
            continue
        score = len(anchor_set & chunk_set) / max(1, len(anchor_set))
        if score > best_score:
            best_score = score
            best_start = start
            best_end = end
        if best_score >= 0.8:
            break

    excerpt = text[best_start:best_end]
    prefix = "..." if best_start > 0 else ""
    suffix = "..." if best_end < len(text) else ""
    return (prefix + excerpt + suffix, best_score)


# -------- Evidence summarization helpers --------
def translit_score(text: str) -> int:
    t = (text or "").lower()
    score = 0
    for pat in ("um-ma", "qi-bi", "qí-bi", "dumu", "kù.babbar", "ku.babbar", "an.na", "kišib", "kisib", "igi"):
        if pat in t:
            score += 1
    if re.search(r"\bvs\.\s*\d|\brs\.\s*\d|\[\d+\]", t):
        score += 2
    return score

def strip_markers(s: str) -> str:
    """Remove <<highlight>> markers and normalize whitespace."""
    if not s:
        return ""
    s = re.sub(r"<<(.+?)>>", r"\1", s)
    s = " ".join(s.strip().split())
    return s


BAD_PATTERNS = [
    r"\bPIHANS\b",
    r"\bed\.\b",
    r"\bVolume\b",
    r"\bVeenhof\b",
    r"\bseen\b",
    r"\bgesehen\b",
    r"\bwie er\b",
    r"\bMatou[s?]\b",
    r"\bvgl\.\b",
    r"\bcf\.\b",
    r"\bFigure\b",
    r"\bAbb\.\b",
    r"\bpage\b",
    r"\bp\.\s*\d+",
]


def is_biblioish(s: str) -> bool:
    t = norm_key(s)
    if len(t.split()) < 6:
        return True
    for pat in BAD_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False

# Simple heuristic for language detection (English vs likely German)
GERMAN_MARKERS = {
    " der ", " die ", " das ", " und ", " ist ", " nicht ", " nur ",
    " sich ", " wird ", " eine ", " eines ", " handelt ", " kein ",
    " diese ", " dieser ", " wurde ", " zeigen ", " zeigt ",
}


def looks_english(s: str) -> bool:
    """
    Heuristic: treat as English if it contains mostly ASCII words
    and does not contain common German function words.
    """
    if not s:
        return False
    sl = s.lower()
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii > max(3, len(s) * 0.05):
        return False
    for g in GERMAN_MARKERS:
        if g in f" {sl} ":
            return False
    return True


def is_fragment_or_quote(s: str) -> bool:
    """
    Detect line fragments, ellipses, or citation-heavy text
    that should not appear verbatim in summaries.
    """
    if not s:
        return True
    if "..." in s:
        return True
    if re.search(r"\b(ATHE|BIN|TTC|TC|CAD)\b", s):
        return True
    if re.search(r"\d+:\d+", s):  # line refs like 27:19
        return True
    return False


def classify_key_point(text: str, source: dict) -> str:
    t = (text or "").lower()
    pdf = (source.get("pdf_name") or "").lower()
    if re.search(r"\b(siegel|seal|kisib|siegelabrollung)\b", t, flags=re.IGNORECASE):
        return "Seal impression"
    if "cad" in pdf:
        return "Lexical parallel"
    if re.search(r"\b(ATHE|BIN|TTC|TC|ICK)\b", text):
        return "Primary text"
    if source.get("doc_type") in ("commentary", "article"):
        return "Scholarly interpretation"
    return "Textual note"

def classify_keypoint(text: str, citations: list[dict]) -> tuple[str, str]:
    """
    Returns (kind, label).
    kind is a short machine tag, label is human-readable.
    """
    t = (text or "").lower()
    t_norm = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    pdfs = " ".join([(c.get("pdf_name") or "").lower() for c in (citations or [])])

    if "cad" in pdfs:
        return "lexical", "Lexical reference (dictionary)"
    if "afo_realien" in pdfs or "realien" in pdfs:
        return "reference", "Reference / realia index"

    if re.search(r"\b(ick|cct|kt\s|akt|bin|ttc|tc\s)\b", t_norm):
        if re.search(r"\b(hulle|huelle|tafel|fragment|bruchstuck|bruchstueck|join|zugehorig|zugehoerig|abschrift)\b", t_norm):
            return "reconstruction", "Document join / reconstruction"
        return "primary_text", "Primary text citation"

    if re.search(r"\b(chronolog|dated|date|eponym|monatsname|month|woche|limu)\b", t_norm):
        return "chronology", "Chronology / dating"

    if re.search(r"\b(owes|owe|debt|loan|interest|pay|paid|payment|due)\b", t_norm):
        return "debt_payment", "Debt / payment record"

    if re.search(r"\b(plaintiff|evidence|oath|witness|court|trial|case|settlement)\b", t_norm):
        return "legal_procedure", "Legal procedure / evidentiary note"

    if re.search(r"\b(archive|archiv|familie|family|son of|dumu)\b", t_norm):
        return "prosopography", "People / archive context"

    return "commentary", "Scholarly interpretation / note"


def extract_themes_and_keywords(key_points: list[dict], combined_text: str) -> tuple[list[str], list[str]]:
    """
    Return (themes, keywords).
    themes: English phrases used in the summary.
    keywords: short chips for UI filtering.
    """
    tb = (combined_text or "").lower()
    tb_norm = unicodedata.normalize("NFKD", tb).encode("ascii", "ignore").decode("ascii")

    scores: dict[str, int] = {}
    chips: set[str] = set()

    def bump(theme: str, chip: str | None, n: int):
        if n <= 0:
            return
        scores[theme] = scores.get(theme, 0) + n
        if chip:
            chips.add(chip)

    n_join = len(
        re.findall(
            r"\b(hulle|huelle|tafel|fragment|bruchstuck|bruchstueck|join|zugehorig|zugehoerig|tablet|envelope)\b",
            tb_norm,
            re.IGNORECASE,
        )
    )
    bump("the relationship between tablets and their associated envelopes (joins, fragments, reconstructions)", "joins", n_join)

    n_copy = len(re.findall(r"\b(abschrift|copy|copie|excerpt)\b", tb_norm, re.IGNORECASE))
    bump("whether passages are copies or excerpts of earlier originals", "copies", n_copy)

    n_chron = len(
        re.findall(
            r"\b(chronolog|dated|date|eponym|monatsname|month|woche|limu)\b",
            tb_norm,
            re.IGNORECASE,
        )
    )
    bump("chronology and dating (eponyms, month names, dated attestations)", "chronology", n_chron)

    n_people = len(
        re.findall(r"\b(archiv|archive|familie|family|dumu|son of|daughter)\b", tb_norm, re.IGNORECASE)
    )
    bump("archive and people context (prosopography, family links, named individuals)", "people", n_people)

    n_money = len(
        re.findall(r"\b(silver|gold|tin|mina|shekel|ku\\.babbar|an\\.na)\b", tb_norm, re.IGNORECASE)
    )
    bump("references to amounts and commodities (silver/gold/tin, measures)", "money", n_money)

    n_debt = len(
        re.findall(r"\b(owes|owe|debt|loan|interest|pay|paid|payment|due)\b", tb_norm, re.IGNORECASE)
    )
    n_debt += len(
        re.findall(r"\b(ku\\.babbar|gin|ma-na|mina|an\\.na)\b", tb_norm, re.IGNORECASE)
    )
    bump("debts and payments (owed amounts, repayment, interest)", "debt", n_debt)

    n_legal = len(
        re.findall(r"\b(plaintiff|evidence|oath|witness|court|trial|case|settlement)\b", tb_norm, re.IGNORECASE)
    )
    bump("legal procedure and evidentiary framing", "legal", n_legal)

    n_credit = len(
        re.findall(r"\b(creance|claim|debt|loan|credit|documents\\s+concerning)\b", tb_norm, re.IGNORECASE)
    )
    bump("documents concerning claims, debts, or credits", "credits", n_credit)

    kinds = [kp.get("kind") for kp in key_points if kp.get("kind")]
    if "lexical" in kinds:
        bump("lexical/dictionary evidence used to support readings", "lexicon", 2)
    if "reference" in kinds:
        bump("reference/index material supporting cross-links", "reference", 2)

    n_unc = len(
        re.findall(r"\b(unsicher|uncertain|debated|question|not secure|alternativ)\b", tb_norm, re.IGNORECASE)
    )
    bump("uncertainty in reconstruction and interpretation (alternative readings)", "uncertainty", n_unc)

    themes = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if scores[k] > 0][:3]

    if re.search(r"\b(ick|cct|akt|kt\s|bin|ttc|tc\s)\b", tb_norm, re.IGNORECASE):
        chips.add("primary sigla")

    keywords = sorted(chips)[:10]
    return themes, keywords


def detect_context_type(text_blob: str) -> str:
    if re.search(r"\b(siegel|seal|kisib|siegelabrollung)\b", text_blob, re.IGNORECASE):
        return "seal"
    if re.search(r"\b(silver|owe|pay|interest)\b", text_blob, re.IGNORECASE):
        return "transaction"
    if re.search(r"\b(letter|um-ma|qi-?bi)\b", text_blob, re.IGNORECASE):
        return "letter"
    return "general"


def best_sentence(snippet: str) -> str:
    s = strip_markers(snippet).replace("\\n", " ").replace("\n", " ")
    s = " ".join(s.split())
    if not s:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", s)

    candidates = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if is_biblioish(p):
            continue
        score = len(p)
        if re.search(r"\b(silver|tin|textile|merchant|owes|interest|witness)\b", p, re.I):
            score += 40
        p_norm = unicodedata.normalize("NFKD", p).encode("ascii", "ignore").decode("ascii")
        if re.search(r"\b(KU\\.BABBAR|AN\\.NA|DUMU|KISIB|um-ma)\b", p_norm):
            score += 30
        candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1][:260]

    return (parts[0] if parts else s)[:260]


def norm_key(s: str) -> str:
    """Normalize a statement for dedupe (lower + ascii fold + remove punctuation-ish)."""
    s = strip_markers(s).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = " ".join(s.split())
    return s


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity for quick near-duplicate detection."""
    ta = set(norm_key(a).split())
    tb = set(norm_key(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


GERMAN_EXPLANATIONS = {
    "es handelt sich um eine sehr einfach ausgefuhrte be'ulatum-urkunde": "This appears to be a very simply executed be'ulatum document.",
    "es handelt sich um eine sehr einfach ausgefuehrte be'ulatum-urkunde": "This appears to be a very simply executed be'ulatum document.",
    "nicht wirklich um zwei belege": "not actually two separate attestations",
    "hulle": "envelope",
    "huelle": "envelope",
    "tafel": "tablet",
}


def build_human_explanation(summary_text: str, key_points: list[dict], keywords: list[str]) -> dict:
    explanation: dict[str, object] = {}
    explanation_by_keyword: dict[str, dict] = {}

    explanation["what_you_are_looking_at"] = (
        "Cited Old Assyrian lines preserved in fragmentary or reconstructed form."
    )

    if "joins" in keywords:
        join_expl = {
            "what_you_are_looking_at": (
                "Evidence about whether a clay tablet and a clay envelope belong "
                "to the same original document package."
            ),
            "main_claim": (
                "The sources argue that the cited tablet and envelope are not two separate documents, "
                "but parts of a single original record that was later published separately."
            ),
            "why_it_matters": (
                "If the tablet and envelope are correctly joined, information preserved on the envelope "
                "(such as seals, witnesses, or metadata) can clarify how the tablet text should be interpreted."
            ),
            "uncertainty": (
                "The join is based on fragmentary evidence and scholarly reconstruction, "
                "so alternative interpretations remain possible."
            ),
        }
        explanation_by_keyword["joins"] = join_expl
        explanation.update(join_expl)

    if "seal" in keywords or re.search(r"\b(siegel|seal|kisib|siegelabrollung)\b", summary_text, re.IGNORECASE):
        seal_expl = {
            "what_you_are_looking_at": (
                "Evidence about seal impressions associated with Old Assyrian documents."
            ),
            "main_claim": (
                "The sources focus on identifying individual seals, their owners, and how fragmentary impressions "
                "should be reconstructed across related tablets and envelopes."
            ),
            "why_it_matters": (
                "Seals and their impressions authenticate documents and can link texts to people, offices, or families."
            ),
            "uncertainty": (
                "Seal evidence is often fragmentary, and reconstructions can depend on editorial judgment."
            ),
        }
        explanation_by_keyword["seal"] = seal_expl
        explanation.update(seal_expl)

    if "chronology" in keywords:
        explanation_by_keyword["chronology"] = {
            "what_you_are_looking_at": "Evidence showing how dates and eponyms anchor the document sequence.",
            "main_claim": "The cited material is used to fix chronology (month names, eponyms, or dated attestations).",
            "why_it_matters": "Dating affects how documents are grouped and interpreted across archives.",
            "uncertainty": "Chronological reconstruction can vary when attestations are sparse or indirect.",
        }

    if "legal" in keywords:
        explanation_by_keyword["legal"] = {
            "what_you_are_looking_at": "Discussion of legal or procedural framing in Old Assyrian records.",
            "main_claim": "The passage is used to characterize legal procedure, evidence, or oath context.",
            "why_it_matters": "Legal framing changes how obligations, witnesses, and authority are understood.",
            "uncertainty": "Legal interpretations can depend on formulaic language and editorial choices.",
        }

    if "copies" in keywords:
        explanation["copy_note"] = (
            "Some passages are identified as copies or excerpts of earlier originals, "
            "rather than independent documents."
        )
        explanation_by_keyword["copies"] = {
            "what_you_are_looking_at": "Evidence about whether passages are copies or excerpts.",
            "main_claim": "Some attestations are treated as copies of earlier originals.",
            "why_it_matters": "Copy status affects whether evidence counts as independent attestation.",
            "uncertainty": "Copy identification is often inferred rather than explicit.",
        }

    if "people" in keywords:
        explanation_by_keyword["people"] = {
            "what_you_are_looking_at": "Prosopographic context (named individuals and family links).",
            "main_claim": "Named individuals are used to connect documents or reconstruct relationships.",
            "why_it_matters": "Person-based links help align fragments and archives.",
            "uncertainty": "Names can be shared or ambiguous across sources.",
        }

    if "reference" in keywords or "lexicon" in keywords:
        explanation_by_keyword["reference"] = {
            "what_you_are_looking_at": "Reference material supporting cross-links between sources.",
            "main_claim": "Indexes/dictionaries are used to support readings or locate attestations.",
            "why_it_matters": "Reference hubs improve traceability across editions.",
            "uncertainty": "Reference entries summarize rather than quote the full text.",
        }

    translated_notes: list[str] = []
    for kp in key_points:
        text_raw = kp.get("text", "") or ""
        text_norm = unicodedata.normalize("NFKD", text_raw).encode("ascii", "ignore").decode("ascii").lower()
        for de, en in GERMAN_EXPLANATIONS.items():
            if de in text_norm:
                translated_notes.append(en)
    if translated_notes:
        unique = []
        for t in translated_notes:
            if t not in unique:
                unique.append(t)
        explanation["translated_note"] = "Key phrases in the sources describe this as: " + "; ".join(unique)

    explanation["human_explanation_by_keyword"] = explanation_by_keyword
    return explanation


def extract_join_entities(text: str) -> dict:
    t = text or ""
    t_norm = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    out = {"tablet_refs": [], "envelope_refs": [], "names": [], "mentions_copy": False}

    refs = re.findall(r"\b(ICK\s+[IVXLC]+\s+\d+[a-z]?)\b", t_norm, flags=re.IGNORECASE)
    norm_refs = []
    for r in refs:
        r = re.sub(r"^ick", "ICK", r, flags=re.IGNORECASE)
        r = re.sub(r"\b([ivxlc]+)\b", lambda m: m.group(1).upper(), r)
        norm_refs.append(r)
    refs = norm_refs

    for r in refs:
        m_tafel = re.search(re.escape(r) + r".{0,30}\(Tafel\)", t_norm, flags=re.IGNORECASE)
        m_hulle = re.search(re.escape(r) + r".{0,30}\((?:Hulle|Huelle)\)", t_norm, flags=re.IGNORECASE)
        if m_tafel:
            out["tablet_refs"].append(r)
        elif m_hulle:
            out["envelope_refs"].append(r)

    if "(Tafel)" in t_norm and not out["tablet_refs"]:
        out["tablet_refs"] = refs[:1]
    if re.search(r"\((?:Hulle|Huelle)\)", t_norm, flags=re.IGNORECASE) and not out["envelope_refs"]:
        out["envelope_refs"] = refs[1:2] if len(refs) > 1 else refs[:1]

    names = re.findall(r"\b([uU]-[A-Za-z]+|[A-Z][a-z]+-[A-Za-z]+)\b", t_norm)
    out["names"] = list(dict.fromkeys(names))[:4]
    out["mentions_copy"] = bool(re.search(r"\b(Abschrift|Kopie|copy)\b", t_norm, flags=re.IGNORECASE))

    out["tablet_refs"] = list(dict.fromkeys(out["tablet_refs"]))[:3]
    out["envelope_refs"] = list(dict.fromkeys(out["envelope_refs"]))[:3]
    return out


def extract_money_facts(text: str) -> list[str]:
    t_norm = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii").lower()
    facts = []
    if re.search(r"\b(ma-na|mina)\b", t_norm):
        facts.append("a mina-denominated amount")
    if re.search(r"\b(gin|shekel)\b", t_norm):
        facts.append("a shekel-denominated amount")
    if re.search(r"\b(ku\\.babbar|silver)\b", t_norm):
        facts.append("silver")
    if re.search(r"\b(gold)\b", t_norm):
        facts.append("gold")
    if re.search(r"\b(an\\.na|tin)\b", t_norm):
        facts.append("tin")
    return facts


def extract_summary_facts(text: str) -> dict:
    t_norm = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii")
    names = re.findall(r"\b([uU]-[A-Za-z]+|[A-Z][a-z]+-[A-Za-z]+)\b", t_norm)
    names = list(dict.fromkeys(names))[:3]
    money = extract_money_facts(text)
    return {"names": names, "money": money}


def extract_series_refs(text: str) -> list[str]:
    t_norm = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii")
    refs: list[str] = []
    patterns = [
        r"\b(?:ICK|CCT|AKT|BIN|TTC|TC|KTS?|KT)\s+[IVXLC0-9]+(?:\s+[0-9]+[a-z]?)?\b",
        r"\b(?:ATHE|POAT|VAT)\s+[0-9]+[a-z]?\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t_norm, flags=re.IGNORECASE):
            raw = " ".join(m.group(0).split())
            parts = raw.split(" ")
            if not parts:
                continue
            parts[0] = parts[0].upper()
            if len(parts) >= 2 and re.fullmatch(r"[ivxlc]+", parts[1], flags=re.IGNORECASE):
                parts[1] = parts[1].upper()
            cleaned = " ".join(parts)
            if cleaned not in refs:
                refs.append(cleaned)
    return refs[:4]


def extract_amount_phrases(text: str) -> list[str]:
    t_norm = unicodedata.normalize("NFKD", (text or "")).encode("ascii", "ignore").decode("ascii")
    out: list[str] = []
    for m in re.finditer(r"\b\d+(?:[\.,]\d+)?\s*(?:ma-na(?:-im)?|mina|gin(?:\.ta)?|shekel(?:s)?)\b", t_norm, flags=re.IGNORECASE):
        phrase = " ".join(m.group(0).split())
        window = t_norm[max(0, m.start() - 24): min(len(t_norm), m.end() + 24)].lower()
        commodity = ""
        if re.search(r"\b(ku\.?babbar|silver)\b", window):
            commodity = " silver"
        elif re.search(r"\b(gold)\b", window):
            commodity = " gold"
        elif re.search(r"\b(an\.?na|tin)\b", window):
            commodity = " tin"
        anchored = phrase + commodity
        if anchored not in out:
            out.append(anchored)
    return out[:3]


def format_human_list(items: list[str]) -> str:
    vals = [x for x in items if x]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def format_commodity_facts(money: list[str]) -> str:
    if not money:
        return "Not detected"
    metal = next((m for m in money if m in ("silver", "gold", "tin")), None)
    unit = next((m for m in money if "mina" in m or "shekel" in m), None)
    if unit:
        unit = unit.replace("a ", "")
    if metal and unit:
        return f"{metal.title()} ({unit})"
    if metal:
        return metal.title()
    return ", ".join(money[:2])


def infer_translation_doc_type(doc_type: str | None, keywords: list[str], text: str) -> str:
    if "debt" in keywords or "credits" in keywords:
        return "Debt / payment record"
    if (doc_type or "").lower() == "legal":
        return "Legal document"
    if (doc_type or "").lower() == "letter":
        return "Letter"
    if detect_context_type(text or "") == "seal":
        return "Seal / envelope"
    return DOC_LABEL.get(doc_type, "Unclassified page")


def translation_confidence_label(text: str, citations: list[str] | None) -> str:
    score = 0
    if looks_like_transliteration(text or ""):
        score += 1
    citations_count = len(citations or [])
    if citations_count >= 3:
        score += 1
    if citations_count >= 6:
        score += 1
    if score >= 2:
        return "Medium-High"
    if score == 1:
        return "Medium"
    return "Low"


def build_context_capsule(text: str, doc_type: str | None, keywords: list[str], citations: list[str] | None) -> dict:
    facts = extract_summary_facts(text)
    names = facts.get("names") or []
    money = facts.get("money") or []
    dt = (doc_type or "").lower()
    doc_label = infer_translation_doc_type(doc_type, keywords, text)
    parties = ", ".join(names) if names else "Not detected"
    commodity = format_commodity_facts(money)
    this_label = "This letter" if dt == "letter" else ("This tablet" if dt == "legal" else "This text")
    source_chain = f"Scholarly analysis -> Archival wrapper -> {this_label}"
    confidence = translation_confidence_label(text, citations)
    return {
        "document_type": doc_label,
        "parties": parties,
        "commodity": commodity,
        "source_chain": source_chain,
        "confidence": confidence,
    }

def build_summary_paragraph(combined_text: str, keywords: list[str], context_type: str) -> str:
    facts = extract_summary_facts(combined_text)
    names = facts.get("names", [])
    money = facts.get("money", [])
    refs = extract_series_refs(combined_text)
    amount_phrases = extract_amount_phrases(combined_text)
    has_uncertainty = bool(re.search(r"\b(unsicher|uncertain|debated|question|not secure|alternativ)\b", combined_text, re.I))

    if "joins" in keywords:
        if len(refs) >= 2:
            s1 = f"The retrieved lines link {refs[0]} and {refs[1]} as parts of one document package (tablet/envelope join)."
        elif refs:
            s1 = f"The retrieved lines treat {refs[0]} as a join-context text, not an isolated record."
        else:
            s1 = "The retrieved lines describe a tablet/envelope join rather than two independent records."
    elif "debt" in keywords or "credits" in keywords:
        s1 = "The retrieved lines describe a debt/payment obligation."
    elif context_type == "seal":
        s1 = "The retrieved lines identify seal evidence used to connect people and documents."
    elif context_type == "letter":
        s1 = "The retrieved lines come from correspondence and preserve concrete instructions or obligations."
    elif "chronology" in keywords:
        s1 = "The retrieved lines use dating markers (months/eponyms) to place texts in sequence."
    else:
        if refs:
            s1 = f"The retrieved lines cite {format_human_list(refs[:2])} and connect those references to one evidentiary thread."
        else:
            s1 = "The retrieved lines provide direct evidence statements for the query."

    anchors = []
    if refs:
        anchors.append(f"references: {format_human_list(refs[:2])}")
    if names:
        anchors.append(f"names: {format_human_list(names[:2])}")
    if amount_phrases:
        anchors.append(f"amounts: {format_human_list(amount_phrases[:2])}")
    elif money:
        anchors.append(f"commodity/unit markers: {format_human_list(money[:2])}")

    if anchors:
        s2 = "Text anchors in the snippets -> " + "; ".join(anchors) + "."
    else:
        s2 = "No strong name/amount anchors were extracted; use the quoted snippets for exact wording."

    if has_uncertainty or "uncertainty" in keywords or "copies" in keywords or "joins" in keywords:
        s3 = "Some lines are marked uncertain or reconstructed, so alternative readings remain possible."
    else:
        s3 = "Uncertainty markers are limited in the selected lines."

    return " ".join([s1, s2, s3]).strip()


def paraphrase_keypoint(kp: dict) -> dict:
    text = kp.get("text", "")
    kind = kp.get("kind", "")
    result = {
        "context": "Cited evidence line",
        "paraphrase": "This key point paraphrases a cited snippet; open the linked source for exact wording and context.",
        "confidence_note": "Snippet-level paraphrase",
    }

    if kind == "reconstruction":
        meta = extract_join_entities(text)
        money = extract_money_facts(text)
        pieces = []
        if meta["tablet_refs"] or meta["envelope_refs"]:
            tr = ", ".join(meta["tablet_refs"]) if meta["tablet_refs"] else "a tablet"
            er = ", ".join(meta["envelope_refs"]) if meta["envelope_refs"] else "an envelope"
            pieces.append(
                f"The claim is that {tr} (tablet) and {er} (envelope) belong to the same original document package."
            )
        if meta["names"]:
            pieces.append(
                f"A named individual appears in the argument (e.g., {', '.join(meta['names'])}), used as supporting evidence for the join."
            )
        if money:
            pieces.append(f"The note links the join to {', '.join(money)}.")
        if meta["mentions_copy"]:
            pieces.append(
                "The note also suggests one witness/line segment may represent a copy or excerpt rather than an independent document."
            )
        if not pieces:
            pieces.append(
                "This passage argues that a clay tablet and a clay envelope belong to the same original document, "
                "rather than being two separate records."
            )
        result = {
            "context": "Document reconstruction",
            "paraphrase": " ".join(pieces),
            "confidence_note": "Based on reconstruction and comparison of fragments",
        }
    elif kind == "primary_text":
        result = {
            "context": "Primary text reference",
            "paraphrase": (
                "This line identifies a specific Old Assyrian text or document by its published reference, "
                "without offering an interpretation."
            ),
            "confidence_note": "Direct citation",
        }
    elif kind == "chronology":
        result = {
            "context": "Chronology and dating",
            "paraphrase": (
                "This passage is used to establish when a document was written, using month names, eponyms, "
                "or other dating conventions."
            ),
            "confidence_note": "Depends on chronological reconstruction",
        }
    elif kind == "legal_procedure":
        result = {
            "context": "Legal or procedural context",
            "paraphrase": (
                "This passage reflects a legal situation, such as a dispute, presentation of evidence, "
                "or procedural interaction between parties."
            ),
            "confidence_note": "Contextual interpretation",
        }
    elif kind == "lexical":
        result = {
            "context": "Lexical reference",
            "paraphrase": (
                "This material is cited to explain the meaning or usage of a word or phrase, "
                "rather than to describe an event."
            ),
            "confidence_note": "Dictionary-based",
        }
    elif kind == "reference":
        result = {
            "context": "Reference or index material",
            "paraphrase": (
                "This source is used as an index or catalog to link documents or names, "
                "not to narrate an action."
            ),
            "confidence_note": "Cataloging aid",
        }
    elif kind == "debt_payment":
        result = {
            "context": "Debt / payment record",
            "paraphrase": (
                "This passage appears to describe an obligation involving an amount owed, repayment terms, "
                "or interest. It is being cited as evidence about a debt/payment relationship rather than a narrative event."
            ),
            "confidence_note": "Based on transactional vocabulary (owed/paid/interest) and money-unit markers",
        }

    return result


def build_evidence_summary(results: list[dict], top_n: int = 6) -> dict:
    """
    Build:
      - Summary paragraph (2-4 sentences)
      - Key points (deduped), each with citations (page_id/pdf/page)
      - Evidence list: quoted snippet + source info
      - Confidence: simple score based on number/quality/diversity of sources
    """
    if not results:
        return {
            "summary": "",
            "key_points": [],
            "evidence": [],
            "confidence": {"score": 0.0, "level": "Low", "reasons": ["No results."]},
        }

    ranked = sorted(results, key=lambda r: -float(r.get("evidence_weight", 0.0)))
    ranked = ranked[: max(top_n, 3)]

    points = []
    for r in ranked:
        sn = r.get("snippet") or ""
        stmt = best_sentence(sn)
        if not stmt:
            continue
        points.append(
            {
                "text": stmt,
                "source": {
                    "page_id": r.get("page_id"),
                    "pdf_name": r.get("pdf_name"),
                    "page_number": r.get("page_number"),
                    "doc_type": r.get("doc_type"),
                    "source_role": r.get("source_role"),
                    "source_role_label": r.get("source_role_label"),
                    "page_url": r.get("page_url") or source_record_url(r.get("page_id")),
                    "asset_thumb_url": r.get("asset_thumb_url"),
                    "asset_full_url": r.get("asset_full_url"),
                    "pdf_render_url": r.get("pdf_render_url"),
                },
                "evidence_weight": float(r.get("evidence_weight", 0.0)),
            }
        )

    clusters: list[dict] = []
    for p in points:
        placed = False
        for c in clusters:
            if jaccard(p["text"], c["rep"]["text"]) >= 0.55:
                c["members"].append(p)
                placed = True
                break
        if not placed:
            clusters.append({"rep": p, "members": [p]})

    key_points = []
    for c in clusters:
        rep = c["rep"]
        members = c["members"]
        citations = []
        seen = set()
        for m in members:
            s = m["source"]
            key = (s.get("page_id"), s.get("pdf_name"), s.get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(s)
        kind, label = classify_keypoint(rep.get("text", ""), citations)
        key_points.append(
            {
                "text": rep["text"],
                "citations": citations,
                "support_count": len(citations),
                "kind": kind,
                "label": label,
                "language": "en" if looks_english(rep.get("text", "")) else "non-en",
                "plain_english": paraphrase_keypoint(
                    {"text": rep.get("text", ""), "kind": kind, "label": label}
                ),
            }
        )

    key_points.sort(key=lambda kp: (-kp["support_count"], kp["text"]))
    key_points = key_points[: min(6, len(key_points))]
    combined_text = " ".join([kp.get("text", "") for kp in key_points])
    context_type = detect_context_type(combined_text)
    _themes, keywords = extract_themes_and_keywords(key_points, combined_text)
    summary = build_summary_paragraph(combined_text, keywords, context_type)
    if not summary:
        summary = "Evidence is available in the quoted sources below; no high-confidence summary sentences were extracted from the snippets."
    human_explanation = build_human_explanation(summary, key_points, keywords)

    # lightweight tag extraction from citations and key point text
    tags = []
    for kp in key_points:
        for cit in kp.get("citations", []):
            ref = cit.get("pdf_name")
            if ref and ref not in tags and len(tags) < 8:
                tags.append(ref)
        words = [w for w in re.findall(r"\b[A-Z][A-Za-z0-9\.-]{2,}\b", kp.get("text", "")) if len(w) > 2]
        for w in words:
            if w not in tags and len(tags) < 12:
                tags.append(w)

    evidence = []
    for r in ranked[: min(len(ranked), 10)]:
        evidence.append(
            {
                "quote": strip_markers(r.get("snippet") or ""),
                "source": {
                    "page_id": r.get("page_id"),
                    "pdf_name": r.get("pdf_name"),
                    "page_number": r.get("page_number"),
                    "doc_type": r.get("doc_type"),
                    "source_role": r.get("source_role"),
                    "source_role_label": r.get("source_role_label"),
                    "topics": r.get("topics", []),
                    "institutions": r.get("institutions", []),
                    "page_url": r.get("page_url") or source_record_url(r.get("page_id")),
                    "asset_thumb_url": r.get("asset_thumb_url"),
                    "asset_full_url": r.get("asset_full_url"),
                    "pdf_render_url": r.get("pdf_render_url"),
                    "structural_intelligence": r.get("structural_intelligence") or {},
                },
                "rank": {
                    "evidence_weight": float(r.get("evidence_weight", 0.0)),
                    "rank_reason": r.get("rank_reason") or r.get("fit", {}).get("breakdown"),
                },
            }
        )

    distinct_sources = {(e["source"]["pdf_name"], e["source"]["page_number"]) for e in evidence if e.get("source")}
    primaries = sum(1 for e in evidence if (e.get("source", {}).get("source_role") == "primary_text"))
    avg_weight = round(sum(float(r.get("evidence_weight", 0.0)) for r in ranked) / max(1, len(ranked)), 3)

    score = 0.0
    reasons = []
    if len(distinct_sources) >= 4:
        score += 0.4
        reasons.append(f"{len(distinct_sources)} distinct sources (+0.4)")
    elif len(distinct_sources) >= 3:
        score += 0.3
        reasons.append(f"{len(distinct_sources)} distinct sources (+0.3)")
    elif len(distinct_sources) >= 2:
        score += 0.2
        reasons.append(f"{len(distinct_sources)} distinct sources (+0.2)")
    else:
        reasons.append("Single source (no diversity)")

    if primaries >= 2:
        score += 0.35
        reasons.append(f"{primaries} primary pages (+0.35)")
    elif primaries == 1:
        score += 0.2
        reasons.append("1 primary page (+0.2)")
    else:
        reasons.append("No primary pages detected")

    score += min(0.3, avg_weight * 0.3)
    reasons.append(f"avg evidence weight {avg_weight} (+{min(0.3, avg_weight * 0.3):.2f})")

    # clamp perfection and penalize uncertainty language
    combined_text = " ".join(
        [summary]
        + [kp.get("text", "") for kp in key_points]
        + [e.get("quote", "") for e in evidence]
    )

    chips = set()
    text_blob = combined_text.lower()
    if "silver" in text_blob:
        chips.add("silver")
    if re.search(r"\bpn\b", combined_text, re.I):
        chips.add("personal obligation")
    if re.search(r"\b(ATHE|BIN|TTC|TC)\b", combined_text):
        chips.add("primary texts")
    if re.search(r"\buncertain|debated\b", combined_text, re.I):
        chips.add("textual uncertainty")
    if re.search(r"\b(uncertain|debated|not secure|unsicher)\b", combined_text, re.I):
        score = min(score, 0.85)
    else:
        score = min(score, 0.9)
    if context_type == "seal":
        score = min(score, 0.75)
    score = round(score, 2)
    level = "High" if (score >= 0.80 and len(distinct_sources) >= 3) else ("Medium" if score >= 0.45 else "Low")
    commentary_count = sum(1 for e in evidence if (e.get("source", {}).get("doc_type") == "commentary"))
    if commentary_count >= 3 and level == "High":
        level = "Medium"
    if re.search(r"\b(unsicher|question|debated|uncertain|cf\\.)\b", combined_text, re.I):
        score = min(score, 0.85)
    if primaries == 0 and level == "High":
        level = "Medium"

    return {
        "summary": summary,
        "key_points": key_points,
        "evidence": evidence,
        "selection_trace": {
            "candidate_count": len(evidence),
            "selected_page_id": evidence[0]["source"]["page_id"] if evidence else None,
            "selected_pdf_name": evidence[0]["source"]["pdf_name"] if evidence else "",
            "selected_page_number": evidence[0]["source"]["page_number"] if evidence else None,
            "selected_rank_reason": evidence[0]["rank"]["rank_reason"] if evidence else "",
            "selected_weight": evidence[0]["rank"]["evidence_weight"] if evidence else 0.0,
        },
        "tags": tags,
        "keywords": keywords,
        "human_explanation": human_explanation,
        "confidence": {"score": score, "level": level, "reasons": reasons},
    }

DOC_LABEL = {
    "legal": "Legal discussion",
    "letter": "Letter / correspondence",
    "commentary": "Scholarly commentary",
    "index": "Index / reference",
    "bibliography": "Bibliography",
    "front_matter": "Front matter",
    "unknown": "Unclassified page",
}

MARKER_LABEL = {
    "gin": "shekel unit (GIN)",
    "ma-na": "mina unit (ma-na)",
    "dumu": "son of (DUMU)",
    "igi": "witness marker (IGI)",
    "igs": "witness marker (IGI)",
}


def pretty_title(pdf_name: str) -> str:
    name = (pdf_name or "").replace(".pdf", "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def explain_rank(doc_type: str | None, citations: list | None, extras: dict | None = None):
    extras = extras or {}
    citations = citations or []
    doc_scores = {"letter": 3, "legal": 3, "commentary": 2, "index": 1, "bibliography": 1, "front_matter": 0}
    score = 0
    reasons = []
    if doc_type in doc_scores:
        score += doc_scores[doc_type]
        reasons.append(f"doc_type={doc_type} (+{doc_scores[doc_type]})")
    if citations:
        c = len(citations)
        score += c
        reasons.append(f"citations={c} (+{c})")
    for label, val in extras.items():
        if val:
            score += val
            reasons.append(f"{label} (+{val})")
    return score, ", ".join(reasons) if reasons else "default ordering"


def compute_fit(doc_type: str | None, has_signal: bool, has_cross: bool, citations_count: int, is_reference: bool):
    weight_map = {
        "letter": 1.0,
        "legal": 1.0,
        "commentary": 0.6,
        "index": 0.4,
        "bibliography": 0.4,
        "front_matter": 0.1,
    }
    score = weight_map.get(doc_type, 0.0)
    breakdown = [f"doc_type={doc_type or 'unknown'} ({score})"]
    if has_signal:
        score += 0.2
        breakdown.append("+ signal match (0.2)")
    if has_cross:
        score += 0.2
        breakdown.append("+ cross-signal (0.2)")
    if citations_count and not is_reference:
        score += 0.1
        breakdown.append("+ citations (0.1)")
    if is_reference:
        score -= 0.2
        breakdown.append("- reference penalty (0.2)")
    return round(score, 2), "; ".join(breakdown)


PRIMARY_SERIES = ("kt ", "cct", "ick", "bin", "akt", "kts", "poat", "vat")
SECONDARY_SERIES = ("afo", "pihans", "oaas", "jcs", "obo", "raq", "huca")

COMMENTARY_HINTS = (
    "pihans",
    "jaos",
    "afo",
    "diss",
    "dissertation",
    "studies",
    "review",
    "assyria and beyond",
    "old assyrian studies",
    "cdog",
    "hdo",
    "fs ",
)

REFERENCE_HINTS = (
    "bibliography",
    "index",
    "realien",
    "catalog",
    "liste",
    "lexicon",
    "cad",
)

ARCHIVAL_HINTS = (
    "envelope",
    "huelle",
    "hulle",
    "seal",
    "siegel",
    "siegelabrollung",
    "kisib",
    "witness list",
    "witnesses",
)

SOURCE_ROLE_ENUM = ("primary_text", "archival_wrapper", "scholarly_commentary")

SOURCE_ROLE_LABEL = {
    "primary_text": "Primary text",
    "archival_wrapper": "Archival wrapper",
    "scholarly_commentary": "Scholarly commentary",
}


def normalize_source_role(role: str | None) -> str:
    return role if role in SOURCE_ROLE_ENUM else "scholarly_commentary"


def get_source_role_label(role: str | None) -> str:
    return SOURCE_ROLE_LABEL[normalize_source_role(role)]


def is_translation_allowed(source_role: str | None) -> bool:
    return normalize_source_role(source_role) == "primary_text"


def build_source_asset_fields(
    page_id: str | None = None,
    pdf_name: str | None = None,
    page_number: int | None = None,
) -> dict:
    """
    Optional artifact preview/render pointers.
    These are intentionally nullable until asset plumbing is populated.
    """
    matches = find_visual_candidates(pdf_name, page_number, max_results=6)
    best = matches[0] if matches else {}
    best_url = best.get("online_transcript") or best.get("aicc_translation") or None
    visual_candidates_url = f"/corpus/page/{page_id}/visuals" if page_id else None
    visual_compare_url = f"/corpus/page/{page_id}/compare" if page_id else None
    return {
        "asset_thumb_url": None,
        "asset_full_url": None,
        "pdf_render_url": None,
        "external_visual_url": best_url,
        "external_visual_count": len(matches),
        "external_visual_candidates_url": visual_candidates_url,
        "visual_compare_url": visual_compare_url,
    }


def _load_visual_candidate_rows() -> list[dict]:
    global _VISUAL_CANDIDATE_ROWS
    if _VISUAL_CANDIDATE_ROWS is not None:
        return _VISUAL_CANDIDATE_ROWS
    rows: list[dict] = []
    if not PUBLISHED_TEXTS_CSV.exists():
        _VISUAL_CANDIDATE_ROWS = rows
        return rows
    with PUBLISHED_TEXTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip()
            publication_catalog = (row.get("publication_catalog") or "").strip()
            aliases = (row.get("aliases") or "").strip()
            note = (row.get("note") or "").strip()
            searchable = normalize_lookup_text(
                " ".join([label, publication_catalog, aliases, note])
            )
            if not searchable:
                continue
            rows.append(
                {
                    "oare_id": (row.get("oare_id") or "").strip(),
                    "online_transcript": (row.get("online transcript") or "").strip(),
                    "aicc_translation": (row.get("AICC_translation") or "").strip(),
                    "cdli_id": (row.get("cdli_id") or "").strip(),
                    "label": label,
                    "publication_catalog": publication_catalog,
                    "aliases": aliases,
                    "searchable": searchable,
                }
            )
    _VISUAL_CANDIDATE_ROWS = rows
    return rows


def _visual_score(row: dict, base_norm: str, page_text: str) -> int:
    searchable = row.get("searchable", "")
    pub = normalize_lookup_text(row.get("publication_catalog"))
    label = normalize_lookup_text(row.get("label"))
    aliases = normalize_lookup_text(row.get("aliases"))
    score = 0
    if base_norm and base_norm in searchable:
        score += 8
    if base_norm and pub.startswith(base_norm):
        score += 8
    if base_norm and base_norm in pub:
        score += 6
    if base_norm and base_norm in label:
        score += 5
    if base_norm and base_norm in aliases:
        score += 4
    if page_text and page_text in searchable:
        score += 2
    if row.get("online_transcript"):
        score += 2
    if row.get("aicc_translation"):
        score += 1
    return score


def find_visual_candidates(
    pdf_name: str | None, page_number: int | str | None, max_results: int = 6
) -> list[dict]:
    base_norm = normalize_lookup_text(pdf_name)
    if not base_norm:
        return []
    page_text = str(page_number or "").strip()
    cache_key = (base_norm, page_text)
    cached = _VISUAL_MATCH_CACHE.get(cache_key)
    if cached is not None:
        return cached[:max_results]

    rows = _load_visual_candidate_rows()
    matches: list[tuple[int, dict]] = []
    for row in rows:
        if base_norm not in row.get("searchable", ""):
            continue
        score = _visual_score(row, base_norm, page_text)
        if score <= 0:
            continue
        matches.append((score, row))
    matches.sort(
        key=lambda x: (
            -x[0],
            normalize_lookup_text(x[1].get("publication_catalog")),
            normalize_lookup_text(x[1].get("label")),
        )
    )
    dedup: list[dict] = []
    seen_urls: set[str] = set()
    for score, row in matches:
        url = row.get("online_transcript") or row.get("aicc_translation") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        rec = dict(row)
        rec["match_score"] = score
        dedup.append(rec)
        if len(dedup) >= max(20, max_results):
            break
    _VISUAL_MATCH_CACHE[cache_key] = dedup
    return dedup[:max_results]


def source_record_url(page_id: str | None) -> str:
    pid = str(page_id or "").strip()
    if not pid:
        return "/corpus"
    return f"/corpus/page/{pid}/record"


def override_doc_type(pdf_name: str | None, doc_type: str | None) -> str:
    name = (pdf_name or "").lower()
    dt = (doc_type or "").lower()
    if any(h in name for h in COMMENTARY_HINTS):
        if dt not in ("index", "bibliography", "front_matter"):
            return "commentary"
    return dt or "unknown"


def classify_ref(ref_norm: str) -> tuple[str, str]:
    r = ref_norm
    if any(r.startswith(s) for s in PRIMARY_SERIES):
        return "primary", "text"
    if any(r.startswith(s) for s in SECONDARY_SERIES):
        return "secondary", "study"
    return "unknown", ""


def looks_like_transliteration(text: str) -> bool:
    """
    Heuristic: detect transliteration/edition-like patterns.
    Helps avoid calling commentary pages "primary text".
    """
    if not text:
        return False
    t = text.lower()
    hits = 0
    if "um-ma" in t or "qi-bi" in t:
        hits += 1
    if "dumu" in t or "ku.babbar" in t or "an.na" in t:
        hits += 1
    if re.search(r"\bvs\.\s*\d|\brs\.\s*\d|\bline\s*\d", t):
        hits += 1
    if re.search(r"\[\d+\]|\b\d+\)", t):
        hits += 1
    if "siegelabrollung" in t or "kisib" in t:
        hits += 1
    return hits >= 2


def looks_like_archival_wrapper(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower()
    if re.search(r"\b(envelope|huelle|hulle)\b", t):
        return True
    hits = 0
    if re.search(r"\b(seal|siegel|siegelabrollung|kisib)\b", t):
        hits += 1
    if re.search(r"\b(witness list|witnesses|witness)\b", t):
        hits += 1
    if re.search(r"\bigi\b", t):
        hits += 1
    return hits >= 2


def compute_source_role(pdf_name: str | None, doc_type: str | None, text_norm: str | None) -> str:
    name = (pdf_name or "").lower()
    dt = (doc_type or "").lower()

    if any(h in name for h in REFERENCE_HINTS) or dt in ("index", "bibliography", "front_matter"):
        return "scholarly_commentary"
    if any(h in name for h in COMMENTARY_HINTS) or dt == "commentary":
        return "scholarly_commentary"
    if any(h in name for h in ARCHIVAL_HINTS) or looks_like_archival_wrapper(text_norm or ""):
        return "archival_wrapper"
    if dt in ("letter", "legal"):
        return "primary_text" if looks_like_transliteration(text_norm or "") else "scholarly_commentary"
    if looks_like_transliteration(text_norm or ""):
        return "primary_text"
    return "scholarly_commentary"


def build_story(bundle: dict) -> dict:
    soc = bundle.get("social", {}) or {}
    phil = bundle.get("philology", {}) or {}
    doc_type_raw = soc.get("doc_type", "unknown")
    doc_type = override_doc_type(bundle.get("pdf_name"), doc_type_raw)
    source_role = compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm"))
    source_role_label = get_source_role_label(source_role)
    asset_fields = build_source_asset_fields(
        bundle.get("page_id"),
        bundle.get("pdf_name"),
        bundle.get("page_number"),
    )
    weight_map = {
        "letter": 1.0,
        "legal": 1.0,
        "commentary": 0.6,
        "index": 0.4,
        "bibliography": 0.4,
        "front_matter": 0.1,
    }
    structural = compute_structural_intelligence(
        bundle.get("text_norm"),
        source_role=source_role,
        internal_score=float(weight_map.get(doc_type, 0.0)),
        internal_gap=None,
    )
    citations = phil.get("citations", []) or []
    divine = phil.get("divine_names", []) or []

    def ref_kind(ref: str) -> str:
        rn = normalize_ref(ref)
        if rn.startswith(("kt ", "ick", "cct", "bin", "akt", "kts", "vat", "poat")):
            return "primary"
        if rn.startswith(("afo", "pihans", "oaas", "jcs", "obo", "huca")):
            return "secondary"
        return "unknown"

    primary = sum(1 for r in citations if ref_kind(r) == "primary")
    secondary = sum(1 for r in citations if ref_kind(r) == "secondary")

    highlights = []
    if "istarzaat" in divine:
        highlights.append({"label": "Deity mentioned", "value": "Ištar-ZA.AT"})
    if "assur" in divine:
        highlights.append({"label": "Also mentions", "value": "Aššur"})
    for m in (phil.get("accounting_markers", []) or [])[:3]:
        highlights.append({"label": "Money terms", "value": MARKER_LABEL.get(m, m)})

    why = []
    if doc_type in ("legal", "letter"):
        why.append("Evidence comes from a primary document context.")
    if primary >= 3:
        why.append("Heavily cross-referenced to primary editions (strong traceability).")
    if "igi" in (phil.get("legal_markers_norm", []) or []):
        why.append("Witness markers suggest oath/legal framing.")
    if not why:
        why.append("Provides contextual support; open sources for full evidence.")

    cit_preview = []
    for r in citations[:10]:
        cit_preview.append({"ref": r, "ref_type": ref_kind(r), "doc_hint": "text"})

    snips = []
    for s in (phil.get("formula_snippets_structured", []) or [])[:6]:
        snips.append(
            {
                "topic": f"Mentions {s.get('marker','')}",
                "text": s.get("snippet", ""),
                "tags": ["Formula/Name"],
            }
        )

    return {
        "ok": True,
        "page": {
            "page_id": bundle.get("page_id"),
            "title": pretty_title(bundle.get("pdf_name")),
            "page_number": bundle.get("page_number"),
            "doc_type": doc_type,
            "page_url": source_record_url(bundle.get("page_id")),
            **asset_fields,
            "type": {
                "label": DOC_LABEL.get(doc_type, "Unclassified page"),
                "badge": source_role_label,
                "confidence": "High" if doc_type in ("legal", "letter") else "Medium",
            },
        },
        "highlights": highlights,
        "why_this_matters": why,
        "sources": {
            "total": len(citations),
            "primary": primary,
            "secondary": secondary,
        },
        "citations": cit_preview,
        "snippets": snips,
        "source_role": source_role,
        "source_role_label": source_role_label,
        **asset_fields,
        "structural_intelligence": structural,
        "raw_available": {
            "page_bundle": f"/corpus/page/{bundle.get('page_id')}?include_text=false"
        },
    }


def find_citation_matches(ref: str, limit: int = 10):
    ref_norm = normalize_ref(ref)
    patterns = [f"%{ref_norm}%"]
    raw = " ".join((ref or "").lower().strip().split())
    if raw and raw != ref_norm:
        patterns.append(f"%{raw}%")
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.page_id, p.text_norm, ph.json AS ph_json, s.json AS s_json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        LEFT JOIN social s ON s.page_id = p.page_id
        WHERE lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (patterns[0], limit * len(patterns)),
    )
    if len(patterns) > 1 and not rows:
        rows = q(
            """
            SELECT p.pdf_name, p.page_number, p.page_id, p.text_norm, ph.json AS ph_json, s.json AS s_json
            FROM philology ph
            JOIN page_registry p ON p.page_id = ph.page_id
            LEFT JOIN social s ON s.page_id = p.page_id
            WHERE lower(ph.json) LIKE ?
            LIMIT ?
            """,
            (patterns[1], limit),
        )
    results = []
    for r in rows:
        ph_payload = safe_json_loads(r["ph_json"])
        s_payload = safe_json_loads(r["s_json"])
        doc_type_raw = s_payload.get("doc_type")
        doc_type = override_doc_type(r["pdf_name"], doc_type_raw)
        source_role = compute_source_role(r["pdf_name"], doc_type, r["text_norm"])
        source_role_label = get_source_role_label(source_role)
        asset_fields = build_source_asset_fields(r["page_id"], r["pdf_name"], r["page_number"])
        is_ref = doc_type in {"index", "bibliography", "front_matter"}
        ref_type, doc_hint = classify_ref(ref_norm)
        has_signal = True  # citation matched
        has_cross = bool(ph_payload.get("formula_markers")) and bool(ph_payload.get("divine_names"))
        fit_score, fit_breakdown = compute_fit(doc_type, has_signal, has_cross, len(ph_payload.get("citations", [])), is_ref)
        score, reason = explain_rank(doc_type, ph_payload.get("citations", []))
        structural = compute_structural_intelligence(
            r["text_norm"],
            source_role=source_role,
            internal_score=float(fit_score),
            internal_gap=None,
        )
        results.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "page_id": r["page_id"],
                "ref_norm": ref_norm,
                "ref_type": ref_type,
                "doc_hint": doc_hint,
                "doc_type": doc_type,
                "source_role": source_role,
                "source_role_label": source_role_label,
                "topics": s_payload.get("topics", []),
                "institutions": s_payload.get("institutions", []),
                "citations": ph_payload.get("citations", []),
                "snippet": snippet(r["text_norm"] or "", ref),
                "page_url": source_record_url(r["page_id"]),
                **asset_fields,
                "links": {
                    "page_bundle": f"/corpus/page/{r['page_id']}?include_text=false",
                    "page_text": f"/corpus/page/{r['page_id']}?include_text=true",
                },
                "evidence_weight": fit_score,
                "rank_reason": reason,
                "fit": {"score": fit_score, "breakdown": fit_breakdown},
                "structural_intelligence": structural,
                "source": "PRIMARY"
                if source_role == "primary_text"
                else ("ARCHIVAL" if source_role == "archival_wrapper" else "COMMENTARY"),
            }
        )
    # sort by evidence_weight then citations count
    results.sort(key=lambda r: (-r.get("evidence_weight", 0), -len(r.get("citations", []))))
    return results[:limit]


def search_deity(name: str, require_citations: bool = False, limit: int = 20):
    fetch_limit = limit * 20 if require_citations else limit
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.page_id, p.text_norm, ph.json, s.json AS s_json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        LEFT JOIN social s ON s.page_id = p.page_id
        WHERE lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (f"%{name.lower()}%", fetch_limit),
    )
    results = []
    for r in rows:
        payload = safe_json_loads(r["json"])
        cits = payload.get("citations", [])
        # Clean OCR tails like 2g -> 29, 3g -> 39
        cits = [re.sub(r"\b([0-9])g\b", r"\g<1>9", c) for c in cits]
        if require_citations and not cits:
            continue
        s_payload = safe_json_loads(r["s_json"])
        doc_type_raw = s_payload.get("doc_type")
        doc_type = override_doc_type(r["pdf_name"], doc_type_raw)
        source_role = compute_source_role(r["pdf_name"], doc_type, r["text_norm"])
        source_role_label = get_source_role_label(source_role)
        asset_fields = build_source_asset_fields(r["page_id"], r["pdf_name"], r["page_number"])
        doc_priority = {"letter": 3, "legal": 3, "commentary": 2, "index": 1, "bibliography": 1, "front_matter": 0}
        priority_score = doc_priority.get(doc_type, 0)
        rank_reason = f"doc_type={doc_type} (+{priority_score}), citations={len(cits)} (+{len(cits)})"
        is_ref = doc_type in {"index", "bibliography", "front_matter"}
        weight_map = {
            "letter": 1.0,
            "legal": 1.0,
            "commentary": 0.6,
            "index": 0.4,
            "bibliography": 0.4,
            "front_matter": 0.1,
        }
        results.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "page_id": r["page_id"],
                "citations": cits,
                "doc_type": doc_type,
                "source_role": source_role,
                "source_role_label": source_role_label,
                "topics": s_payload.get("topics", []),
                "institutions": s_payload.get("institutions", []),
                "snippet": snippet(r["text_norm"] or "", name),
                "page_url": source_record_url(r["page_id"]),
                **asset_fields,
                "evidence_weight": weight_map.get(doc_type, 0.0),
                "is_reference_page": is_ref,
                "rank_reason": rank_reason,
                "_priority_score": priority_score,
                "structural_intelligence": compute_structural_intelligence(
                    r["text_norm"],
                    source_role=source_role,
                    internal_score=float(weight_map.get(doc_type, 0.0)),
                    internal_gap=None,
                ),
            }
        )
        if len(results) >= limit:
            break
    return results


def search_formula(marker: str, limit: int = 20):
    norm = normalize_marker(marker)
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.page_id, p.text_norm, ph.json, s.json AS s_json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        LEFT JOIN social s ON s.page_id = p.page_id
        WHERE lower(ph.json) LIKE '%"formula_markers_norm"%'
          AND lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (f'%"{norm}"%', limit),
    )
    results = []
    for r in rows:
        payload = safe_json_loads(r["json"])
        s_payload = safe_json_loads(r["s_json"])
        snips = payload.get("formula_snippets", [])
        best = snips[0] if snips else (r["text_norm"] or "")
        best = highlight(best[:300], marker)
        doc_type_raw = s_payload.get("doc_type")
        doc_type = override_doc_type(r["pdf_name"], doc_type_raw)
        source_role = compute_source_role(r["pdf_name"], doc_type, r["text_norm"])
        source_role_label = get_source_role_label(source_role)
        asset_fields = build_source_asset_fields(r["page_id"], r["pdf_name"], r["page_number"])
        weight_map = {
            "letter": 1.0,
            "legal": 1.0,
            "commentary": 0.6,
            "index": 0.4,
            "bibliography": 0.4,
            "front_matter": 0.1,
        }
        results.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "page_id": r["page_id"],
                "formula_markers": payload.get("formula_markers", []),
                "formula_snippets": payload.get("formula_snippets", [])[:3],
                "doc_type": doc_type,
                "source_role": source_role,
                "source_role_label": source_role_label,
                "topics": s_payload.get("topics", []),
                "institutions": s_payload.get("institutions", []),
                "snippet": best,
                "citations": payload.get("citations", []),
                "page_url": source_record_url(r["page_id"]),
                **asset_fields,
                "evidence_weight": weight_map.get(doc_type, 0.0),
                "is_reference_page": doc_type in {"index", "bibliography", "front_matter"},
                "structural_intelligence": compute_structural_intelligence(
                    r["text_norm"],
                    source_role=source_role,
                    internal_score=float(weight_map.get(doc_type, 0.0)),
                    internal_gap=None,
                ),
            }
        )
        # attach rank explanation
        score, reason = explain_rank(doc_type, payload.get("citations", []), {"formula_hits": len(payload.get("formula_snippets", []))})
        results[-1]["_priority_score"] = score
        results[-1]["rank_reason"] = reason
    return results


def search_institution(inst: str, limit: int = 20):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.page_id, p.text_norm, ph.json AS ph_json, s.json AS s_json
        FROM social s
        JOIN page_registry p ON p.page_id = s.page_id
        LEFT JOIN philology ph ON ph.page_id = p.page_id
        WHERE lower(s.json) LIKE ?
        LIMIT ?
        """,
        (f"%{inst.lower()}%", limit),
    )
    results = []
    for r in rows:
        ph_payload = safe_json_loads(r["ph_json"])
        s_payload = safe_json_loads(r["s_json"])
        doc_type_raw = s_payload.get("doc_type")
        doc_type = override_doc_type(r["pdf_name"], doc_type_raw)
        source_role = compute_source_role(r["pdf_name"], doc_type, r["text_norm"])
        source_role_label = get_source_role_label(source_role)
        asset_fields = build_source_asset_fields(r["page_id"], r["pdf_name"], r["page_number"])
        weight_map = {
            "letter": 1.0,
            "legal": 1.0,
            "commentary": 0.6,
            "index": 0.4,
            "bibliography": 0.4,
            "front_matter": 0.1,
        }
        results.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "page_id": r["page_id"],
                "doc_type": doc_type,
                "source_role": source_role,
                "source_role_label": source_role_label,
                "topics": s_payload.get("topics", []),
                "institutions": s_payload.get("institutions", []),
                "citations": ph_payload.get("citations", []),
                "snippet": snippet(r["text_norm"] or "", inst),
                "page_url": source_record_url(r["page_id"]),
                **asset_fields,
                "evidence_weight": weight_map.get(doc_type, 0.0),
                "is_reference_page": doc_type in {"index", "bibliography", "front_matter"},
                "structural_intelligence": compute_structural_intelligence(
                    r["text_norm"],
                    source_role=source_role,
                    internal_score=float(weight_map.get(doc_type, 0.0)),
                    internal_gap=None,
                ),
            }
        )
        score, reason = explain_rank(
            doc_type,
            ph_payload.get("citations", []),
            {"institution_tag": 2 if inst.lower() in [x.lower() for x in s_payload.get("institutions", [])] else 0},
        )
        results[-1]["_priority_score"] = score
        results[-1]["rank_reason"] = reason
    return results


def get_page_bundle(page_id: str):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.text_norm, p.text_raw,
               ph.json AS ph_json, s.json AS s_json
        FROM page_registry p
        LEFT JOIN philology ph ON ph.page_id = p.page_id
        LEFT JOIN social s ON s.page_id = p.page_id
        WHERE p.page_id = ?
        """,
        (page_id,),
    )
    if not rows:
        return None
    r = rows[0]
    phil = safe_json_loads(r["ph_json"])
    soc = safe_json_loads(r["s_json"])
    asset_fields = build_source_asset_fields(page_id, r["pdf_name"], r["page_number"])

    def _structured_snippets(payload, key):
        out = []
        for snip in payload.get(key, []):
            if ": " in snip:
                marker, text = snip.split(": ", 1)
                out.append({"marker": marker, "snippet": text})
        return out

    return {
        "pdf_name": r["pdf_name"],
        "page_number": r["page_number"],
        "page_id": page_id,
        "text_norm": r["text_norm"],
        "text_raw": r["text_raw"],
        **asset_fields,
        "philology": {
            **phil,
            "formula_snippets_structured": _structured_snippets(phil, "formula_snippets"),
        },
        "social": soc,
    }


@corpus_bp.get("/favicon.jpg")
def favicon():
    return send_file(PROJECT_ROOT / "favcon.jpg", mimetype="image/jpeg")


@corpus_bp.get("/")
def corpus_help():
    return jsonify(
        {
            "ok": True,
            "routes": {
                "deity": "/corpus/search/deity?name=istarzaat&require_citations=false&limit=5",
                "formula": "/corpus/search/formula?marker=li-t%C3%B9-la&limit=5",
                "institution": "/corpus/search/institution?inst=naruqqum&limit=5",
                "help": "/corpus/help",
                "page": "/corpus/page/<page_id>",
                "page_visuals": "/corpus/page/<page_id>/visuals",
                "page_compare": "/corpus/page/<page_id>/compare",
                "demo": "/corpus/demo",
                "browse": "/corpus/browse",
                "stats": "/corpus/stats",
            },
            "notes": [
                "Use /corpus/help for a full UI guide with navigation controls and keyboard shortcuts",
                "Use require_citations=true to filter deity results to those with citations",
                "marker is matched against formula_markers and snippets; ASCII folding supported (li-tu-la)",
                "inst is matched against social.institutions",
                "page endpoint supports include_text=true to return text_norm/text_raw; defaults to false",
            ],
        }
    )


@corpus_bp.get("/help")
def corpus_help_page():
    html_doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MIM Corpus Help</title>
  <style>
    :root {
      --bg: #0b1021;
      --panel: #131d3a;
      --card: #111a32;
      --accent: #7ec7ff;
      --muted: #9cb4e6;
      --border: #23365c;
    }
    body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg); color: #e7ecf7; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 22px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
    h1 { margin: 0 0 8px; color: var(--accent); }
    h2 { margin: 0 0 8px; color: var(--accent); font-size: 18px; }
    p, li { color: #d9e6ff; line-height: 1.45; }
    ul { margin: 0; padding-left: 18px; }
    code {
      background: #0f1832; border: 1px solid #2d4472; border-radius: 5px; padding: 1px 6px;
      color: #d7e6ff; font-family: Consolas, "Courier New", monospace; font-size: 12px;
    }
    .meta { color: var(--muted); font-size: 13px; margin: 6px 0; }
    .actions { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
    .actions a {
      background:#1f2c4d; color:#b6c8ff; border:1px solid #2d4472; border-radius:6px; padding:6px 10px;
      text-decoration:none; font-size:12px;
    }
    .actions a:hover { background:#27406d; }
    .kbd { display:inline-block; min-width:18px; text-align:center; padding:1px 6px; border:1px solid #3d5f95; border-radius:5px; background:#132149; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MIM Corpus Help</h1>
    <div class="meta">How to use the interface, routes, and evidence navigation quickly.</div>
    <div class="actions">
      <a href="/corpus/mim">Open MIM UI</a>
      <a href="/corpus/browse" target="_blank" rel="noopener">Browse all sources</a>
      <a href="/corpus/" target="_blank" rel="noopener">API route index</a>
    </div>

    <section class="panel">
      <h2>Start Here</h2>
      <ul>
        <li>Run server: <code>python app/corpus/query_api.py</code></li>
        <li>Main UI: <code>/corpus/mim</code></li>
        <li>Root path <code>/</code> is not defined for this app and returns Not Found by design.</li>
      </ul>
    </section>

    <section class="panel">
      <h2>Main Controls</h2>
      <ul>
        <li><strong>View Mode:</strong> Internal Only, Routed (recommended), ORACC Only.</li>
        <li><strong>Structural Filters:</strong> fragmentation, template, policy, origin.</li>
        <li><strong>Result Pool:</strong> top-N loaded per section (3/10/25/50/100).</li>
        <li><strong>Global Search + Browse:</strong> full corpus navigation with pagination and jump controls.</li>
        <li><strong>Cohesive Method Header:</strong> active artifact preview, structural profile, routing decision, trace summary.</li>
      </ul>
    </section>

    <section class="panel">
      <h2>Evidence Navigation</h2>
      <ul>
        <li>Use Prev/Next to move across ranked evidence in current scope.</li>
        <li>Scope switch: All / Belief / Speech / Behavior.</li>
        <li>Jump directly with Go-to index (<code>#</code> input).</li>
        <li>Search within loaded evidence via Search evidence.</li>
        <li>Tag/Note artifacts to revisit later (saved in browser local storage).</li>
        <li>Use <strong>Prefer Primary ON/OFF</strong> to bias ranking toward primary text candidates.</li>
      </ul>
    </section>

    <section class="panel">
      <h2>Keyboard Shortcuts</h2>
      <ul>
        <li><span class="kbd">J</span> next evidence</li>
        <li><span class="kbd">K</span> previous evidence</li>
        <li><span class="kbd">/</span> focus evidence search</li>
        <li><span class="kbd">Enter</span> open active source in Source Explorer</li>
        <li><span class="kbd">E</span> toggle Citation Explorer</li>
        <li><span class="kbd">Esc</span> close tooltip</li>
      </ul>
    </section>

    <section class="panel">
      <h2>Visual and Source Links</h2>
      <ul>
        <li><code>Open source record</code>: source page record for the selected artifact.</li>
        <li><code>Open likely visual source</code>: best external visual candidate URL when available.</li>
        <li><code>Open visual candidates</code>: ranked visual candidate list.</li>
        <li><code>Compare source vs visual</code>: side-by-side compare page for source context and selected candidate.</li>
      </ul>
      <p class="meta">Not all sources have local scan images indexed. Commentary/wrapper items can be context-only and still useful for routing.</p>
    </section>

    <section class="panel">
      <h2>Core Routes</h2>
      <ul>
        <li><code>/corpus/mim</code> main UI</li>
        <li><code>/corpus/help</code> this page</li>
        <li><code>/corpus/browse</code> full corpus browser</li>
        <li><code>/corpus/demo</code> demo JSON payload</li>
        <li><code>/corpus/page/&lt;page_id&gt;/story</code> source/story payload</li>
        <li><code>/corpus/page/&lt;page_id&gt;/citations</code> citations list</li>
        <li><code>/corpus/page/&lt;page_id&gt;/visuals</code> visual candidates</li>
        <li><code>/corpus/page/&lt;page_id&gt;/compare</code> source-vs-visual compare</li>
      </ul>
    </section>
  </div>
</body>
</html>"""
    return Response(html_doc, mimetype="text/html")


@corpus_bp.get("/stats")
@json_errors
def corpus_stats():
    total = q("SELECT COUNT(*) AS n FROM page_registry")[0]["n"]

    def _scalar(sql: str, params=()):
        return q(sql, params)[0]["n"]

    def _pct(n: int, d: int) -> float:
        return round((100.0 * n / d), 1) if d else 0.0

    citations = _scalar(
        """
        SELECT COUNT(*) AS n
        FROM philology
        WHERE lower(json) LIKE '%"citations"%'
          AND lower(json) NOT LIKE '%"citations": []%'
        """
    )

    formulas = _scalar(
        """
        SELECT COUNT(*) AS n
        FROM philology
        WHERE lower(json) LIKE '%"formula_markers"%'
          AND lower(json) NOT LIKE '%"formula_markers": []%'
        """
    )

    institutions = _scalar(
        """
        SELECT COUNT(*) AS n
        FROM social
        WHERE lower(json) LIKE '%"institutions"%'
          AND lower(json) NOT LIKE '%"institutions": []%'
        """
    )

    classified = _scalar(
        """
        SELECT COUNT(*) AS n
        FROM social
        WHERE lower(json) LIKE '%"doc_type"%'
          AND lower(json) NOT LIKE '%"doc_type": "unknown"%'
          AND lower(json) NOT LIKE '%"doc_type": ""%'
        """
    )

    return jsonify(
        {
            "ok": True,
            "counts": {
                "total": total,
                "citations": citations,
                "formula_markers": formulas,
                "institutions": institutions,
                "doc_type_classified": classified,
            },
            "percentages": {
                "citations": _pct(citations, total),
                "formula_markers": _pct(formulas, total),
                "institutions": _pct(institutions, total),
                "doc_type_classified": _pct(classified, total),
            },
        }
    )


@corpus_bp.get("/demo")
@json_errors
def demo_route():
    try:
        demo_limit = int(request.args.get("limit", "25"))
    except ValueError:
        demo_limit = 25
    demo_limit = max(3, min(demo_limit, 100))
    fetch_limit = max(50, demo_limit)

    deity_results = search_deity("istarzaat", require_citations=True, limit=fetch_limit)
    doc_priority = {"letter": 3, "legal": 3, "commentary": 2, "index": 1, "bibliography": 1, "front_matter": 0}
    deity_results.sort(
        key=lambda r: (
            -(doc_priority.get(r.get("doc_type"), 0)),
            -len(r.get("citations", [])),
        )
    )
    deity_results = deity_results[:demo_limit]

    formula_results = search_formula("li-tù-la", limit=fetch_limit)
    formula_results.sort(
        key=lambda r: (
            -(doc_priority.get(r.get("doc_type"), 0)),
            -len(r.get("formula_snippets", [])),
            -len(r.get("citations", [])),
        )
    )
    formula_results = formula_results[:demo_limit]

    inst_results = search_institution("naruqqum", limit=fetch_limit)
    inst_results.sort(
        key=lambda r: (
            -(doc_priority.get(r.get("doc_type"), 0)),
            -len(r.get("citations", [])),
        )
    )
    inst_results = inst_results[:demo_limit]
    return jsonify(
        {
            "ok": True,
            "narrative": [
                "1) Deity attestations: find mentions of istarzaat with citations and rank by doc_type + citation count.",
                "2) Formula examples: retrieve pages where li-tù-la appears and show local context snippets.",
                "3) Institutions: retrieve naruqqum occurrences with topics and citations.",
            ],
            "deity_attestations": {
                "query": {"name": "istarzaat", "require_citations": True, "limit": demo_limit},
                "count_returned": len(deity_results),
                "results": deity_results,
            },
            "formula_examples": {
                "query": {"marker": "li-tù-la", "limit": demo_limit},
                "count_returned": len(formula_results),
                "results": formula_results,
            },
            "institution_examples": {
                "query": {"inst": "naruqqum", "limit": demo_limit},
                "count_returned": len(inst_results),
                "results": inst_results,
            },
        }
    )


@corpus_bp.get("/mim")
def demo_ui():
    html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <link rel="icon" type="image/jpeg" href="/corpus/favicon.jpg" />
  <title>MIM Corpus Demo</title>
  <style>
    :root {
      --bg: #0b1021;
      --card: #111a32;
      --panel: #131d3a;
      --accent: #7ec7ff;
      --muted: #9cb4e6;
      --border: #23365c;
    }
    body { margin: 0; padding: 0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg); color: #e7ecf7; }
    header { padding: 16px 24px; background: linear-gradient(120deg, #0f1b3d, #12284a); border-bottom: 1px solid var(--border); }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0.4px; }
    main { padding: 20px 24px 40px; display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
    .panel h2 { margin: 0 0 8px; font-size: 19px; color: var(--accent); }
    .connector { grid-column: 1 / -1; background: #101a33; border: 1px dashed var(--border); }
    .connector .meta { font-size: 13px; color: var(--muted); }
    .card { border: 1px solid #1f2d4f; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: var(--card); }
    .card.active-evidence-card { border-color: #4ba3ff; box-shadow: 0 0 0 1px rgba(75, 163, 255, 0.55) inset, 0 8px 20px rgba(15, 88, 160, 0.28); }
    .card-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .card-thumb-link { display:inline-flex; width:34px; height:34px; border:1px solid #2d4472; border-radius:6px; overflow:hidden; background:#0d1734; flex: 0 0 auto; }
    .card-thumb-link img { width:100%; height:100%; object-fit:cover; display:block; }
    .card-thumb-fallback { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; font-size:10px; font-weight:700; letter-spacing:0.4px; color:#b6c8ff; }
    .card-actions { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .chip-btn { background:#1f2c4d; color:#cfe0ff; border:1px solid #2d4472; border-radius:999px; font-size:11px; padding:2px 8px; cursor:pointer; }
    .chip-btn:hover { background:#27406d; }
    .chip-btn.active { background:#335f3b; border-color:#3f8a5d; color:#d8ffe7; }
    .chip-btn.action-tag { background:#1f4f2f; border-color:#3f8a5d; color:#d9ffe8; }
    .chip-btn.action-tag:hover { background:#26673d; }
    .chip-btn.action-note { background:#5a4320; border-color:#9a7131; color:#ffe9c7; }
    .chip-btn.action-note:hover { background:#72552a; }
    .chip-btn.action-prefer { background:#2e476f; border-color:#41679d; color:#d9e8ff; }
    .chip-btn.action-prefer.active { background:#2f5f3c; border-color:#3f8a5d; color:#d8ffe7; }
    .chip-btn.action-prefer:hover { background:#3f5f8f; }
    .meta { font-size: 12px; color: var(--muted); margin-bottom: 6px; display: flex; gap: 8px; flex-wrap: wrap; }
    .tag { background: #1f2c4d; color: #b6c8ff; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
    .tag.source-primary { background: #225a37; color: #d8ffe7; border: 1px solid #3f8a5d; }
    .tag.source-wrapper { background: #32465f; color: #deebff; border: 1px solid #4e6a8f; }
    .tag.source-commentary { background: #20283a; color: #aebad4; border: 1px solid #3a4a68; opacity: 0.8; }
    .method-bar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:10px; }
    .method-label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.4px; }
    .mode-toggle { display:flex; gap:8px; flex-wrap:wrap; }
    .mode-btn { background:#1a2849; color:#b6c8ff; border:1px solid #2d4472; padding:6px 10px; border-radius:999px; font-size:12px; cursor:pointer; }
    .mode-btn.active { background:#2a65ff; border-color:#2a65ff; color:#fff; }
    .method-select { background:#0f1832; color:#d7e6ff; border:1px solid #2d4472; border-radius:6px; padding:5px 8px; font-size:12px; }
    .routing-drawer { margin-top:6px; border:1px solid var(--border); border-radius:8px; background:#0f1832; padding:8px 10px; }
    .routing-drawer summary { cursor:pointer; color:var(--accent); font-size:12px; }
    .route-pills { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
    .route-pill { background:#1f2c4d; color:#d6e4ff; border:1px solid #2d4472; border-radius:999px; padding:3px 8px; font-size:11px; }
    .struct-header { border:1px solid #2d4472; border-radius:8px; background:#0f1832; padding:10px; margin:8px 0; }
    .method-header-panel { grid-column: 1 / -1; }
    .cohesive-header-grid { display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:10px; margin-top:8px; }
    .cohesive-cell { border:1px solid #2b4372; border-radius:8px; background:#0f1832; padding:10px; min-height: 170px; }
    .cohesive-cell-title { margin:0 0 8px; color:var(--accent); font-size:14px; font-weight:700; display:flex; align-items:center; gap:8px; }
    .artifact-preview-box { border:1px solid #2b4372; border-radius:8px; background:#0d1734; min-height:96px; display:flex; align-items:center; justify-content:center; overflow:hidden; margin-bottom:8px; }
    .artifact-preview-box img { width:100%; height:100%; max-height:180px; object-fit:cover; display:block; }
    .artifact-placeholder { font-size:12px; color:#9cb4e6; line-height:1.4; padding:10px; }
    .artifact-actions { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
    .artifact-actions a, .artifact-actions button { background:#1f2c4d; color:#b6c8ff; border:1px solid #2d4472; border-radius:6px; padding:5px 8px; font-size:11px; cursor:pointer; text-decoration:none; }
    .artifact-actions a:hover, .artifact-actions button:hover { background:#27406d; text-decoration:none; }
    .cohesive-metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:6px; }
    .cohesive-metric { font-size:12px; color:#d9e6ff; background:#132149; border:1px solid #28406f; border-radius:6px; padding:6px 8px; }
    .decision-trace-summary { margin-top:10px; border:1px solid #2d4472; border-radius:8px; background:#0d1734; padding:10px; }
    .decision-trace-title { margin:0 0 8px; color:#9fd6ff; font-size:13px; font-weight:700; display:flex; align-items:center; gap:8px; }
    .decision-trace-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:6px; }
    .decision-trace-item { font-size:12px; color:#d8e4ff; background:#10203f; border:1px solid #2a467a; border-radius:6px; padding:6px 8px; }
    .confidence-components { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .confidence-components .tag { border:1px solid #2d4472; background:#10203f; color:#d7e6ff; }
    .cohesive-rationale { font-size:12px; color:#d6e4ff; margin-top:6px; line-height:1.4; }
    .evidence-nav-controls { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:8px; }
    .btn-nav-step { background:#1f2c4d; color:#b6c8ff; border:1px solid #2d4472; border-radius:6px; padding:5px 8px; font-size:11px; cursor:pointer; }
    .btn-nav-step:hover { background:#27406d; }
    a.btn-nav-step.nav-link-btn { text-decoration:none; display:inline-flex; align-items:center; }
    a.btn-nav-step.nav-link-btn:hover { text-decoration:none; }
    .nav-search-input { min-width: 170px; }
    .nav-goto-input { width: 78px; min-width: 78px; }
    .tagged-select { min-width: 220px; max-width: 360px; }
    .struct-title { margin:0 0 6px; color:var(--accent); font-size:14px; font-weight:700; display:flex; align-items:center; gap:8px; }
    .struct-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:6px; margin-bottom:8px; }
    .struct-item { font-size:12px; color:#d9e6ff; background:#132149; border:1px solid #28406f; border-radius:6px; padding:6px 8px; }
    .routing-box { border:1px solid #2f4a7e; border-radius:8px; background:#0d1734; padding:8px; }
    .routing-line { font-size:12px; color:#d8e4ff; margin:3px 0; }
    .policy-chip { background:#2a65ff; color:#fff; border-radius:999px; padding:2px 8px; font-size:11px; margin-left:6px; }
    .policy-chip.internal { background:#335f3b; }
    .policy-chip.hybrid { background:#225b6a; }
    .policy-chip.fallback { background:#6a4d22; }
    .policy-chip.strong_rerank { background:#5b2b66; }
    .struct-badges { display:flex; gap:6px; flex-wrap:wrap; margin:6px 0 2px; }
    .struct-badge { background:#10203f; border:1px solid #2a467a; color:#cfe0ff; border-radius:999px; padding:2px 8px; font-size:11px; }
    .delta-note { font-size:11px; color:#9cc3ff; margin-bottom:8px; }
    .method-sequence { margin: 0 0 8px; border: 1px dashed #2b4372; border-radius: 8px; background: #0e1836; padding: 8px; }
    .method-sequence .meta { margin-bottom: 0; }
    .decision-panel { border: 1px solid #355188; border-radius: 8px; background: #0d1734; padding: 10px; margin-bottom: 10px; }
    .decision-title { display: flex; align-items: center; gap: 8px; margin: 0 0 6px; color: var(--accent); font-size: 13px; font-weight: 700; }
    .decision-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 6px; margin: 0 0 8px; }
    .decision-cell { font-size: 12px; color: #d8e4ff; background: #111f45; border: 1px solid #2a4578; border-radius: 6px; padding: 6px 8px; }
    .confidence-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 8px; font-size: 11px; border: 1px solid transparent; }
    .confidence-pill.high { background: #1f4d2f; color: #d7ffe7; border-color: #3f8a5d; }
    .confidence-pill.medium { background: #2e476e; color: #d9e8ff; border-color: #4d6ea1; }
    .confidence-pill.low { background: #5f3a2a; color: #ffe0d3; border-color: #92563e; }
    .selection-row { display: flex; gap: 8px; flex-wrap: wrap; font-size: 11px; color: #9fc0ee; margin-bottom: 6px; }
    .selection-pill { border-radius: 999px; border: 1px solid #375b93; background: #10203f; color: #cfe2ff; padding: 2px 8px; }
    .selection-pill.changed { border-color: #3f8a5d; background: #133723; color: #dcffe9; }
    .trace-details { border: 1px solid #2b4472; border-radius: 6px; background: #0c1530; margin-top: 8px; padding: 6px 8px; }
    .trace-details summary { cursor: pointer; color: var(--accent); font-size: 12px; }
    .trace-list { margin: 6px 0 0; padding-left: 16px; color: #d2e2ff; font-size: 12px; }
    .trace-list li { margin: 3px 0; }
    .card.changed-card { border-color: #3f8a5d; box-shadow: 0 0 0 1px rgba(63, 138, 93, 0.45) inset; }
    .tag.changed { background: #133723; color: #dcffe9; border: 1px solid #3f8a5d; }
    .tag.context-only { background: #2b2f3d; color: #b9c2d9; border: 1px solid #495372; }
    .result-diff { font-size: 11px; color: #a7c8f4; margin: 4px 0 8px; }
    /* Evidence summary */
    #evidence-summary.panel { grid-column: 1 / -1; }
    .evidence-summary-block, .evidence-keypoints, .evidence-quotes, .evidence-confidence { margin-top: 8px; }
    .evidence-summary-block h3 { margin: 0 0 6px; color: var(--accent); }
    .evidence-keypoints h4, .evidence-quotes h4 { margin: 6px 0 4px; color: var(--muted); }
    .evidence-keypoints ul { padding-left: 16px; margin: 0; color: #e7ecf7; }
    .evidence-keypoints li { margin: 4px 0; }
    .evidence-quotes details { background: #0f1832; border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; }
    .evidence-quotes details.context-only { border-style: dashed; opacity: 0.9; }
    .evidence-quotes summary { cursor: pointer; color: var(--accent); }
    .evidence-quotes blockquote { margin: 6px 0; color: #e7ecf7; font-size: 13px; line-height: 1.4; }
    .evidence-confidence ul { padding-left: 16px; margin: 4px 0 0; }
    .evidence-driving { border: 1px solid #2f4a7e; border-radius: 8px; background: #0d1734; padding: 10px; margin-top: 8px; }
    .evidence-driving h4 { margin: 0 0 6px; color: var(--accent); font-size: 14px; }
    .evidence-driving .meta { margin-bottom: 8px; }
    .evidence-candidate { border: 1px solid #28406f; border-radius: 8px; background: #0f1b39; padding: 8px; margin-bottom: 8px; }
    .evidence-candidate.selected { border-color: #3f8a5d; background: #122944; }
    .evidence-candidate.context-only { opacity: 0.9; border-style: dashed; }
    .candidate-title { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 6px; font-size: 12px; color: #d9e8ff; }
    .candidate-quote { font-size: 12.5px; color: #e7ecf7; line-height: 1.45; margin: 6px 0; }
    .match-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .match-badge { font-size: 11px; border-radius: 999px; border: 1px solid #3d588f; background: #142750; color: #d9e8ff; padding: 2px 8px; }
    .match-badge.off { border-color: #4d5773; background: #1f2638; color: #b3bed8; }
    .candidate-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .candidate-actions a { font-size: 12px; }
    .alt-candidates details { border: 1px solid #28406f; border-radius: 6px; background: #0f1a37; padding: 6px 8px; margin-bottom: 6px; }
    .alt-candidates summary { cursor: pointer; color: #bcd6ff; }
    .confidence-breakdown { margin-top: 8px; border: 1px solid #2a4475; border-radius: 8px; background: #0d1732; padding: 8px; }
    .confidence-breakdown h4 { margin: 0 0 6px; font-size: 13px; color: var(--accent); }
    .confidence-breakdown ul { margin: 0; padding-left: 16px; color: #d6e5ff; font-size: 12px; }
    .confidence-breakdown li { margin: 3px 0; }
    .context-note { font-size: 11px; color: #b5c4df; margin-top: 6px; }
    .snippet { font-size: 13px; color: #e7ecf7; line-height: 1.5; max-height: 3.6em; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
    .snippet mark { background: #ffe599; color: #111; padding: 0 2px; border-radius: 2px; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    footer { padding: 14px 24px 20px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--border); background: var(--bg); }
    .stats { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; margin-bottom: 10px; align-items: stretch; width: 100%; justify-content: space-between; }
    .stat-card { display: flex; flex-direction: column; justify-content: flex-start; align-items: flex-start; gap: 8px; background: #0f1832; border: 1px solid #2d4472; border-radius: 6px !important; padding: 16px 18px 18px; min-height: 140px; min-width: 180px; flex: 1 1 220px; box-sizing: border-box; position: relative; box-shadow: 0 10px 24px rgba(0,0,0,0.36); }
    .stat-card.clickable-stat { cursor: pointer; }
    .stat-card.clickable-stat:hover { border-color: #4ba3ff; box-shadow: 0 10px 24px rgba(0,0,0,0.36), 0 0 0 1px rgba(75,163,255,0.45) inset; }
    .stat-label { font-size: 13px; color: var(--muted); line-height: 1.35; }
    .stat-value { font-size: 20px; color: var(--accent); font-weight: 700; line-height: 1.25; }
    .stat-card.help { cursor: help; }
    .stat-help { position: absolute; top: 6px; right: 6px; width: 18px; height: 18px; border-radius: 999px; border: 1px solid var(--border); background: #0f1832; color: var(--accent); font-size: 12px; line-height: 18px; padding: 0; cursor: pointer; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }
    .chip { border: 1px solid var(--border); background: #0f1832; color: #b6c8ff; padding: 4px 10px; border-radius: 999px; font-size: 12px; cursor: pointer; user-select: none; }
    .chip.active { background: #1f2c4d; color: #e7ecf7; }
    .interp-card { border: 1px solid var(--border); background: #0f1832; border-radius: 8px; padding: 12px; margin-top: 8px; }
    .interp-card h4 { margin: 0 0 6px; color: var(--accent); font-size: 14px; }
    .interp-row { font-size: 12.5px; line-height: 1.45; margin: 4px 0; }
    .interp-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .interp-actions a { font-size: 12px; color: var(--accent); }
    .plain-english { margin-top: 6px; padding: 8px; border: 1px solid var(--border); border-radius: 6px; background: #0d152b; }
    .plain-english .pe-context, .plain-english .pe-paraphrase { font-size: 12px; line-height: 1.4; }
    .plain-english .pe-confidence { font-size: 11.5px; color: var(--muted); margin-top: 4px; }
    .context-list { margin: 6px 0 0; padding-left: 16px; }
    .context-list li { margin-bottom: 4px; font-size: 12.5px; line-height: 1.45; color: #e7ecf7; }
    .snippet-card { border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin-top: 8px; background: #0d1326; }
    .badge.mode-on { background: #1f2c4d; color: #b6c8ff; }
    .badge.mode-off { background: #1a243a; color: #9cb4e6; opacity: 0.7; }
    .gate-copy { color: #d2b993; }
    textarea { width: 100%; min-height: 120px; background: #0f152b; color: #e7ecf7; border: 1px solid #24345c; border-radius: 6px; padding: 8px; }
    button { background: #2a65ff; color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1f4bcc; }
    /* Drawer */
    #drawer { position: fixed; top: 0; right: -480px; width: 460px; height: 100vh; background: #0f152b; border-left: 1px solid var(--border); box-shadow: -6px 0 20px rgba(0,0,0,0.4); transition: right 0.25s ease; z-index: 999; padding: 16px; overflow: hidden; display: flex; flex-direction: column; }
    #drawer.open { right: 0; }
    #drawer h3 { margin: 0 0 8px; color: var(--accent); }
    #drawer .close { cursor: pointer; color: var(--muted); font-size: 12px; }
    #drawer .list { flex: 1; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; padding: 8px; background: #0d1326; min-height: 100px; }
    #drawer .list-item { padding: 6px; border-bottom: 1px solid #1a2644; cursor: pointer; }
    #drawer .list-item:last-child { border-bottom: none; }
    #drawer .detail { margin-top: 10px; border: 1px solid var(--border); border-radius: 6px; padding: 10px; background: #0d1326; min-height: 120px; overflow-y: auto; }
    #drawer .detail .snippet { max-height: none; -webkit-line-clamp: unset; -webkit-box-orient: unset; display: block; white-space: pre-wrap; }
    #drawer .badge { background: #1f2c4d; color: #b6c8ff; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; }
    #drawer-divider { height: 6px; margin: 6px 0; background: #1a2644; border-radius: 3px; cursor: row-resize; user-select: none; }

    /* Source Explorer */
    #source-panel { position: fixed; top: 0; right: -520px; width: 500px; height: 100vh; background: #0c1227; border-left: 1px solid var(--border); box-shadow: -6px 0 20px rgba(0,0,0,0.4); transition: right 0.25s ease; z-index: 1000; padding: 16px; overflow-y: auto; }
    #source-panel.open { right: 0; }
    #source-panel h3 { margin: 0 0 8px; color: var(--accent); }
    #source-panel .close { cursor: pointer; color: var(--muted); font-size: 12px; }
    #source-content .section { border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin-bottom: 10px; background: #0d152d; }
    #source-content .section h4 { margin: 0 0 6px; color: var(--accent); font-size: 14px; }
    #source-content .badge { background: #1f2c4d; color: #b6c8ff; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; }
    #source-content .badge.mode-on { background: #1f2c4d; color: #b6c8ff; }
    #source-content .badge.mode-off { background: #1a243a; color: #9cb4e6; opacity: 0.7; }
    #source-content .source-header { position: sticky; top: 0; background: #0c1227; border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-bottom: 10px; z-index: 2; }
    #source-content .source-header .role-title { font-size: 14px; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
    #source-content .source-header .gate-copy { color: #d2b993; }
    #source-content .paper-ladder { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
    #source-content .paper-ladder .ladder-step { border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; font-size: 12px; background: #0d152d; color: #c7d6ff; }
    #source-content .paper-ladder button.ladder-step { width: 100%; text-align: left; cursor: pointer; }
    #source-content .paper-ladder .ladder-step.disabled { opacity: 0.6; cursor: not-allowed; }
    #source-content .paper-ladder .ladder-step:not(.disabled):hover { border-color: #4ba3ff; background: #11214a; }
    #source-content .paper-ladder .ladder-step.active { background: #1f2c4d; color: #e7ecf7; border-color: #2d4472; }
    #source-content .paper-ladder .ladder-arrow { text-align: center; color: var(--muted); font-size: 12px; line-height: 1; }
    #source-content .ladder-cta { margin-top: 6px; }
    #source-content .ladder-cta.muted { color: var(--muted); font-size: 12px; }
    #source-content .ladder-cta.ok { color: #9ee6b8; font-size: 12px; }
    #source-content .source-tools-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; align-items: center; }
    #source-content .source-tools-row a,
    #source-content .source-tools-row button { background: #1f2c4d; color: #b6c8ff; border: 1px solid #2d4472; border-radius: 6px; padding: 5px 8px; font-size: 11px; cursor: pointer; text-decoration: none; }
    #source-content .source-tools-row a:hover,
    #source-content .source-tools-row button:hover { background: #27406d; text-decoration: none; }
    /* Tooltip */
    .term { display: inline-flex; align-items: center; gap: 8px; }
    .help {
      width: 18px; height: 18px; border-radius: 999px;
      border: 1px solid var(--border);
      background: #0f1832; color: var(--accent);
      font-size: 12px; line-height: 18px; padding: 0;
      cursor: pointer;
    }
    .help:hover { background: #13224a; }
    .tooltip {
      position: fixed;
      max-width: 360px;
      background: #0d1326;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 12px 10px;
      box-shadow: 0 14px 40px rgba(0,0,0,0.5);
      z-index: 9999;
    }
    .tooltip-title {
      font-weight: 700; color: var(--accent);
      margin-bottom: 6px; font-size: 13px;
    }
    .tooltip-body {
      color: #e7ecf7; font-size: 12.5px; line-height: 1.45;
    }
    @media (max-width: 1100px) {
      .cohesive-header-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MIM Corpus Demo</h1>
    <div style="font-size:13px;color:var(--muted);margin-top:6px;">Live evidence retrieved from the corpus - click to inspect sources.</div>
    <div class="method-bar" style="margin-top:8px;">
      <a href="/corpus/help" target="_blank" rel="noopener" class="btn-nav-step nav-link-btn">Open Help</a>
      <button class="help" type="button" data-tip-title="Help page" data-tip="Opens the full usage guide, navigation map, routes, and keyboard shortcuts.">?</button>
    </div>
    <div class="method-bar">
      <span class="method-label">View Mode</span>
      <button class="help" type="button" data-tip-title="View mode" data-tip="Switch retrieval perspective. Internal Only prioritizes in-corpus matches. Routed applies structural policy routing. ORACC Only prioritizes external memory context.">?</button>
      <div class="mode-toggle" id="view-mode-toggle">
        <button type="button" class="mode-btn" data-mode="internal">Internal Only</button>
        <button class="help" type="button" data-tip-title="Internal Only" data-tip="Use in-corpus memory first and keep selection conservative. Best for high-confidence direct matches.">?</button>
        <button type="button" class="mode-btn active" data-mode="routed">Routed (Recommended)</button>
        <button class="help" type="button" data-tip-title="Routed (Recommended)" data-tip="Profile structure, choose retrieval policy, and apply calibrated gating before final selection.">?</button>
        <button type="button" class="mode-btn" data-mode="oracc">ORACC Only</button>
        <button class="help" type="button" data-tip-title="ORACC Only" data-tip="Favor external ORACC-linked context memory. Useful for low-confidence or fragmentary cases.">?</button>
      </div>
    </div>
    <div class="method-bar">
      <span class="method-label">Structural Filters</span>
      <button class="help" type="button" data-tip-title="Structural filters" data-tip="Filter result cards by profile labels and routing policy. These filters do not retrain the model; they only change visible result subsets.">?</button>
      <select id="filter-fragmentation" class="method-select">
        <option value="">Fragmentation: All</option>
        <option value="Complete">Complete</option>
        <option value="Partial">Partial</option>
        <option value="Fragmentary">Fragmentary</option>
      </select>
      <button class="help" type="button" data-tip-title="Fragmentation filter" data-tip="Complete = minimal damage markers. Partial = some uncertainty markers. Fragmentary = bracket/unknown-heavy text.">?</button>
      <select id="filter-template" class="method-select">
        <option value="">Template: All</option>
        <option value="Slot-Structured">Slot-Structured</option>
        <option value="Narrative">Narrative</option>
        <option value="Hybrid">Hybrid</option>
      </select>
      <button class="help" type="button" data-tip-title="Template filter" data-tip="Slot-Structured favors formulaic fields, Narrative favors prose-like flow, Hybrid mixes both.">?</button>
      <select id="filter-policy" class="method-select">
        <option value="">Policy: All</option>
        <option value="INTERNAL">INTERNAL</option>
        <option value="HYBRID">HYBRID</option>
        <option value="FALLBACK">FALLBACK</option>
        <option value="STRONG_RERANK">STRONG_RERANK</option>
      </select>
      <button class="help" type="button" data-tip-title="Policy filter" data-tip="INTERNAL = no expansion. HYBRID = mixed pool. FALLBACK = external assist under low confidence. STRONG_RERANK = larger rerank emphasis.">?</button>
      <select id="filter-origin" class="method-select">
        <option value="">Origin: All</option>
        <option value="primary_text">Internal (Primary)</option>
        <option value="archival_wrapper">Wrapper</option>
        <option value="scholarly_commentary">Commentary</option>
      </select>
      <button class="help" type="button" data-tip-title="Origin filter" data-tip="Primary text is direct translation evidence. Wrapper/commentary are context layers used for routing and support.">?</button>
    </div>
    <div class="method-bar">
      <span class="method-label">Result Pool</span>
      <button class="help" type="button" data-tip-title="Result pool size" data-tip="Top-ranked items loaded per section. Start with top evidence, then increase pool to browse deeper candidates.">?</button>
      <select id="demo-limit-select" class="method-select">
        <option value="3">Top 3</option>
        <option value="10">Top 10</option>
        <option value="25" selected>Top 25</option>
        <option value="50">Top 50</option>
        <option value="100">Top 100</option>
      </select>
      <a href="/corpus/browse" target="_blank" rel="noopener" class="btn-nav-step nav-link-btn">Browse All Sources</a>
      <button class="help" type="button" data-tip-title="Browse all sources" data-tip="Opens the full corpus browser with pagination/search across all indexed pages.">?</button>
    </div>
    <div class="method-bar">
      <span class="method-label">Global Search</span>
      <button class="help" type="button" data-tip-title="Global search" data-tip="Search across the full corpus (all indexed pages), then jump directly to any item/page in the browser.">?</button>
      <input id="global-source-search" class="method-select nav-search-input" placeholder="Search whole corpus..." />
      <button type="button" id="btn-open-browse-search" class="btn-nav-step">Open in Browser</button>
    </div>
    <details id="routing-analytics" class="routing-drawer">
      <summary>Routing Summary (Session) <button class="help" type="button" data-tip-title="Routing summary" data-tip="Session counts by route plus structural distribution of currently visible result cards.">?</button></summary>
      <div id="routing-summary" class="meta" style="margin-top:8px;">No route data yet.</div>
      <div id="struct-distribution" class="meta" style="margin-top:6px;"></div>
    </details>
    <div id="stats" class="stats"></div>
  </header>
  <main id="panels">
    <div class="panel connector">
      <div class="meta">Istar-ZA.AT &rarr; oath / witness formula &rarr; naruqqum (joint-stock / pooled capital)</div>
    </div>
    <div class="panel"><h2>Loading demo...</h2><div id="status" class="meta"></div></div>
  </main>
  <div id="evidence-summary" class="panel" style="margin: 0 24px 20px 24px;"></div>

  <div id="drawer">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Citation Explorer <button class="help" type="button" data-tip-title="Citation explorer" data-tip="Browse citation links for the selected source page, inspect matched excerpts, and step across linked references.">?</button></h3>
      <button type="button" class="close" onclick="closeDrawer()" style="background:none;border:none;color:var(--muted);cursor:pointer;">Close</button>
    </div>
    <div id="drawer-source" class="meta" style="margin-bottom:6px;"></div>
    <div id="drawer-list" class="list"></div>
    <div id="drawer-divider"></div>
    <div id="drawer-detail" class="detail">Select a citation to see context.</div>
  </div>

  <div id="source-panel">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Source Explorer</h3>
      <span class="close" onclick="closeSource()">Close</span>
    </div>
    <div id="source-content" class="meta" style="margin-top:8px;">Select a source to view here.</div>
  </div>

  <footer>
    <div>Routes: /corpus/help, /corpus/demo, /corpus/browse, /corpus/search/deity, /corpus/search/formula, /corpus/search/institution, /corpus/page/&lt;page_id&gt;, /corpus/page/&lt;page_id&gt;/visuals, /corpus/page/&lt;page_id&gt;/compare</div>
    <div style="margin-top:6px;">MIM Robots LLC - <a href="https://www.mimrobots.com" target="_blank">www.mimrobots.com</a></div>
  </footer>

  <script>
    const panelsEl = document.getElementById("panels");
    const statsEl = document.getElementById("stats");
    const drawer = document.getElementById("drawer");
    const drawerList = document.getElementById("drawer-list");
    const drawerDetail = document.getElementById("drawer-detail");
    const drawerSource = document.getElementById("drawer-source");
    const drawerDivider = document.getElementById("drawer-divider");
    const sourcePanel = document.getElementById("source-panel");
    const sourceContent = document.getElementById("source-content");
    const routingSummaryEl = document.getElementById("routing-summary");
    const structDistributionEl = document.getElementById("struct-distribution");
    const filterFragmentationEl = document.getElementById("filter-fragmentation");
    const filterTemplateEl = document.getElementById("filter-template");
    const filterPolicyEl = document.getElementById("filter-policy");
    const filterOriginEl = document.getElementById("filter-origin");
    const demoLimitSelectEl = document.getElementById("demo-limit-select");
    const globalSourceSearchEl = document.getElementById("global-source-search");
    const openBrowseSearchBtnEl = document.getElementById("btn-open-browse-search");
    let currentPage = null;
    let currentCitationList = [];
    let currentCitationIndex = -1;
      let currentSourceResults = [];
      let currentSourceIndex = -1;
    let currentSourceMeta = null;
    let demoCache = null;
    let currentViewMode = "routed";
    const UI_STATE_KEY = "mim_ui_state_v1";
    const USER_TAGS_KEY = "mim_user_tags_v1";
    const DEMO_LIMIT_KEY = "mim_demo_limit_v1";
    const PREFER_PRIMARY_KEY = "mim_prefer_primary_v1";
    let demoResultLimit = 25;
    let navScope = "all";
    let navSearchQuery = "";
    let activeEvidencePageId = null;
    let activeSelectionSource = "auto";
    let pendingActiveScroll = false;
    let lastNavEntries = [];
    let lastNavEntriesAll = [];
    let userTags = {};
    let preferPrimary = true;
    // Resizable list/detail within drawer (vertical split)
    let isResizing = false;
    let startY = 0;
    let startListHeight = 0;
    let startDetailHeight = 0;
    const minHeight = 80;

    const onMouseMove = (e) => {
      if (!isResizing) return;
      const dy = e.clientY - startY;
      const newListH = Math.max(minHeight, startListHeight + dy);
      const newDetailH = Math.max(minHeight, startDetailHeight - dy);
      drawerList.style.height = `${newListH}px`;
      drawerDetail.style.height = `${newDetailH}px`;
    };
    const onMouseUp = () => {
      if (!isResizing) return;
      isResizing = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    drawerDivider.addEventListener("mousedown", (e) => {
      isResizing = true;
      startY = e.clientY;
      startListHeight = drawerList.offsetHeight;
      startDetailHeight = drawerDetail.offsetHeight;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });

    const escapeHtml = (s) => (s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

    const markTerm = (text) => {
      const safe = escapeHtml(text || "");
      return safe.replace(/&lt;&lt;(.+?)&gt;&gt;/g, "<mark>$1</mark>");
    };

    const closestFromTargetV2 = (target, selector) => {
      if (!target) return null;
      const base = (target.nodeType === 1) ? target : target.parentElement;
      if (!base || typeof base.closest !== "function") return null;
      return base.closest(selector);
    };

    const loadUiStateV2 = () => {
      try {
        const raw = localStorage.getItem(UI_STATE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          if (parsed.scope && ["all", "belief", "speech", "behavior"].includes(parsed.scope)) {
            navScope = parsed.scope;
          }
          if (parsed.page_id) {
            activeEvidencePageId = String(parsed.page_id);
            activeSelectionSource = "restored";
          }
        }
      } catch (err) {}
    };

    const saveUiStateV2 = () => {
      try {
        localStorage.setItem(
          UI_STATE_KEY,
          JSON.stringify({
            scope: navScope,
            page_id: activeEvidencePageId,
          }),
        );
      } catch (err) {}
    };

    const clampDemoLimitV2 = (value) => {
      const n = Number(value);
      if (!Number.isFinite(n)) return 25;
      if (n <= 3) return 3;
      if (n <= 10) return 10;
      if (n <= 25) return 25;
      if (n <= 50) return 50;
      return 100;
    };

    const loadDemoLimitV2 = () => {
      try {
        const raw = localStorage.getItem(DEMO_LIMIT_KEY);
        if (!raw) return;
        demoResultLimit = clampDemoLimitV2(raw);
      } catch (err) {
        demoResultLimit = 25;
      }
    };

    const saveDemoLimitV2 = () => {
      try {
        localStorage.setItem(DEMO_LIMIT_KEY, String(demoResultLimit));
      } catch (err) {}
    };

    const loadPreferPrimaryV2 = () => {
      try {
        const raw = localStorage.getItem(PREFER_PRIMARY_KEY);
        if (raw === null) return;
        preferPrimary = String(raw).toLowerCase() !== "false";
      } catch (err) {
        preferPrimary = true;
      }
    };

    const savePreferPrimaryV2 = () => {
      try {
        localStorage.setItem(PREFER_PRIMARY_KEY, preferPrimary ? "true" : "false");
      } catch (err) {}
    };

    const loadUserTagsV2 = () => {
      try {
        const raw = localStorage.getItem(USER_TAGS_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          userTags = parsed;
        }
      } catch (err) {
        userTags = {};
      }
    };

    const saveUserTagsV2 = () => {
      try {
        localStorage.setItem(USER_TAGS_KEY, JSON.stringify(userTags || {}));
      } catch (err) {}
    };

    const normalizeTagMetaV2 = (meta = {}) => {
      return {
        pdf_name: meta.pdf_name || "Unknown source",
        page_number: meta.page_number ?? "-",
      };
    };

    const isTaggedV2 = (pageId) => {
      const key = String(pageId || "");
      return !!(key && userTags[key]);
    };

    const tagNoteV2 = (pageId) => {
      const key = String(pageId || "");
      return (userTags[key] && userTags[key].note) ? String(userTags[key].note) : "";
    };

    const upsertTagV2 = (pageId, meta = {}, note = null) => {
      const key = String(pageId || "");
      if (!key) return;
      const base = normalizeTagMetaV2(meta);
      const prev = userTags[key] || {};
      userTags[key] = {
        page_id: key,
        pdf_name: base.pdf_name,
        page_number: base.page_number,
        note: note === null ? (prev.note || "") : String(note || ""),
        updated_at: Date.now(),
      };
      saveUserTagsV2();
    };

    const removeTagV2 = (pageId) => {
      const key = String(pageId || "");
      if (!key || !userTags[key]) return;
      delete userTags[key];
      saveUserTagsV2();
    };

    const taggedEntriesV2 = () => {
      return Object.values(userTags || {}).sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0));
    };

    const taggedOptionsHtmlV2 = () => {
      const rows = taggedEntriesV2();
      if (!rows.length) {
        return `<option value="">Tagged: none</option>`;
      }
      return [
        `<option value="">Tagged (${rows.length})</option>`,
        ...rows.map((row) => {
          const note = row.note ? ` | ${String(row.note).replace(/\s+/g, " ").slice(0, 36)}` : "";
          return `<option value="${escapeHtml(String(row.page_id || ""))}">${escapeHtml(`${row.pdf_name} (p${row.page_number})${note}`)}</option>`;
        }),
      ].join("");
    };

    loadUiStateV2();
    loadDemoLimitV2();
    loadUserTagsV2();
    loadPreferPrimaryV2();

    const structuralOf = (r) => (r && r.structural_intelligence) ? r.structural_intelligence : null;
    const routingOf = (r) => {
      const s = structuralOf(r);
      return (s && s.routing) ? s.routing : null;
    };
    const profileOf = (r) => {
      const s = structuralOf(r);
      return (s && s.profile) ? s.profile : null;
    };

    const policyClass = (policy) => {
      const p = (policy || "").toUpperCase();
      if (p === "INTERNAL") return "internal";
      if (p === "HYBRID") return "hybrid";
      if (p === "FALLBACK") return "fallback";
      if (p === "STRONG_RERANK") return "strong_rerank";
      return "internal";
    };

    const rolePreferenceBoostV2 = (r, mode = "routed") => {
      const role = String(r?.source_role || "");
      if (mode === "oracc") {
        if (preferPrimary) {
          if (role === "primary_text") return 0.08;
          if (role === "archival_wrapper") return 0.04;
          return 0.0;
        }
        if (role === "scholarly_commentary") return 0.12;
        if (role === "archival_wrapper") return 0.03;
        return 0.0;
      }
      if (preferPrimary) {
        if (role === "primary_text") return 0.18;
        if (role === "archival_wrapper") return 0.06;
        return -0.08;
      }
      if (role === "primary_text") return 0.1;
      if (role === "archival_wrapper") return 0.04;
      return 0.02;
    };

    const scoreForMode = (r, mode) => {
      const rt = routingOf(r) || {};
      const internal = Number(rt.internal_score ?? r.evidence_weight ?? 0);
      const oracc = Number(rt.oracc_score ?? internal);
      const policy = (rt.selected_policy || "INTERNAL").toUpperCase();
      if (mode === "internal") {
        const roleBoost = rolePreferenceBoostV2(r, "internal");
        return internal + roleBoost;
      }
      if (mode === "oracc") {
        const roleBoost = rolePreferenceBoostV2(r, "oracc");
        return oracc + roleBoost;
      }
      const policyBoost = policy === "HYBRID" ? 0.08 : (policy === "FALLBACK" ? 0.05 : (policy === "STRONG_RERANK" ? 0.04 : 0.0));
      const roleBoost = rolePreferenceBoostV2(r, "routed");
      return internal + policyBoost + roleBoost;
    };

    const applyViewMode = (results, mode) => {
      const arr = [...(results || [])];
      arr.sort((a, b) => scoreForMode(b, mode) - scoreForMode(a, mode));
      if (mode === "oracc") {
        const byCommentary = arr.filter(r => r.source_role === "scholarly_commentary");
        const others = arr.filter(r => r.source_role !== "scholarly_commentary");
        return [...byCommentary, ...others];
      }
      return arr;
    };

    const applyStructuralFilters = (results) => {
      const frag = (filterFragmentationEl?.value || "").trim();
      const tpl = (filterTemplateEl?.value || "").trim();
      const pol = (filterPolicyEl?.value || "").trim().toUpperCase();
      const origin = (filterOriginEl?.value || "").trim();
      return (results || []).filter((r) => {
        const p = profileOf(r) || {};
        const rt = routingOf(r) || {};
        if (frag && (p.fragmentation || "") !== frag) return false;
        if (tpl && (p.template_type || "") !== tpl) return false;
        if (pol && (rt.selected_policy || "").toUpperCase() !== pol) return false;
        if (origin && (r.source_role || "") !== origin) return false;
        return true;
      });
    };

    const selectionDelta = (base, current) => {
      const baseIds = new Set((base || []).slice(0, 3).map(x => x.page_id));
      const curIds = new Set((current || []).slice(0, 3).map(x => x.page_id));
      let changed = 0;
      curIds.forEach((id) => { if (!baseIds.has(id)) changed += 1; });
      return changed;
    };

    const compareTopSelectionV2 = (base, current) => {
      const baseTop = (base || [])[0] || null;
      const curTop = (current || [])[0] || null;
      return {
        changed: !!(baseTop && curTop && String(baseTop.page_id || "") !== String(curTop.page_id || "")),
        baseTop: baseTop,
        currentTop: curTop,
        basePage: baseTop ? baseTop.page_id : "-",
        currentPage: curTop ? curTop.page_id : "-",
        baseOrigin: baseTop ? (baseTop.source_role || "unknown") : "unknown",
        currentOrigin: curTop ? (curTop.source_role || "unknown") : "unknown",
      };
    };

    const confidenceRankV2 = (routing) => {
      const policy = String((routing || {}).selected_policy || "INTERNAL").toUpperCase();
      if (policy === "INTERNAL") return 3;
      if (policy === "HYBRID" || policy === "STRONG_RERANK") return 2;
      return 1;
    };

    const evidenceStrengthV2 = (r, mode) => {
      const base = scoreForMode(r, mode || "routed");
      const routing = routingOf(r) || {};
      const citations = (r.citations || []).length;
      const primaryBoost = preferPrimary
        ? (r.source_role === "primary_text" ? 0.08 : (r.source_role === "archival_wrapper" ? 0.03 : -0.03))
        : 0.0;
      const confBoost = confidenceRankV2(routing) * 0.04;
      const citationBoost = Math.min(0.08, citations * 0.01);
      return Number(base + primaryBoost + confBoost + citationBoost);
    };

    const navEntriesFromGroupsV2 = (groups, mode, scope = "all") => {
      const rows = [];
      (groups || []).forEach((g) => {
        const sectionKey = g.sectionKey || "section";
        const sectionLabel = g.section || "Section";
        const baselineTop = (g.internal || [])[0] || null;
        (g.current || []).forEach((r, idx) => {
          rows.push({
            sectionKey: sectionKey,
            section: sectionLabel,
            kind: g.kind || "unknown",
            rank: idx + 1,
            currentTop: r,
            baselineTop: baselineTop,
            compare: compareTopSelectionV2(g.internal || [], [r]),
            score: evidenceStrengthV2(r, mode),
          });
        });
      });
      const filtered = scope === "all" ? rows : rows.filter((x) => x.sectionKey === scope);
      filtered.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        if (a.rank !== b.rank) return a.rank - b.rank;
        return String(a.currentTop?.page_id || "").localeCompare(String(b.currentTop?.page_id || ""));
      });
      const seen = new Set();
      const unique = [];
      filtered.forEach((row) => {
        const pid = String(row.currentTop?.page_id || "");
        if (!pid || seen.has(pid)) return;
        seen.add(pid);
        unique.push(row);
      });
      return unique;
    };

    const resolveActiveFocusV2 = (groups, mode) => {
      const scoped = navEntriesFromGroupsV2(groups, mode, navScope);
      const all = navEntriesFromGroupsV2(groups, mode, "all");
      const navEntries = scoped.length ? scoped : all;
      if (!scoped.length && navScope !== "all") {
        navScope = "all";
      }
      let sourceReason = activeSelectionSource || "auto";
      let selected = null;
      const selectedId = activeEvidencePageId ? String(activeEvidencePageId) : "";
      if (selectedId) {
        selected = navEntries.find((x) => String(x.currentTop?.page_id || "") === selectedId) || null;
      }
      if (!selected && navEntries.length) {
        selected = navEntries[0];
        sourceReason = "auto";
      }
      if (!selected) {
        lastNavEntries = [];
        lastNavEntriesAll = [];
        return {
          focus: null,
          navEntries: [],
          navIndex: 0,
          navTotal: 0,
          selectionReason: "No evidence available.",
        };
      }
      activeEvidencePageId = String(selected.currentTop?.page_id || "");
      activeSelectionSource = sourceReason;
      saveUiStateV2();
      lastNavEntries = navEntries;
      lastNavEntriesAll = all;
      const navIndex = navEntries.findIndex((x) => String(x.currentTop?.page_id || "") === activeEvidencePageId) + 1;
      const statLabelMap = {
        total: "corpus indexed",
        citations: "traceable citations",
        formula_markers: "recognizable formulas",
        institutions: "institution mentions",
        doc_type_classified: "classified page types",
      };
      const selectionReason = sourceReason === "restored"
        ? "Active artifact restored from your last session."
        : sourceReason === "navigation"
        ? "Active artifact selected via evidence navigation."
        : sourceReason === "search"
        ? "Active artifact selected from evidence search."
        : sourceReason.startsWith("stat:")
        ? `Active artifact selected from stats (${statLabelMap[sourceReason.slice(5)] || sourceReason.slice(5)}).`
        : "Active artifact auto-selected as strongest evidence in session.";
      const preferenceSuffix = preferPrimary
        ? " Primary-first preference is ON."
        : " Primary-first preference is OFF.";
      return {
        focus: selected,
        navEntries: navEntries,
        navIndex: Math.max(navIndex, 1),
        navTotal: navEntries.length,
        selectionReason: `${selectionReason}${preferenceSuffix}`,
      };
    };

    const setActiveEvidenceByIdV2 = (pageId, source = "navigation", scroll = false) => {
      if (!pageId) return;
      activeEvidencePageId = String(pageId);
      activeSelectionSource = source;
      pendingActiveScroll = !!scroll;
      saveUiStateV2();
      if (demoCache) renderDemo(demoCache);
    };

    const moveActiveEvidenceV2 = (delta) => {
      if (!lastNavEntries.length) return;
      const curIdx = lastNavEntries.findIndex((x) => String(x.currentTop?.page_id || "") === String(activeEvidencePageId || ""));
      const start = curIdx >= 0 ? curIdx : 0;
      const next = (start + delta + lastNavEntries.length) % lastNavEntries.length;
      const target = lastNavEntries[next];
      if (!target?.currentTop?.page_id) return;
      setActiveEvidenceByIdV2(target.currentTop.page_id, "navigation", false);
    };

    const gotoActiveEvidenceIndexV2 = (indexValue) => {
      if (!lastNavEntries.length) return;
      const n = Number(indexValue);
      if (!Number.isFinite(n)) return;
      const idx = Math.max(1, Math.min(lastNavEntries.length, Math.floor(n))) - 1;
      const target = lastNavEntries[idx];
      if (!target?.currentTop?.page_id) return;
      setActiveEvidenceByIdV2(target.currentTop.page_id, "navigation", false);
    };

    const runEvidenceSearchV2 = (query) => {
      const q = String(query || "").trim().toLowerCase();
      navSearchQuery = q;
      if (!q || !lastNavEntries.length) return;
      const match = lastNavEntries.find((x) => {
        const r = x.currentTop || {};
        const hay = `${r.pdf_name || ""} ${r.snippet || ""} ${r.page_id || ""} ${x.section || ""}`.toLowerCase();
        return hay.includes(q);
      });
      if (match?.currentTop?.page_id) {
        setActiveEvidenceByIdV2(match.currentTop.page_id, "search", true);
      }
    };

    const focusByStatV2 = (statKey) => {
      const key = String(statKey || "").trim();
      const entries = lastNavEntriesAll.length ? lastNavEntriesAll : lastNavEntries;
      if (!entries.length) return;
      let pool = [...entries];
      if (key === "citations") {
        pool = pool.filter((x) => ((x.currentTop?.citations || []).length > 0));
        pool.sort((a, b) => ((b.currentTop?.citations || []).length - (a.currentTop?.citations || []).length) || (b.score - a.score));
      } else if (key === "formula_markers") {
        pool = pool.filter((x) => ((x.currentTop?.formula_markers || []).length > 0) || ((x.currentTop?.formula_snippets || []).length > 0));
        pool.sort((a, b) => {
          const bFormula = (b.currentTop?.formula_markers || []).length + (b.currentTop?.formula_snippets || []).length;
          const aFormula = (a.currentTop?.formula_markers || []).length + (a.currentTop?.formula_snippets || []).length;
          return (bFormula - aFormula) || (b.score - a.score);
        });
      } else if (key === "institutions") {
        pool = pool.filter((x) => ((x.currentTop?.institutions || []).length > 0));
        pool.sort((a, b) => ((b.currentTop?.institutions || []).length - (a.currentTop?.institutions || []).length) || (b.score - a.score));
      } else if (key === "doc_type_classified") {
        pool = pool.filter((x) => {
          const dt = String(x.currentTop?.doc_type || "unknown").toLowerCase();
          return dt && dt !== "unknown";
        });
      }
      const ranked = pool.length ? pool : entries;
      const currentId = String(activeEvidencePageId || "");
      const target = ranked.find((x) => String(x.currentTop?.page_id || "") !== currentId) || ranked[0];
      if (!target?.currentTop?.page_id) return;
      setActiveEvidenceByIdV2(target.currentTop.page_id, `stat:${key}`, false);
    };

    const sourceDisplayNameV2 = (r) => {
      if (!r) return "Unknown source";
      const pdf = r.pdf_name || "Unknown source";
      const page = (r.page_number ?? "-");
      return `${pdf} (p${page})`;
    };

    const renderCohesiveHeaderV2 = (focus, mode, navMeta = {}) => {
      if (!focus || !focus.currentTop) {
        return `<div class="panel method-header-panel"><h2>Method Header</h2><div class="meta">No routed results available.</div></div>`;
      }
      const currentTop = focus.currentTop;
      const baselineTop = focus.baselineTop;
      const compare = focus.compare || {};
      const si = structuralOf(currentTop) || {};
      const profile = si.profile || {};
      const routing = si.routing || {};
      const conf = confidenceBandV2(routing);
      const policy = String(routing.selected_policy || "INTERNAL").toUpperCase();
      const thresholds = routing.thresholds_applied || {};
      const pageUrl = currentTop.page_url || `/corpus/page/${currentTop.page_id}/record`;
      const thumbUrl = currentTop.asset_thumb_url || currentTop.pdf_render_url || "";
      const fullUrl = currentTop.asset_full_url || thumbUrl || "";
      const externalVisualUrl = currentTop.external_visual_url || "";
      const visualCandidatesUrl = currentTop.external_visual_candidates_url || (currentTop.page_id ? `/corpus/page/${currentTop.page_id}/visuals` : "");
      const visualCompareUrl = currentTop.visual_compare_url || (currentTop.page_id ? `/corpus/page/${currentTop.page_id}/compare` : "");
      const isPrimarySource = String(currentTop.source_role || "") === "primary_text";
      const firstCitationRef = ((currentTop.citations || [])[0] || "").toString();
      const baselineSourceType = sourceTypeLabel((baselineTop && baselineTop.source_role) || "scholarly_commentary");
      const currentSourceType = sourceTypeLabel(currentTop.source_role || "scholarly_commentary");
      const primaryAvailable = isPrimarySource || !!firstCitationRef;
      const reason = methodExplanationV2(currentTop);
      const baselineLabel = sourceDisplayNameV2(baselineTop);
      const currentLabel = sourceDisplayNameV2(currentTop);
      const navScopeValue = navMeta.scope || "all";
      const navIndex = Number(navMeta.index || 0);
      const navTotal = Number(navMeta.total || 0);
      const selectionReason = navMeta.selectionReason || "Active artifact auto-selected as strongest evidence in session.";
      const sortBasis = mode === "internal"
        ? "internal_score"
        : mode === "oracc"
        ? "oracc_score"
        : "evidence_strength (routed)";
      const primaryPreferenceLabel = preferPrimary ? "Prefer Primary: ON" : "Prefer Primary: OFF";
      const activeTagged = isTaggedV2(currentTop.page_id);
      const activeNote = tagNoteV2(currentTop.page_id);
      const taggedOptions = taggedOptionsHtmlV2();
      const confidenceTags = [
        `Band: ${conf.level}`,
        `Policy: ${policy}`,
        `Internal: ${Number(routing.internal_score ?? 0).toFixed(2)}`,
        `ORACC: ${Number(routing.oracc_score ?? 0).toFixed(2)}`,
      ];
      const artifactPreview = thumbUrl
        ? `<div class="artifact-preview-box"><img src="${escapeHtml(thumbUrl)}" alt="Artifact preview for ${escapeHtml(currentLabel)}"></div>`
        : `<div class="artifact-preview-box"><div class="artifact-placeholder">No local scan image is indexed for this source. Use visual candidate links and source record/citations for artifact verification.</div></div>`;
      return `
        <div class="panel method-header-panel">
          <h2>Cohesive Method Header
            <button class="help" type="button" data-tip-title="Cohesive method header" data-tip="This block shows the full decision chain in one place: artifact context, structural profile, routing policy, and decision trace.">?</button>
          </h2>
          <div class="meta">Focus section: ${escapeHtml(focus.section)} | View mode: ${escapeHtml(String(mode || "routed").toUpperCase())}</div>
          <div class="meta">${escapeHtml(selectionReason)} Basis: evidence_strength + structural compatibility.</div>
          <div class="evidence-nav-controls">
            <span class="method-label">Evidence Navigation
              <button class="help" type="button" data-tip-title="Keyboard shortcuts" data-tip="J/K: next/previous evidence, /: focus search, Enter: open active source, E: toggle citation explorer.">?</button>
            </span>
            <button type="button" class="btn-nav-step" id="btn-nav-prev">&lt; Prev</button>
            <button type="button" class="btn-nav-step" id="btn-nav-next">Next &gt;</button>
            <select class="method-select" id="nav-scope-select">
              <option value="all" ${navScopeValue === "all" ? "selected" : ""}>All</option>
              <option value="belief" ${navScopeValue === "belief" ? "selected" : ""}>Belief</option>
              <option value="speech" ${navScopeValue === "speech" ? "selected" : ""}>Speech</option>
              <option value="behavior" ${navScopeValue === "behavior" ? "selected" : ""}>Behavior</option>
            </select>
            <span class="route-pill">Item ${navIndex}/${navTotal} (${navScopeValue === "all" ? "all sections" : navScopeValue})</span>
            <span class="route-pill">Sorted by: ${escapeHtml(sortBasis)}</span>
            <span class="route-pill">${escapeHtml(primaryPreferenceLabel)}</span>
            <input type="number" id="nav-goto-item" class="method-select nav-goto-input" min="1" max="${Math.max(navTotal, 1)}" placeholder="#">
            <button type="button" class="btn-nav-step" id="btn-nav-goto">Go</button>
            <input type="search" id="nav-search-input" class="method-select nav-search-input" value="${escapeHtml(navSearchQuery || "")}" placeholder="Search evidence">
            <button type="button" class="btn-nav-step" id="btn-nav-search">Find</button>
            <button type="button" class="chip-btn action-prefer ${preferPrimary ? "active" : ""}" id="btn-toggle-prefer-primary">${preferPrimary ? "Prefer Primary ON" : "Prefer Primary OFF"}</button>
            <button type="button" class="chip-btn action-tag ${activeTagged ? "active" : ""}" id="btn-tag-current" data-page-id="${escapeHtml(String(currentTop.page_id || ""))}" data-pdf-name="${escapeHtml(currentTop.pdf_name || "")}" data-page-number="${escapeHtml(String(currentTop.page_number ?? ""))}">${activeTagged ? "Tagged" : "Add Tag"}</button>
            <button type="button" class="chip-btn action-note" id="btn-note-current" data-page-id="${escapeHtml(String(currentTop.page_id || ""))}" data-pdf-name="${escapeHtml(currentTop.pdf_name || "")}" data-page-number="${escapeHtml(String(currentTop.page_number ?? ""))}">${activeNote ? "Edit Note" : "Add Note"}</button>
            <select class="method-select tagged-select" id="tagged-select">${taggedOptions}</select>
            <button type="button" class="btn-nav-step" id="btn-load-tagged">Load Tagged</button>
          </div>
          ${activeNote ? `<div class="meta">Tag note: ${escapeHtml(activeNote)}</div>` : ""}
          <div class="cohesive-header-grid">
            <section class="cohesive-cell">
              <div class="cohesive-cell-title">Artifact Preview
                <button class="help" type="button" data-tip-title="Artifact preview" data-tip="Shows tablet/page image when available. If absent, use PDF page and source trace links.">?</button>
              </div>
              ${artifactPreview}
              <div class="meta">${escapeHtml(currentLabel)}</div>
              <div class="artifact-actions">
                ${fullUrl ? `<a href="${escapeHtml(fullUrl)}" target="_blank" rel="noopener">Open source scan</a>` : ""}
                ${(!fullUrl && externalVisualUrl) ? `<a href="${escapeHtml(externalVisualUrl)}" target="_blank" rel="noopener">Open likely visual source</a>` : ""}
                ${visualCandidatesUrl ? `<a href="${escapeHtml(visualCandidatesUrl)}" target="_blank" rel="noopener">Open visual candidates</a>` : ""}
                ${visualCompareUrl ? `<a href="${escapeHtml(visualCompareUrl)}" target="_blank" rel="noopener">Compare source vs visual</a>` : ""}
                <a href="${escapeHtml(pageUrl)}" target="_blank" rel="noopener">Open source record</a>
                <button type="button" class="btn-header-open-source" data-page-id="${escapeHtml(String(currentTop.page_id || ""))}">Show excerpt</button>
                ${(!isPrimarySource && firstCitationRef) ? `<button type="button" class="btn-header-open-primary" data-page-id="${escapeHtml(String(currentTop.page_id || ""))}" data-ref="${escapeHtml(firstCitationRef)}">Open linked primary excerpt</button>` : ""}
              </div>
            </section>
            <section class="cohesive-cell">
              <div class="cohesive-cell-title">Structural Profile
                <button class="help" type="button" data-tip-title="Structural profile" data-tip="Auto-labeled artifact structure used to route retrieval policy and candidate scoring.">?</button>
              </div>
              <div class="cohesive-metrics">
                <div class="cohesive-metric"><strong>Fragmentation:</strong> ${escapeHtml(profile.fragmentation || "Unknown")}</div>
                <div class="cohesive-metric"><strong>Formula Density:</strong> ${escapeHtml(profile.formula_density || "Unknown")}</div>
                <div class="cohesive-metric"><strong>Numeric Density:</strong> ${escapeHtml(profile.numeric_density || "Unknown")}</div>
                <div class="cohesive-metric"><strong>Template Type:</strong> ${escapeHtml(profile.template_type || "Unknown")}</div>
                <div class="cohesive-metric"><strong>Length Bucket:</strong> ${escapeHtml(profile.length_bucket || "Unknown")}</div>
                <div class="cohesive-metric"><strong>Domain Intent:</strong> ${escapeHtml(profile.domain_intent || "Unknown")}</div>
              </div>
            </section>
            <section class="cohesive-cell">
              <div class="cohesive-cell-title">Routing Decision
                <span class="policy-chip ${policyClass(policy)}">${escapeHtml(policy)}</span>
                <span class="confidence-pill ${conf.cls}">${conf.level}</span>
              </div>
              <div class="cohesive-metrics">
                <div class="cohesive-metric"><strong>Internal score:</strong> ${Number(routing.internal_score ?? 0).toFixed(2)}</div>
                <div class="cohesive-metric"><strong>ORACC score:</strong> ${Number(routing.oracc_score ?? 0).toFixed(2)}</div>
                <div class="cohesive-metric"><strong>internal_low:</strong> ${Number(thresholds.internal_low ?? 0.5).toFixed(2)}</div>
                <div class="cohesive-metric"><strong>internal_high:</strong> ${Number(thresholds.internal_high ?? 0.6).toFixed(2)}</div>
              </div>
              <div class="cohesive-rationale"><strong>Rationale:</strong> ${escapeHtml(routing.rationale || "No rationale available.")}</div>
            </section>
          </div>
          <section class="decision-trace-summary">
            <div class="decision-trace-title">Decision Trace Summary
              <button class="help" type="button" data-tip-title="Decision trace" data-tip="Compares baseline internal selection against current routed selection and shows confidence components used for final choice.">?</button>
            </div>
            <div class="decision-trace-grid">
              <div class="decision-trace-item"><strong>Selected source:</strong> ${escapeHtml(currentLabel)}</div>
              <div class="decision-trace-item"><strong>Baseline vs routed:</strong> ${escapeHtml(baselineLabel)} -> ${escapeHtml(currentLabel)}</div>
              <div class="decision-trace-item"><strong>Selection change:</strong> ${compare.changed ? "YES" : "NO"} | <strong>Section:</strong> ${escapeHtml(focus.section)}</div>
              <div class="decision-trace-item"><strong>Baseline selection type:</strong> ${escapeHtml(baselineSourceType.text)} | <strong>Current selection type:</strong> ${escapeHtml(currentSourceType.text)}${isPrimarySource ? "" : " (context-only)"} | <strong>Primary available:</strong> ${primaryAvailable ? "Yes" : "No"}</div>
              <div class="decision-trace-item"><strong>Why:</strong> ${escapeHtml(reason)}</div>
            </div>
            <div class="confidence-components">
              ${confidenceTags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}
            </div>
          </section>
        </div>
      `;
    };

    const confidenceBandV2 = (routing) => {
      const rt = routing || {};
      const policy = String(rt.selected_policy || "INTERNAL").toUpperCase();
      const internal = Number(rt.internal_score ?? 0);
      const thr = rt.thresholds_applied || {};
      const low = Number(thr.internal_low ?? 0.5);
      const high = Number(thr.internal_high ?? 0.6);
      if (policy === "FALLBACK") return { level: "Low", cls: "low", note: "Fallback route engaged due to low-confidence match." };
      if (policy === "HYBRID" || policy === "STRONG_RERANK") return { level: "Medium", cls: "medium", note: "Hybrid routing used because the match was ambiguous." };
      if (internal >= high) return { level: "High", cls: "high", note: "Internal score cleared high threshold." };
      if (internal >= low) return { level: "Medium", cls: "medium", note: "Internal score in middle band." };
      return { level: "Low", cls: "low", note: "Internal score below low threshold." };
    };

    const structuralBadgesHtmlV2 = (r) => {
      const p = profileOf(r);
      if (!p) return "";
      const badges = [];
      if (p.numeric_density === "High") badges.push("Numeric-heavy");
      if (p.fragmentation === "Fragmentary") badges.push("Fragmentary");
      if (p.template_type === "Narrative") badges.push("Narrative");
      if (p.template_type === "Slot-Structured") badges.push("Slot-structured");
      if (p.domain_intent === "Legal" || p.domain_intent === "Administrative") badges.push("Institutional");
      if (!badges.length) return "";
      return `<div class="struct-badges">${badges.map(b => `<span class="struct-badge">${b}</span>`).join("")}</div>`;
    };

    const methodExplanationV2 = (r) => {
      const si = structuralOf(r);
      if (!si || !si.method_explanation) return "Method explanation unavailable.";
      return si.method_explanation;
    };

    const routingTraceHtmlV2 = (si) => {
      if (!si) return "";
      const p = si.profile || {};
      const rf = si.raw_features || {};
      const rt = si.routing || {};
      const conf = confidenceBandV2(rt);
      return `
        <details class="trace-details">
          <summary>View Routing Trace</summary>
          <ul class="trace-list">
            <li>Fragmentation: ${escapeHtml(p.fragmentation || "Unknown")}</li>
            <li>Template: ${escapeHtml(p.template_type || "Unknown")}</li>
            <li>Numeric density: ${escapeHtml(p.numeric_density || "Unknown")}</li>
            <li>Internal score: ${Number(rt.internal_score ?? 0).toFixed(2)}</li>
            <li>ORACC score: ${Number(rt.oracc_score ?? 0).toFixed(2)}</li>
            <li>Confidence: ${conf.level}</li>
            <li>Raw features: bracket_ratio=${Number(rf.bracket_ratio ?? 0).toFixed(4)}, digit_ratio=${Number(rf.digit_ratio ?? 0).toFixed(4)}</li>
            <li>Rationale: ${escapeHtml(rt.rationale || "No rationale available.")}</li>
          </ul>
        </details>
      `;
    };

    const renderStructuralHeaderV2 = (si, opts = {}) => {
      if (!si) {
        return "<div class='struct-header'><div class='meta'>Structural profile unavailable.</div></div>";
      }
      const p = si.profile || {};
      const r = si.routing || {};
      const policy = (r.selected_policy || "INTERNAL").toUpperCase();
      const thresholds = r.thresholds_applied || {};
      const internalLow = (thresholds.internal_low ?? 0.5).toFixed(2);
      const internalHigh = (thresholds.internal_high ?? 0.6).toFixed(2);
      const conf = confidenceBandV2(r);
      const compact = opts.compact === true;
      return `
        <div class="struct-header">
          <div class="struct-title">Structural Profile
            <button class="help" type="button" data-tip-title="Structural profile" data-tip="Auto-labeled text structure used for routing decisions: fragmentation, formula density, numeric density, template type, length bucket, and domain intent.">?</button>
          </div>
          <div class="struct-grid">
            <div class="struct-item"><strong>Fragmentation:</strong> ${escapeHtml(p.fragmentation || "Unknown")}</div>
            <div class="struct-item"><strong>Formula Density:</strong> ${escapeHtml(p.formula_density || "Unknown")}</div>
            <div class="struct-item"><strong>Numeric Density:</strong> ${escapeHtml(p.numeric_density || "Unknown")}</div>
            <div class="struct-item"><strong>Template Type:</strong> ${escapeHtml(p.template_type || "Unknown")}</div>
            <div class="struct-item"><strong>Length Bucket:</strong> ${escapeHtml(p.length_bucket || "Unknown")}</div>
            <div class="struct-item"><strong>Domain Intent:</strong> ${escapeHtml(p.domain_intent || "Unknown")}</div>
          </div>
          <div class="routing-box">
            <div class="struct-title" style="font-size:13px;margin-bottom:4px;">Routing Decision
              <span class="policy-chip ${policyClass(policy)}">${escapeHtml(policy)}</span>
              <span class="confidence-pill ${conf.cls}">${conf.level} confidence</span>
              <button class="help" type="button" data-tip-title="Method explanation" data-tip="${escapeHtml(si.method_explanation || "No explanation available.")}" aria-label="Method explanation">?</button>
            </div>
            <div class="routing-line"><strong>Internal Score:</strong> ${Number(r.internal_score ?? 0).toFixed(2)} | <strong>ORACC Score:</strong> ${Number(r.oracc_score ?? 0).toFixed(2)}</div>
            <div class="routing-line"><strong>Thresholds Applied:</strong> internal_low=${internalLow}, internal_high=${internalHigh}</div>
            <div class="routing-line"><strong>Rationale:</strong> ${escapeHtml(r.rationale || "No rationale available.")}</div>
          </div>
          ${compact ? "" : routingTraceHtmlV2(si)}
        </div>
      `;
    };

    const renderDecisionPanelV2 = (sectionLabel, currentResults, internalResults, mode) => {
      const topCurrent = (currentResults || [])[0] || null;
      if (!topCurrent) {
        return `<div class="decision-panel"><div class="meta">No results after filtering.</div></div>`;
      }
      const si = structuralOf(topCurrent) || {};
      const profile = si.profile || {};
      const routing = si.routing || {};
      const conf = confidenceBandV2(routing);
      const topCompare = compareTopSelectionV2(internalResults || [], currentResults || []);
      const changedCls = topCompare.changed ? "changed" : "";
      return `
        <div class="method-sequence">
          <div class="meta"><strong>Method sequence:</strong> Artifact analyzed -> Structural context identified -> Retrieval policy selected -> Candidate pool scored -> Evidence traced</div>
        </div>
        <div class="decision-panel">
          <div class="decision-title">Routing Decision for ${escapeHtml(sectionLabel)}
            <span class="policy-chip ${policyClass(routing.selected_policy || "INTERNAL")}">${escapeHtml(String(routing.selected_policy || "INTERNAL").toUpperCase())}</span>
            <span class="confidence-pill ${conf.cls}">${conf.level}</span>
          </div>
          <div class="selection-row">
            <span class="selection-pill ${changedCls}">Selection Change: ${topCompare.changed ? "YES" : "NO"}</span>
            <span class="selection-pill">Mode: ${escapeHtml(String(mode || "routed").toUpperCase())}</span>
            <span class="selection-pill">Baseline page: ${escapeHtml(String(topCompare.basePage || "-"))}</span>
            <span class="selection-pill">Current page: ${escapeHtml(String(topCompare.currentPage || "-"))}</span>
          </div>
          <div class="decision-grid">
            <div class="decision-cell"><strong>Internal score:</strong> ${Number(routing.internal_score ?? 0).toFixed(2)}</div>
            <div class="decision-cell"><strong>ORACC score:</strong> ${Number(routing.oracc_score ?? 0).toFixed(2)}</div>
            <div class="decision-cell"><strong>Template:</strong> ${escapeHtml(profile.template_type || "Unknown")}</div>
            <div class="decision-cell"><strong>Fragmentation:</strong> ${escapeHtml(profile.fragmentation || "Unknown")}</div>
            <div class="decision-cell"><strong>Origin shift:</strong> ${escapeHtml(topCompare.baseOrigin)} -> ${escapeHtml(topCompare.currentOrigin)}</div>
            <div class="decision-cell"><strong>Reason:</strong> ${escapeHtml(methodExplanationV2(topCurrent))}</div>
          </div>
          ${routingTraceHtmlV2(si)}
        </div>
      `;
    };

    const methodExplanation = (r) => {
      const si = structuralOf(r);
      if (!si || !si.method_explanation) return "Method explanation unavailable.";
      return si.method_explanation;
    };

    const structuralBadgesHtml = (r) => {
      const p = profileOf(r);
      if (!p) return "";
      const badges = [];
      if (p.numeric_density === "High") badges.push("🔢 Numeric-heavy");
      if (p.fragmentation === "Fragmentary") badges.push("🧱 Fragmentary");
      if (p.template_type === "Narrative") badges.push("📜 Narrative");
      if (p.template_type === "Slot-Structured") badges.push("🧾 Slot-structured");
      if (p.domain_intent === "Legal" || p.domain_intent === "Administrative") badges.push("🏛 Institutional");
      if (!badges.length) return "";
      return `<div class="struct-badges">${badges.map(b => `<span class="struct-badge">${b}</span>`).join("")}</div>`;
    };

    const renderStructuralHeader = (si) => {
      if (!si) {
        return "<div class='struct-header'><div class='meta'>Structural profile unavailable.</div></div>";
      }
      const p = si.profile || {};
      const r = si.routing || {};
      const policy = (r.selected_policy || "INTERNAL").toUpperCase();
      const thresholds = r.thresholds_applied || {};
      const internalLow = (thresholds.internal_low ?? 0.5).toFixed(2);
      const internalHigh = (thresholds.internal_high ?? 0.6).toFixed(2);
      return `
        <div class="struct-header">
          <div class="struct-title">🧠 Structural Profile</div>
          <div class="struct-grid">
            <div class="struct-item"><strong>Fragmentation:</strong> ${escapeHtml(p.fragmentation || "Unknown")}</div>
            <div class="struct-item"><strong>Formula Density:</strong> ${escapeHtml(p.formula_density || "Unknown")}</div>
            <div class="struct-item"><strong>Numeric Density:</strong> ${escapeHtml(p.numeric_density || "Unknown")}</div>
            <div class="struct-item"><strong>Template Type:</strong> ${escapeHtml(p.template_type || "Unknown")}</div>
            <div class="struct-item"><strong>Length Bucket:</strong> ${escapeHtml(p.length_bucket || "Unknown")}</div>
            <div class="struct-item"><strong>Domain Intent:</strong> ${escapeHtml(p.domain_intent || "Unknown")}</div>
          </div>
          <div class="routing-box">
            <div class="struct-title" style="font-size:13px;margin-bottom:4px;">🔀 Routing Decision
              <span class="policy-chip ${policyClass(policy)}">${escapeHtml(policy)}</span>
              <button class="help" type="button" data-tip-title="Method explanation" data-tip="${escapeHtml(si.method_explanation || "No explanation available.")}" aria-label="Method explanation">?</button>
            </div>
            <div class="routing-line"><strong>Internal Score:</strong> ${Number(r.internal_score ?? 0).toFixed(2)} · <strong>ORACC Score:</strong> ${Number(r.oracc_score ?? 0).toFixed(2)}</div>
            <div class="routing-line"><strong>Thresholds Applied:</strong> internal_low=${internalLow}, internal_high=${internalHigh}</div>
            <div class="routing-line"><strong>Rationale:</strong> ${escapeHtml(r.rationale || "No rationale available.")}</div>
          </div>
        </div>
      `;
    };

    const updateRoutingAnalytics = (groups) => {
      const counts = { INTERNAL: 0, HYBRID: 0, FALLBACK: 0, STRONG_RERANK: 0 };
      const dist = { Fragmentary: 0, "Slot-Structured": 0, "Numeric-heavy": 0, Narrative: 0, total: 0 };
      (groups || []).flat().forEach((r) => {
        const rt = routingOf(r);
        const policy = ((rt && rt.selected_policy) || "INTERNAL").toUpperCase();
        counts[policy] = (counts[policy] || 0) + 1;
        const pf = profileOf(r) || {};
        if (pf.fragmentation === "Fragmentary") dist.Fragmentary += 1;
        if (pf.template_type === "Slot-Structured") dist["Slot-Structured"] += 1;
        if (pf.numeric_density === "High") dist["Numeric-heavy"] += 1;
        if (pf.template_type === "Narrative") dist.Narrative += 1;
        dist.total += 1;
      });
      const pill = (label, count) => `<span class="route-pill">${label}: ${count}</span>`;
      routingSummaryEl.innerHTML = `
        <div class="route-pills">
          ${pill("INTERNAL", counts.INTERNAL || 0)}
          ${pill("HYBRID", counts.HYBRID || 0)}
          ${pill("FALLBACK", counts.FALLBACK || 0)}
          ${pill("STRONG_RERANK", counts.STRONG_RERANK || 0)}
        </div>
      `;
      if (!dist.total) {
        structDistributionEl.innerHTML = "No structural distribution available.";
        return;
      }
      const pct = (n) => `${((100 * n) / dist.total).toFixed(1)}%`;
      structDistributionEl.innerHTML = `
        <div><strong>Structural Distribution</strong> · Fragmentary ${pct(dist.Fragmentary)} · Slot-Structured ${pct(dist["Slot-Structured"])} · Numeric-heavy ${pct(dist["Numeric-heavy"])} · Narrative ${pct(dist.Narrative)}</div>
      `;
    };

    const humanDoc = (dt) => {
      if (dt === "legal") return "Legal text";
      if (dt === "letter") return "Letter";
      if (dt === "commentary") return "Study / commentary";
      if (dt === "index" || dt === "bibliography" || dt === "front_matter") return "Reference";
      return "Unknown";
    };

    const sourceBadge = (r) => {
      if (r.source_role === "primary_text") return "PRIMARY TEXT";
      if (r.source_role === "archival_wrapper") return "ARCHIVAL WRAPPER";
      if (r.source_role === "scholarly_commentary") return "SCHOLARLY COMMENTARY";
      return "SOURCE";
    };
    const docTypeLabel = (dt) => {
      if (dt === "legal") return "Legal document";
      if (dt === "letter") return "Letter (correspondence)";
      if (dt === "commentary") return "Study / commentary";
      if (dt === "index" || dt === "bibliography" || dt === "front_matter") return "Reference";
      return "Unknown";
    };
    const sourceTypeLabel = (role) => {
      if (role === "primary_text") return { text: "Primary text", cls: "source-primary" };
      if (role === "archival_wrapper") return { text: "Archival wrapper", cls: "source-wrapper" };
      if (role === "scholarly_commentary") return { text: "Scholarly commentary", cls: "source-commentary" };
      return { text: "Scholarly commentary", cls: "source-commentary" };
    };

    const sourceRoleHeaderLabel = (role) => {
      if (role === "primary_text") return "Primary Text";
      if (role === "archival_wrapper") return "Archival Wrapper";
      return "Scholarly Commentary";
    };

    const sourceRoleSecondaryTag = (role, docType) => {
      if (role === "primary_text") {
        if (docType === "legal") return "Legal document";
        if (docType === "letter") return "Letter";
        return docTypeLabel(docType);
      }
      if (docType === "legal") return "Contains legal-language excerpts";
      if (docType === "letter") return "Contains letter excerpts";
      if (docType === "commentary") return "Scholarly analysis";
      if (docType === "index" || docType === "bibliography" || docType === "front_matter") return "Reference material";
      return "Contains cited excerpts";
    };

    const isTranslationAllowed = (source) => {
      return source && source.source_role === "primary_text";
    };

    const translationReplacementLabel = (role) => {
      if (role === "archival_wrapper") return "View metadata / seals / witnesses";
      if (role === "scholarly_commentary") return "Why scholars link these texts";
      return "Translate this text";
    };

    const metaLine = (r) => {
      const lines = [];
      if (r.divine_names && r.divine_names.length) lines.push(`Mentions: ${r.divine_names.slice(0,3).join(", ")}`);
      if (r.formula_markers && r.formula_markers.length) lines.push(`Formula: ${r.formula_markers.slice(0,3).join(", ")}`);
      if (r.institutions && r.institutions.length) lines.push(`Institution: ${r.institutions.slice(0,3).join(", ")}`);
      if (!lines.length && r.topics && r.topics.length) lines.push(`Topics: ${r.topics.join(", ")}`);
      return lines.join(" · ");
    };

    const trustLine = (r) => {
      const sigs = [];
      if (r.citations && r.citations.length) sigs.push("citations");
      if (r.formula_markers && r.formula_markers.length) sigs.push("formula");
      if (r.institutions && r.institutions.length) sigs.push("institution");
      return `Sources: ${r.citations ? r.citations.length : 0} citation(s) · Signals: ${sigs.join(" + ") || "none"}`;
    };

    let activeChip = null;

    function renderKeywordChips(summary, onChange) {
      const wrap = document.createElement("div");
      wrap.className = "chips";
      const keywords = (summary && summary.keywords) ? summary.keywords : [];
      if (!keywords.length) return wrap;

      keywords.forEach((kw) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = kw;
        chip.onclick = () => {
          activeChip = (activeChip === kw) ? null : kw;
          [...wrap.querySelectorAll(".chip")].forEach(el => {
            el.classList.toggle("active", el.textContent === activeChip);
          });
          if (typeof onChange === "function") onChange(activeChip);
        };
        wrap.appendChild(chip);
      });
      return wrap;
    }

    function matchesChip(itemText, itemKind, chip) {
      if (!chip) return true;
      const t = (itemText || "").toLowerCase();
      const k = (itemKind || "").toLowerCase();
      const c = chip.toLowerCase();

      if (c === "joins") return k.includes("reconstruction") || t.includes("hülle") || t.includes("tafel") || t.includes("tablet") || t.includes("envelope") || t.includes("join");
      if (c === "legal") return k.includes("legal") || t.includes("plaintiff") || t.includes("evidence") || t.includes("witness") || t.includes("oath") || t.includes("trial");
      if (c === "chronology") return k.includes("chronology") || t.includes("chronolog") || t.includes("eponym") || t.includes("month") || t.includes("dated");
      if (c === "lexicon" || c === "lexical") return k.includes("lexical") || t.includes("cad");

      return t.includes(c);
    }

    function sourceOriginLabel(sourceRole) {
      if (sourceRole === "primary_text") return "Internal (Primary)";
      if (sourceRole === "archival_wrapper") return "Wrapper context";
      return "Commentary context";
    }

    function evidenceSignals(ev) {
      const src = ev?.source || {};
      const si = src.structural_intelligence || {};
      const profile = si.profile || {};
      const signals = [];
      signals.push({ label: "Section compatibility", ok: !!(profile.template_type && profile.template_type !== "Unknown") });
      signals.push({ label: "Fragment compatibility", ok: (profile.fragmentation || "Unknown") !== "Unknown" });
      signals.push({ label: "Digit pattern", ok: /\d/.test(ev?.quote || "") });
      signals.push({ label: "Formula overlap", ok: /\b(igi|oath|witness|li-?tu-?la|um-ma)\b/i.test(ev?.quote || "") });
      return signals;
    }

    function appendEvidenceDrivingSelection(summary, chip, sections) {
      if (!summary?.evidence?.length || !sections) return;
      const filtered = summary.evidence.filter(ev => matchesChip(ev.quote, ev.source?.doc_type, chip));
      if (!filtered.length) return;

      const selected = filtered[0];
      const selectedSource = selected.source || {};
      const selectedRole = selectedSource.source_role || "scholarly_commentary";
      const selectedContextOnly = selectedRole !== "primary_text";
      const trace = summary.selection_trace || {};
      const block = document.createElement("div");
      block.className = "evidence-driving";
      const selectedPolicy = ((selectedSource.structural_intelligence || {}).routing || {}).selected_policy || "INTERNAL";
      const selectedWeight = Number(selected.rank?.evidence_weight ?? trace.selected_weight ?? 0).toFixed(3);
      block.innerHTML = `
        <h4>Evidence Driving Selection</h4>
        <div class="meta">Selected candidate: ${escapeHtml(selectedSource.pdf_name || "Unknown source")} (p${escapeHtml(String(selectedSource.page_number || "-"))}) | Origin: ${escapeHtml(sourceOriginLabel(selectedRole))} | Policy: ${escapeHtml(String(selectedPolicy).toUpperCase())} | Score: ${selectedWeight}</div>
      `;

      const selectedCard = document.createElement("div");
      selectedCard.className = `evidence-candidate selected ${selectedContextOnly ? "context-only" : ""}`.trim();
      const signals = evidenceSignals(selected);
      selectedCard.innerHTML = `
        <div class="candidate-title">
          <span class="tag">Selected</span>
          ${selectedContextOnly ? '<span class="tag context-only">Context only source</span>' : '<span class="tag source-primary">Translation source</span>'}
        </div>
        <div class="candidate-quote">${escapeHtml(selected.quote || "(no excerpt available)")}</div>
        <div class="match-badges">
          ${signals.map(s => `<span class="match-badge ${s.ok ? "" : "off"}">${s.ok ? "Yes" : "No"} ${escapeHtml(s.label)}</span>`).join("")}
        </div>
        <div class="candidate-actions">
          ${selectedSource.page_url ? `<a href="${selectedSource.page_url}" target="_blank" rel="noopener">View source page</a>` : ""}
          ${selectedSource.page_id ? `<a href="#" onclick="openSource('${selectedSource.page_id}', true); return false;">Open source trace</a>` : ""}
        </div>
        <div class="context-note">${escapeHtml(selected.rank?.rank_reason || trace.selected_rank_reason || "")}</div>
      `;
      block.appendChild(selectedCard);

      if (filtered.length > 1) {
        const altWrap = document.createElement("div");
        altWrap.className = "alt-candidates";
        const title = document.createElement("div");
        title.className = "meta";
        title.textContent = "Alternative candidates";
        altWrap.appendChild(title);
        filtered.slice(1, 4).forEach((ev, idx) => {
          const src = ev.source || {};
          const role = src.source_role || "scholarly_commentary";
          const contextOnly = role !== "primary_text";
          const det = document.createElement("details");
          det.innerHTML = `
            <summary>${escapeHtml(src.pdf_name || `Candidate ${idx + 2}`)} (p${escapeHtml(String(src.page_number || "-"))}) | ${escapeHtml(sourceOriginLabel(role))}</summary>
            <div class="evidence-candidate ${contextOnly ? "context-only" : ""}">
              <div class="candidate-quote">${escapeHtml(ev.quote || "(no excerpt available)")}</div>
              <div class="context-note">${escapeHtml(ev.rank?.rank_reason || "")}</div>
            </div>
          `;
          altWrap.appendChild(det);
        });
        block.appendChild(altWrap);
      }

      sections.appendChild(block);
    }

    function appendConfidenceBreakdown(summary, sections) {
      if (!summary?.confidence || !sections) return;
      const conf = summary.confidence;
      const box = document.createElement("div");
      box.className = "confidence-breakdown";
      const reasons = Array.isArray(conf.reasons) ? conf.reasons : [];
      box.innerHTML = `<h4>Confidence Components</h4><div class="meta">Overall: ${escapeHtml(String(conf.level || "Unknown"))} (${escapeHtml(String(conf.score ?? ""))})</div>`;
      if (reasons.length) {
        const ul = document.createElement("ul");
        reasons.forEach((r) => {
          const li = document.createElement("li");
          li.textContent = r;
          ul.appendChild(li);
        });
        box.appendChild(ul);
      }
      sections.appendChild(box);
    }

    function renderKeyPointsAndEvidence(summary, chip) {
      const sections = document.getElementById("evidence-sections");
      if (!sections) return;
      sections.innerHTML = "";
      renderHumanExplanation(summary, chip);
      appendEvidenceDrivingSelection(summary, chip, sections);

      if (summary.key_points && summary.key_points.length) {
        const kpWrap = document.createElement("div");
        kpWrap.className = "evidence-keypoints";
        const h4 = document.createElement("h4");
        h4.textContent = chip ? `Key Points (filtered: ${chip})` : "Key Points";
        kpWrap.appendChild(h4);

        const ul = document.createElement("ul");
        summary.key_points
          .filter(kp => matchesChip(kp.text, kp.kind || kp.label, chip))
          .forEach(kp => {
            const li = document.createElement("li");
            const badge = document.createElement("span");
            badge.className = "tag kp-badge";
            badge.textContent = kp.label || "Key point";
            li.appendChild(badge);
            li.appendChild(document.createTextNode(" "));
            const span = document.createElement("span");
            span.textContent = kp.text;
            li.appendChild(span);
            if (kp.plain_english) {
              const pe = document.createElement("div");
              pe.className = "plain-english";
              const ctx = document.createElement("div");
              ctx.className = "pe-context";
              ctx.innerHTML = `<strong>What this is about:</strong> ${kp.plain_english.context}`;
              const para = document.createElement("div");
              para.className = "pe-paraphrase";
              para.innerHTML = `<strong>Plain English:</strong> ${kp.plain_english.paraphrase}`;
              const confNote = document.createElement("div");
              confNote.className = "pe-confidence";
              confNote.textContent = kp.plain_english.confidence_note || "";
              pe.appendChild(ctx);
              pe.appendChild(para);
              pe.appendChild(confNote);
              li.appendChild(pe);
            }
            if (kp.citations && kp.citations.length) {
              const cite = kp.citations.map(c => c.pdf_name ? `${c.pdf_name} p.${c.page_number}` : c.page_id).join("; ");
              const citeSpan = document.createElement("span");
              citeSpan.className = "citation-list";
              citeSpan.textContent = ` (${cite})`;
              li.appendChild(citeSpan);
            }
            ul.appendChild(li);
          });

        kpWrap.appendChild(ul);
        sections.appendChild(kpWrap);
      }

      if (summary.evidence && summary.evidence.length) {
        const evWrap = document.createElement("div");
        evWrap.className = "evidence-quotes";
        const h4e = document.createElement("h4");
        h4e.textContent = "Supporting references";
        evWrap.appendChild(h4e);

        summary.evidence
          .filter(ev => matchesChip(ev.quote, ev.source?.doc_type, chip))
          .forEach((ev, idx) => {
            const det = document.createElement("details");
            det.className = "evidence-item";
            const sm = document.createElement("summary");
            sm.textContent = ev.source?.pdf_name ? `${ev.source.pdf_name} – page ${ev.source.page_number}` : `Source ${idx+1}`;
            det.appendChild(sm);
            const qb = document.createElement("blockquote");
            qb.textContent = ev.quote || "(no excerpt available)";
            det.appendChild(qb);
            if (ev.source?.page_url) {
              const a = document.createElement("a");
              a.href = ev.source.page_url;
              a.target = "_blank";
              a.rel = "noopener";
              a.textContent = "View source page";
              det.appendChild(a);
            }
              const sourceRole = ev.source?.source_role || "scholarly_commentary";
              det.className = `evidence-item ${sourceRole === "primary_text" ? "" : "context-only"}`.trim();
              if (sourceRole !== "primary_text") {
                const roleTag = document.createElement("div");
                roleTag.className = "context-note";
                roleTag.textContent = "Context-only source: used for routing support and linkage, not direct translation text.";
                det.appendChild(roleTag);
              }
              const canTranslate = isTranslationAllowed(ev.source);
              if (ev.source?.page_id && canTranslate) {
                const btn = document.createElement("button");
                btn.textContent = "Translate this text";
                btn.style.marginLeft = "10px";
                btn.style.background = "#1f2c4d";
                btn.style.color = "#b6c8ff";
                btn.style.border = "none";
                btn.style.padding = "4px 8px";
                btn.style.borderRadius = "4px";
                btn.style.cursor = "pointer";
                btn.onclick = () => openHandoff(ev.source.page_id, ev.quote);
                det.appendChild(btn);
              } else if (ev.source?.page_id) {
                const note = document.createElement("div");
                note.className = "meta";
                note.textContent = "This source discusses documents. It does not contain a document suitable for translation.";
                det.appendChild(note);

                const linkBtn = document.createElement("button");
                linkBtn.textContent = translationReplacementLabel(sourceRole);
                linkBtn.style.marginLeft = "10px";
                linkBtn.style.background = "#1f2c4d";
                linkBtn.style.color = "#b6c8ff";
                linkBtn.style.border = "none";
                linkBtn.style.padding = "4px 8px";
                linkBtn.style.borderRadius = "4px";
                linkBtn.style.cursor = "pointer";
                linkBtn.onclick = () => openSource(ev.source.page_id, true);
                det.appendChild(linkBtn);
              }
            evWrap.appendChild(det);
          });
        sections.appendChild(evWrap);
      }

      appendConfidenceBreakdown(summary, sections);
    }

    function renderHumanExplanation(summary, chip) {
      const box = document.getElementById("interp-box");
      if (!box) return;
      const byKeyword = (summary.human_explanation_by_keyword || (summary.human_explanation && summary.human_explanation.human_explanation_by_keyword) || {});
      const interp = chip && byKeyword[chip] ? byKeyword[chip] : summary.human_explanation;
      if (!interp) {
        box.innerHTML = "";
        return;
      }
      box.innerHTML = "";
      const h4 = document.createElement("h4");
      h4.textContent = "What you're looking at";
      box.appendChild(h4);

      const addRow = (label, value) => {
        if (!value) return;
        const row = document.createElement("div");
        row.className = "interp-row";
        const strong = document.createElement("strong");
        strong.textContent = label + ": ";
        row.appendChild(strong);
        row.appendChild(document.createTextNode(value));
        box.appendChild(row);
      };

      addRow("What you're looking at", interp.what_you_are_looking_at);
      addRow("Main claim", interp.main_claim);
      addRow("Why it matters", interp.why_it_matters);
      addRow("Uncertainty", interp.uncertainty);
      addRow("Copy note", interp.copy_note);
      addRow("Translation note", interp.translated_note);

      if (interp.next_actions && interp.next_actions.length) {
        const actions = document.createElement("div");
        actions.className = "interp-actions";
        interp.next_actions.forEach((a) => {
          const link = document.createElement("a");
          link.href = `/corpus/citation?ref=${encodeURIComponent(a.ref)}&limit=5`;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = `${a.label} (${a.ref})`;
          actions.appendChild(link);
        });
        box.appendChild(actions);
      }
    }

    const buildCards = (results, kind, mode, baselineResults = [], activePageId = null, sectionKey = "") => {
      const topCompare = compareTopSelectionV2(baselineResults || [], results || []);
      const currentTopId = topCompare.currentTop ? String(topCompare.currentTop.page_id || "") : "";
      return results.map((r, idx) => {
        const url = r.page_url || `/corpus/page/${r.page_id}/record`;
        const badge = sourceBadge(r);
        const docLabel = docTypeLabel(r.doc_type);
        const docTipTitle = r.doc_type === "legal" ? "Legal text" : r.doc_type === "letter" ? "Letter" : "";
        const docTipBody = r.doc_type === "legal"
          ? "A formal document type used for contracts, obligations, loans, guarantees, and settlements. Legal texts follow standardized formulas, list witnesses, and often invoke divine authority to enforce agreements."
          : r.doc_type === "letter"
          ? "A private or semi-formal correspondence between individuals. Letters often mix personal language with legal formulas, making them valuable for understanding how institutions and oaths were used in everyday practice."
          : "";
        const srcType = sourceTypeLabel(r.source_role);
        const si = structuralOf(r);
        const routing = routingOf(r) || {};
        const policy = (routing.selected_policy || "INTERNAL").toUpperCase();
        const modeScore = scoreForMode(r, mode);
        const conf = confidenceBandV2(routing);
        const isTop = String(r.page_id || "") === currentTopId;
        const isActive = String(r.page_id || "") === String(activePageId || "");
        const changedTop = topCompare.changed && isTop;
        const cardClass = changedTop ? "card changed-card" : "card";
        const activeClass = isActive ? " active-evidence-card" : "";
        const changedTag = changedTop ? `<span class="tag changed">Changed from Internal baseline</span>` : "";
        const roleContextTag = r.source_role === "primary_text" ? "" : `<span class="tag context-only">Context only</span>`;
        const whySelected = methodExplanationV2(r);
        const diffLine = (idx === 0)
          ? `<div class="result-diff">Selection change: ${topCompare.changed ? "YES" : "NO"} | baseline ${escapeHtml(String(topCompare.basePage || "-"))} -> current ${escapeHtml(String(topCompare.currentPage || "-"))}</div>`
          : "";
        const thumbUrl = r.asset_thumb_url || r.pdf_render_url || "";
        const fullAssetUrl = r.asset_full_url || thumbUrl || r.external_visual_url || url;
        const tagActive = isTaggedV2(r.page_id);
        const noteText = tagNoteV2(r.page_id);
        return `
          <div class="${cardClass}${activeClass} evidence-card" data-page-id="${escapeHtml(String(r.page_id || ""))}" data-section="${escapeHtml(sectionKey || "")}" data-kind="${escapeHtml(kind || "")}">
            <div class="card-head">
              <div class="meta">
                <span class="tag" data-tip-title="Source role" data-tip="Primary text = original letters/contracts. Archival wrapper = envelopes, seals, witness lists that surround the text. Scholarly commentary = analysis, joins, or reconstructions. We show Primary first because it is closest to the original evidence.">${badge}</span>
                <span class="tag" ${docTipTitle ? `data-tip-title="${docTipTitle}"` : ""} ${docTipBody ? `data-tip="${docTipBody}"` : ""}>${docLabel}</span>
                <span class="tag ${srcType.cls}" data-tip-title="Source type" data-tip="Source role controls translation access. Primary texts are translatable; archival wrappers and scholarly commentary provide context and metadata.">${srcType.text}</span>
                <span class="tag" data-tip-title="Routing policy" data-tip="${escapeHtml(whySelected)}">Policy: ${escapeHtml(policy)}</span>
                <span class="confidence-pill ${conf.cls}">${conf.level}</span>
                ${changedTag}
                ${roleContextTag}
              </div>
              ${thumbUrl
                ? `<a class="card-thumb-link" href="${escapeHtml(fullAssetUrl)}" target="_blank" rel="noopener" title="Open artifact image"><img src="${escapeHtml(thumbUrl)}" alt="Artifact thumbnail"></a>`
                : `<a class="card-thumb-link card-thumb-fallback" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="Open source page">DOC</a>`}
            </div>
            ${diffLine}
            <div class="meta">${metaLine(r)}</div>
            ${structuralBadgesHtmlV2(r)}
            <div class="snippet">${markTerm(r.snippet || "")}</div>
            <div class="meta"><a href="${url}" target="_blank">${r.pdf_name} (p${r.page_number})</a></div>
            <div class="meta" style="font-size:11px;color:#8fb0e4;">Mode score: ${Number(modeScore).toFixed(2)} | Internal ${Number(routing.internal_score ?? r.evidence_weight ?? 0).toFixed(2)} | ORACC ${Number(routing.oracc_score ?? r.evidence_weight ?? 0).toFixed(2)}</div>
            <div class="card-actions">
              <button type="button" class="chip-btn ${tagActive ? "active" : ""} btn-tag-toggle" data-page-id="${escapeHtml(String(r.page_id || ""))}" data-pdf-name="${escapeHtml(r.pdf_name || "")}" data-page-number="${escapeHtml(String(r.page_number ?? ""))}">${tagActive ? "Tagged" : "Add Tag"}</button>
              <button type="button" class="chip-btn btn-tag-note" data-page-id="${escapeHtml(String(r.page_id || ""))}" data-pdf-name="${escapeHtml(r.pdf_name || "")}" data-page-number="${escapeHtml(String(r.page_number ?? ""))}">${noteText ? "Edit Note" : "Add Note"}</button>
              ${noteText ? `<span class="tag" title="${escapeHtml(noteText)}">Note: ${escapeHtml(String(noteText).slice(0, 42))}${noteText.length > 42 ? "..." : ""}</span>` : ""}
            </div>
            ${routingTraceHtmlV2(si)}
            <div class="meta" style="font-size:11px;color:#7ea6d6;">
              ${trustLine(r)}
              <button class="help" type="button"
                data-tip-title="Sources (${r.citations ? r.citations.length : 0} citations)"
                data-tip="The number of distinct citations detected on this page (e.g., ICK, CCT, BIN, AKT, Kt…). Higher counts usually mean the page is a strong hub connecting this topic to published primary texts."
                aria-label="Explain sources count">?</button>
              <button class="help" type="button"
                data-tip-title="Signals"
                data-tip="Signals are the features our pipeline detected on this page. Citations = links to specific editions/text IDs. Formula = repeated phrase patterns (oath/witness language). Institution = economic/legal vocabulary (e.g., naruqqum, karum). More signals generally means higher confidence."
                aria-label="Explain signals">?</button>
              · <a href="#" class="btn-view-sources" data-page-id="${r.page_id}" data-kind="${kind}">View sources</a>
            </div>
          </div>
        `;
      }).join("");
    };

    const renderStats = (data) => {
      if (!data || !data.ok) return;
      const p = data.percentages;
      const c = data.counts;
      const total = c.total || 0;
      const stat = (label, key, tipTitle, tipBody) => `
        <div class="stat-card help clickable-stat" data-stat-key="${key}" data-tip-title="${tipTitle}" data-tip="${tipBody}">
          <button class="stat-help" type="button" data-tip-title="${tipTitle}" data-tip="${tipBody}">?</button>
          <div class="stat-label">${label}</div>
          <div class="stat-value">${p[key]}%</div>
          <div class="stat-label">${c[key]} pages</div>
        </div>`;
      statsEl.innerHTML = [
        `<div class="stat-card help clickable-stat" data-stat-key="total" data-tip-title="Corpus indexed" data-tip="The total number of pages the system has scanned and analyzed. Each page is checked for names, formulas, and institutions so it can be searched and compared."><button class="stat-help" type="button" data-tip-title="Corpus indexed" data-tip="The total number of pages the system has scanned and analyzed. Each page is checked for names, formulas, and institutions so it can be searched and compared.">?</button><div class="stat-label">Corpus indexed</div><div class="stat-value">${total}</div></div>`,
        stat("Pages with traceable citations", "citations", "Pages with traceable citations", "Pages that contain clear references to specific texts or editions (like ICK, CCT, AKT). These links let us trace claims back to known sources."),
        stat("Pages with recognizable formulas", "formula_markers", "Pages with recognizable formulas", "Pages that include repeated legal phrases, such as oath or witness language. These formulas help identify contracts, promises, and official statements."),
        stat("Pages mentioning institutions", "institutions", "Pages mentioning institutions", "Pages that include words for organized systems like trade partnerships, loans, or courts (e.g., naruqqum). These show how people worked together in business and law."),
        stat("Pages classified by type", "doc_type_classified", "Pages classified by type", "Pages the system could confidently label as letters, legal documents, commentary, or reference material. Knowing the type helps us decide how strong the evidence is."),
      ].join("");
      statsEl.querySelectorAll(".clickable-stat").forEach((el) => {
        el.addEventListener("click", (evt) => {
          if (closestFromTargetV2(evt.target, ".stat-help")) return;
          const key = el.getAttribute("data-stat-key") || "";
          focusByStatV2(key);
        });
      });
    };

    const renderModeButtons = () => {
      document.querySelectorAll("#view-mode-toggle .mode-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === currentViewMode);
      });
    };

    const renderDemo = (data) => {
      demoCache = data;
      renderModeButtons();

      const deityBase = data.deity_attestations.results || [];
      const formulaBase = data.formula_examples.results || [];
      const instBase = data.institution_examples.results || [];

      const deityView = applyStructuralFilters(applyViewMode(deityBase, currentViewMode));
      const formulaView = applyStructuralFilters(applyViewMode(formulaBase, currentViewMode));
      const instView = applyStructuralFilters(applyViewMode(instBase, currentViewMode));

      const deityInternalView = applyStructuralFilters(applyViewMode(deityBase, "internal"));
      const formulaInternalView = applyStructuralFilters(applyViewMode(formulaBase, "internal"));
      const instInternalView = applyStructuralFilters(applyViewMode(instBase, "internal"));

      const deityDelta = selectionDelta(deityInternalView, deityView);
      const formulaDelta = selectionDelta(formulaInternalView, formulaView);
      const instDelta = selectionDelta(instInternalView, instView);

      updateRoutingAnalytics([deityView, formulaView, instView]);
      const groups = [
        { sectionKey: "belief", section: "Belief", kind: "deity", current: deityView, internal: deityInternalView },
        { sectionKey: "speech", section: "Speech", kind: "formula", current: formulaView, internal: formulaInternalView },
        { sectionKey: "behavior", section: "Behavior", kind: "institution", current: instView, internal: instInternalView },
      ];
      const activeState = resolveActiveFocusV2(groups, currentViewMode);
      const headerFocus = activeState.focus;
      const cohesiveHeaderHtml = renderCohesiveHeaderV2(headerFocus, currentViewMode, {
        scope: navScope,
        index: activeState.navIndex,
        total: activeState.navTotal,
        selectionReason: activeState.selectionReason,
      });
      const activeId = activeEvidencePageId;

      panelsEl.innerHTML = `
        ${cohesiveHeaderHtml}
        <div class="panel">
          <h2>Belief · Deity evidence <span class="term">istarzaat <button class="help" type="button" data-tip-title="Ištar-ZA.AT (istarzaat)" data-tip="A specific way Old Assyrian texts refer to the goddess Ištar, often in formal/legal contexts. In this demo, it’s used as a “belief signal” because it shows where divine authority is invoked in real documents (oaths, contracts, witness statements)." aria-label="What is istarzaat?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.deity_attestations.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${deityDelta}</div>
          ${renderDecisionPanelV2("Belief", deityView, deityInternalView, currentViewMode)}
          ${buildCards(deityView, "deity", currentViewMode, deityInternalView, activeId, "belief")}
        </div>
        <div class="panel">
          <h2>Speech · Oath / witness formulas <span class="term">li-tù-la <button class="help" type="button" data-tip-title="li-tù-la" data-tip="A common Old Assyrian “formula marker” meaning something like “may X be witnesses / may X see.” In this demo, it’s a “speech signal” because it highlights standardized legal language used in letters and contracts." aria-label="What is li-tù-la?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.formula_examples.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${formulaDelta}</div>
          ${renderDecisionPanelV2("Speech", formulaView, formulaInternalView, currentViewMode)}
          ${buildCards(formulaView, "formula", currentViewMode, formulaInternalView, activeId, "speech")}
        </div>
        <div class="panel">
          <h2>Behavior · Economic institutions <span class="term">naruqqum <button class="help" type="button" data-tip-title="naruqqum" data-tip="A joint-stock / pooled-capital partnership used in Old Assyrian commerce. In this demo, it’s a “behavior signal” because it points to how trade was organized (investment, risk, obligations), not just what people said." aria-label="What is naruqqum?">?</button></span></h2>
          <div class="meta" style="margin-bottom:6px;">Showing top ${data.institution_examples.count_returned} (mode: ${currentViewMode.toUpperCase()})</div>
          <div class="delta-note">Δ exemplar selection vs Internal: ${instDelta}</div>
          ${renderDecisionPanelV2("Behavior", instView, instInternalView, currentViewMode)}
          ${buildCards(instView, "institution", currentViewMode, instInternalView, activeId, "behavior")}
        </div>
        <div class="panel">
          <h2>User Provided</h2>
          <p class="meta">Paste text (preview). Demo runs lightweight extraction only.</p>
          <textarea id="userText" placeholder="Paste text..."></textarea>
          <button onclick="analyze()">Analyze</button>
          <div id="userResults" class="card" style="margin-top:8px; display:none;"></div>
        </div>
      `;
      panelsEl.querySelectorAll(".btn-header-open-source").forEach((el) => {
        el.addEventListener("click", (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          const pageId = el.getAttribute("data-page-id") || "";
          if (!pageId) return;
          openSource(pageId, true);
        });
      });
      panelsEl.querySelectorAll(".btn-header-open-primary").forEach((el) => {
        el.addEventListener("click", (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          const pageId = el.getAttribute("data-page-id") || "";
          const ref = el.getAttribute("data-ref") || "";
          if (!pageId || !ref) return;
          openSource(pageId, true, () => {
            fetchPrimaryExcerptFromRef(ref, "");
          });
        });
      });
      const navPrevBtn = panelsEl.querySelector("#btn-nav-prev");
      if (navPrevBtn) {
        navPrevBtn.addEventListener("click", () => moveActiveEvidenceV2(-1));
      }
      const navNextBtn = panelsEl.querySelector("#btn-nav-next");
      if (navNextBtn) {
        navNextBtn.addEventListener("click", () => moveActiveEvidenceV2(1));
      }
      const navScopeSelect = panelsEl.querySelector("#nav-scope-select");
      if (navScopeSelect) {
        navScopeSelect.value = navScope;
        navScopeSelect.addEventListener("change", () => {
          navScope = navScopeSelect.value || "all";
          activeSelectionSource = "navigation";
          pendingActiveScroll = false;
          saveUiStateV2();
          if (demoCache) renderDemo(demoCache);
        });
      }
      const navSearchInput = panelsEl.querySelector("#nav-search-input");
      const navSearchBtn = panelsEl.querySelector("#btn-nav-search");
      const navGotoInput = panelsEl.querySelector("#nav-goto-item");
      const navGotoBtn = panelsEl.querySelector("#btn-nav-goto");
      if (navSearchInput) {
        navSearchInput.addEventListener("keydown", (evt) => {
          if (evt.key !== "Enter") return;
          evt.preventDefault();
          runEvidenceSearchV2(navSearchInput.value || "");
        });
      }
      if (navSearchBtn && navSearchInput) {
        navSearchBtn.addEventListener("click", () => runEvidenceSearchV2(navSearchInput.value || ""));
      }
      if (navGotoInput) {
        navGotoInput.addEventListener("keydown", (evt) => {
          if (evt.key !== "Enter") return;
          evt.preventDefault();
          gotoActiveEvidenceIndexV2(navGotoInput.value || "");
        });
      }
      if (navGotoBtn && navGotoInput) {
        navGotoBtn.addEventListener("click", () => gotoActiveEvidenceIndexV2(navGotoInput.value || ""));
      }
      const preferPrimaryBtn = panelsEl.querySelector("#btn-toggle-prefer-primary");
      if (preferPrimaryBtn) {
        preferPrimaryBtn.addEventListener("click", () => {
          preferPrimary = !preferPrimary;
          savePreferPrimaryV2();
          activeSelectionSource = "auto";
          pendingActiveScroll = false;
          if (demoCache) renderDemo(demoCache);
        });
      }
      const tagCurrentBtn = panelsEl.querySelector("#btn-tag-current");
      if (tagCurrentBtn) {
        tagCurrentBtn.addEventListener("click", () => {
          const pageId = tagCurrentBtn.getAttribute("data-page-id") || "";
          const meta = {
            pdf_name: tagCurrentBtn.getAttribute("data-pdf-name") || "",
            page_number: tagCurrentBtn.getAttribute("data-page-number") || "",
          };
          if (isTaggedV2(pageId)) {
            removeTagV2(pageId);
          } else {
            upsertTagV2(pageId, meta);
          }
          if (demoCache) renderDemo(demoCache);
        });
      }
      const noteCurrentBtn = panelsEl.querySelector("#btn-note-current");
      if (noteCurrentBtn) {
        noteCurrentBtn.addEventListener("click", () => {
          const pageId = noteCurrentBtn.getAttribute("data-page-id") || "";
          const meta = {
            pdf_name: noteCurrentBtn.getAttribute("data-pdf-name") || "",
            page_number: noteCurrentBtn.getAttribute("data-page-number") || "",
          };
          const prev = tagNoteV2(pageId);
          const next = prompt("Tag note for this artifact:", prev || "");
          if (next === null) return;
          upsertTagV2(pageId, meta, next);
          if (demoCache) renderDemo(demoCache);
        });
      }
      const taggedSelect = panelsEl.querySelector("#tagged-select");
      const loadTaggedBtn = panelsEl.querySelector("#btn-load-tagged");
      if (loadTaggedBtn && taggedSelect) {
        loadTaggedBtn.addEventListener("click", () => {
          const pageId = taggedSelect.value || "";
          if (!pageId) return;
          const inCurrent = (lastNavEntriesAll || []).some((x) => String(x.currentTop?.page_id || "") === String(pageId));
          if (inCurrent) {
            setActiveEvidenceByIdV2(pageId, "navigation", true);
          } else {
            openSource(pageId, true);
          }
        });
      }
      panelsEl.querySelectorAll(".btn-tag-toggle").forEach((btn) => {
        btn.addEventListener("click", (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          const pageId = btn.getAttribute("data-page-id") || "";
          const meta = {
            pdf_name: btn.getAttribute("data-pdf-name") || "",
            page_number: btn.getAttribute("data-page-number") || "",
          };
          if (isTaggedV2(pageId)) {
            removeTagV2(pageId);
          } else {
            upsertTagV2(pageId, meta);
          }
          if (demoCache) renderDemo(demoCache);
        });
      });
      panelsEl.querySelectorAll(".btn-tag-note").forEach((btn) => {
        btn.addEventListener("click", (evt) => {
          evt.preventDefault();
          evt.stopPropagation();
          const pageId = btn.getAttribute("data-page-id") || "";
          const meta = {
            pdf_name: btn.getAttribute("data-pdf-name") || "",
            page_number: btn.getAttribute("data-page-number") || "",
          };
          const prev = tagNoteV2(pageId);
          const next = prompt("Tag note for this artifact:", prev || "");
          if (next === null) return;
          upsertTagV2(pageId, meta, next);
          if (demoCache) renderDemo(demoCache);
        });
      });
      panelsEl.querySelectorAll(".evidence-card").forEach((card) => {
        card.addEventListener("click", (evt) => {
          if (closestFromTargetV2(evt.target, "a,button,summary,input,textarea,select")) return;
          const pageId = card.getAttribute("data-page-id") || "";
          if (!pageId) return;
          setActiveEvidenceByIdV2(pageId, "navigation", true);
        });
      });
      if (pendingActiveScroll) {
        const activeCard = panelsEl.querySelector(".evidence-card.active-evidence-card");
        if (activeCard) {
          activeCard.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        pendingActiveScroll = false;
      }
    };

    document.querySelectorAll("#view-mode-toggle .mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const nextMode = btn.dataset.mode || "routed";
        if (!nextMode || nextMode === currentViewMode) return;
        currentViewMode = nextMode;
        renderModeButtons();
        if (demoCache) renderDemo(demoCache);
      });
    });

    [filterFragmentationEl, filterTemplateEl, filterPolicyEl, filterOriginEl].forEach((el) => {
      if (!el) return;
      el.addEventListener("change", () => {
        if (demoCache) renderDemo(demoCache);
      });
    });

    panelsEl.addEventListener("click", (evt) => {
      const sourceLink = closestFromTargetV2(evt.target, ".btn-view-sources");
      if (!sourceLink) return;
      evt.preventDefault();
      evt.stopPropagation();
      const fromCard = closestFromTargetV2(sourceLink, ".evidence-card");
      const pageId = sourceLink.getAttribute("data-page-id") || fromCard?.getAttribute("data-page-id") || "";
      const kind = sourceLink.getAttribute("data-kind") || "";
      if (!pageId) return;
      openDrawer(pageId, kind);
    });

    sourceContent.addEventListener("click", (evt) => {
      const ladderStep = closestFromTargetV2(evt.target, ".paper-ladder .ladder-step");
      if (ladderStep) {
        const recordUrl = ladderStep.getAttribute("data-open-record-url");
        const canJump = ladderStep.hasAttribute("data-jump-primary");
        if (ladderStep.classList.contains("disabled") || (!recordUrl && !canJump)) return;
        evt.preventDefault();
        evt.stopPropagation();
        if (recordUrl) {
          window.open(recordUrl, "_blank", "noopener");
          return;
        }
        if (canJump) {
          jumpToPrimaryFromCurrentSource();
          return;
        }
      }

      const tagBtn = closestFromTargetV2(evt.target, ".btn-source-tag-toggle");
      if (tagBtn) {
        evt.preventDefault();
        evt.stopPropagation();
        const sourcePageId = tagBtn.getAttribute("data-page-id") || "";
        const meta = {
          pdf_name: tagBtn.getAttribute("data-pdf-name") || "",
          page_number: tagBtn.getAttribute("data-page-number") || "",
        };
        if (!sourcePageId) return;
        if (isTaggedV2(sourcePageId)) {
          removeTagV2(sourcePageId);
        } else {
          upsertTagV2(sourcePageId, meta);
        }
        openSource(sourcePageId, true);
        if (demoCache) renderDemo(demoCache);
        return;
      }

      const noteBtn = closestFromTargetV2(evt.target, ".btn-source-note");
      if (noteBtn) {
        evt.preventDefault();
        evt.stopPropagation();
        const sourcePageId = noteBtn.getAttribute("data-page-id") || "";
        const meta = {
          pdf_name: noteBtn.getAttribute("data-pdf-name") || "",
          page_number: noteBtn.getAttribute("data-page-number") || "",
        };
        if (!sourcePageId) return;
        const prev = tagNoteV2(sourcePageId);
        const next = prompt("Tag note for this artifact:", prev || "");
        if (next === null) return;
        upsertTagV2(sourcePageId, meta, next);
        openSource(sourcePageId, true);
        if (demoCache) renderDemo(demoCache);
      }
    });

    function analyze() {
      const txt = document.getElementById("userText").value || "";
      const box = document.getElementById("userResults");
      if (!txt.trim()) { box.style.display='none'; return; }
      box.style.display='block';
      box.innerHTML = `<div class="meta">What we recognize in this text</div><div class="snippet">${markTerm(txt.substring(0,300))}${txt.length>300?'...':''}</div><div class="meta" style="font-size:11px;color:#7ea6d6;">Status: not yet matched to the corpus</div>`;
    }

    // Evidence summary renderer
    function renderEvidenceSummary(summary) {
      const root = document.getElementById("evidence-summary");
      if (!root) return;
      root.innerHTML = "";
      if (!summary || !summary.summary) {
        root.innerHTML = "<p class='meta'>No evidence summary available.</p>";
        return;
      }
      const block = document.createElement("div");
      block.className = "evidence-summary-block";
      const title = document.createElement("h3");
      title.textContent = "Evidence Summary - Decision Trace";
      block.appendChild(title);
      const p = document.createElement("p");
      p.textContent = summary.summary;
      block.appendChild(p);
      const sub = document.createElement("div");
      sub.className = "meta";
      sub.textContent = "This section explains which source was selected, why it was selected, and what alternatives were considered.";
      block.appendChild(sub);
      root.appendChild(block);
      const interpBox = document.createElement("div");
      interpBox.id = "interp-box";
      interpBox.className = "interp-card";
      root.appendChild(interpBox);
      renderHumanExplanation(summary, null);
      const chipsEl = renderKeywordChips(summary, (chip) => {
        renderKeyPointsAndEvidence(summary, chip);
      });
      root.appendChild(chipsEl);

      const sections = document.createElement("div");
      sections.id = "evidence-sections";
      root.appendChild(sections);

      renderKeyPointsAndEvidence(summary, null);
    }

    function openDrawer(pageId, kind) {
      currentPage = pageId;
      // If Source Explorer is open, close it so Citation Explorer can be visible.
      sourcePanel.classList.remove("open");
      sourcePanel.style.right = "-520px";
      drawer.classList.add("open");
      drawer.style.right = "0";
      drawerList.innerHTML = "<div class='meta'>Loading citations...</div>";
      drawerDetail.innerHTML = "Select a citation to see context.";
      currentCitationList = [];
      currentCitationIndex = -1;
      fetch(`/corpus/page/${pageId}/citations`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) throw new Error("No citations");
          const srcMethod = data.source_page?.structural_intelligence?.routing || {};
          const srcPolicy = (srcMethod.selected_policy || "INTERNAL").toUpperCase();
          drawerSource.innerHTML = `${data.source_page.pdf_name} (p${data.source_page.page_number}) | ${data.source_page.doc_type || "unknown"} | weight ${data.source_page.evidence_weight} | policy ${srcPolicy} <button class="help" type="button" data-tip-title="Source line" data-tip="Current page context for this citation list: document, page, evidence weight, and active routing policy.">?</button>`;
          currentCitationList = data.citations || [];
          drawerList.innerHTML = currentCitationList.map((c, idx) => {
            return `<div class="list-item" onclick="loadCitation('${encodeURIComponent(c.ref)}', ${idx})">
              <div><strong>${c.ref}</strong></div>
              <div class="meta">${(c.ref_type || "unknown").toUpperCase()} · ${c.doc_hint || ""}</div>
            </div>`;
          }).join("");
        })
        .catch(err => {
          drawerList.innerHTML = `<div class='meta'>Citation explorer not available. Try another item or open the source page.</div>`;
        });
    }

    function closeDrawer() {
      drawer.classList.remove("open");
      currentPage = null;
      drawer.style.right = "-480px";
    }

      function closeSource() {
        sourcePanel.classList.remove("open");
        sourcePanel.style.right = "-520px";
        // Keep drawer state consistent if it is currently open.
        drawer.style.right = drawer.classList.contains("open") ? "0" : "-480px";
        sourceContent.innerHTML = "Select a source to view here.";
        currentSourceResults = [];
        currentSourceIndex = -1;
        currentSourceMeta = null;
      }

      function buildPaperTrailHeader(sourceRole, hasCitations, currentPageId = "") {
        const role = sourceRole || "scholarly_commentary";
        const recordUrl = currentPageId ? `/corpus/page/${encodeURIComponent(currentPageId)}/record` : "/corpus";
        const roleMeta = {
          primary_text: {
            title: "Primary Text (Tablet)",
            blurb: "This is the original Old Assyrian document.",
          },
          archival_wrapper: {
            title: "Archival Wrapper (Envelope / Seal)",
            blurb: "Contains seals, witnesses, and metadata.",
          },
          scholarly_commentary: {
            title: "Scholarly Commentary",
            blurb: "You are reading analysis, not an original document.",
          },
        };
        const meta = roleMeta[role] || roleMeta.scholarly_commentary;
        const gateCopy = role === "primary_text"
          ? ""
          : '<div class="meta gate-copy">This source discusses documents. It does not contain a document suitable for translation.</div>';
        let ctaHtml = "";
        if (role === "primary_text") {
          ctaHtml = `<div class="meta ladder-cta ok">Translation available</div>
            <div class="source-tools-row">
              <a href="${escapeHtml(recordUrl)}" target="_blank" rel="noopener">Open primary source record</a>
            </div>`;
        } else if (hasCitations) {
          const label = role === "archival_wrapper" ? "View enclosed tablet" : "Jump to linked primary text";
          ctaHtml = `<button class="ladder-cta" type="button" onclick="jumpToPrimaryFromCurrentSource()">${label}</button>`;
        } else {
          ctaHtml = '<div class="meta ladder-cta muted">No linked primary text available.</div>';
        }
        const primaryStepAttrs = role === "primary_text"
          ? `data-open-record-url="${escapeHtml(recordUrl)}"`
          : (hasCitations ? `data-jump-primary="1"` : "");
        const primaryStepDisabled = role !== "primary_text" && !hasCitations;
        const ladderHtml = `
          <div class="paper-ladder">
            <div class="ladder-step ${role === "scholarly_commentary" ? "active" : ""}">Scholarly Commentary</div>
            <div class="ladder-arrow">&darr;</div>
            <div class="ladder-step ${role === "archival_wrapper" ? "active" : ""}">Archival Wrapper (Envelope / Seal)</div>
            <div class="ladder-arrow">&darr;</div>
            <button type="button" class="ladder-step ${role === "primary_text" ? "active" : ""} ${primaryStepDisabled ? "disabled" : ""}" ${primaryStepAttrs}>Primary Text (Tablet / Letter)</button>
          </div>
        `;
        return `
          <div class="source-header">
            <div class="role-title">${meta.title}</div>
            <div class="meta">${meta.blurb}</div>
            ${gateCopy}
            ${ctaHtml}
            ${ladderHtml}
          </div>
        `;
      }

      function openSource(pageId, fromList = false, afterRender = null) {
        sourcePanel.classList.add("open");
        sourcePanel.style.right = "0";
        drawer.style.right = "500px";
      sourceContent.innerHTML = "<div class='meta'>Loading source...</div>";
      fetch(`/corpus/page/${pageId}/story`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) { sourceContent.innerHTML = `<div class='meta'>${data.error || "Could not load source."}</div>`; return; }
          const p = data.page || {};
          const sourcePageId = String(p.page_id || pageId || "");
          const sourceRecordUrl = p.page_url || `/corpus/page/${encodeURIComponent(sourcePageId)}/record`;
          const sourceThumbUrl = p.asset_thumb_url || p.pdf_render_url || "";
          const sourceFullAssetUrl = p.asset_full_url || sourceThumbUrl || "";
          const sourceVisualUrl = p.external_visual_url || "";
          const sourceVisualCandidatesUrl = p.external_visual_candidates_url || (sourcePageId ? `/corpus/page/${encodeURIComponent(sourcePageId)}/visuals` : "");
          const sourceVisualCompareUrl = p.visual_compare_url || (sourcePageId ? `/corpus/page/${encodeURIComponent(sourcePageId)}/compare` : "");
          const sourceRole = data.source_role || "scholarly_commentary";
          const anchorHint = (data.snippets && data.snippets.length) ? (data.snippets[0].text || "") : "";
          currentSourceMeta = {
            source_role: sourceRole,
            citations: data.citations || [],
            anchor: anchorHint,
          };
          const sourceTagged = isTaggedV2(sourcePageId);
          const sourceNote = tagNoteV2(sourcePageId);
          const highlights = (data.highlights || []).map(h => `<div class="meta"><strong>${h.label}:</strong> ${markTerm(h.value || "")}</div>`).join("");
          const why = (data.why_this_matters || []).map(w => `<div class="meta">• ${w}</div>`).join("");
          const cits = (data.citations || []).map(c => `<span class="badge">${c.ref}</span>`).join(" ");
          const snips = (data.snippets || []).map(s => `<div class="section"><div class="meta"><strong>${s.topic || ""}</strong></div><div class="snippet">${markTerm(s.text || "")}</div></div>`).join("");
          const breadcrumb = `Belief · Evidence → ${p.title || "Source"} (p${p.page_number || ""})`;
          const header = buildPaperTrailHeader(sourceRole, (data.citations || []).length > 0, sourcePageId);
          const roleBadge = sourceRoleHeaderLabel(sourceRole);
          const docTag = sourceRoleSecondaryTag(sourceRole, p.doc_type || "");
          const headerMeta = `<span class="badge">${roleBadge}</span>${docTag ? ` <span class="badge">${docTag}</span>` : ""}`;
          const structuralHtml = renderStructuralHeaderV2(data.structural_intelligence || null);
          sourceContent.innerHTML = `
            ${header}
                <div class="section">
                  <h4>${breadcrumb}</h4>
                  <div class="meta">${headerMeta}</div>
                </div>
            <div class="section">
              <h4>Source Tools</h4>
              <div class="meta">Open source links and manage revisit tags/notes from this panel.</div>
              <div class="source-tools-row">
                ${sourceFullAssetUrl ? `<a href="${escapeHtml(sourceFullAssetUrl)}" target="_blank" rel="noopener">Open source scan</a>` : `<span class="meta">No local scan image is indexed.</span>`}
                ${(!sourceFullAssetUrl && sourceVisualUrl) ? `<a href="${escapeHtml(sourceVisualUrl)}" target="_blank" rel="noopener">Open likely visual source</a>` : ""}
                ${sourceVisualCandidatesUrl ? `<a href="${escapeHtml(sourceVisualCandidatesUrl)}" target="_blank" rel="noopener">Open visual candidates</a>` : ""}
                ${sourceVisualCompareUrl ? `<a href="${escapeHtml(sourceVisualCompareUrl)}" target="_blank" rel="noopener">Compare source vs visual</a>` : ""}
                <a href="${escapeHtml(sourceRecordUrl)}" target="_blank" rel="noopener">Open source record</a>
                <button type="button" class="chip-btn action-tag ${sourceTagged ? "active" : ""} btn-source-tag-toggle" data-page-id="${escapeHtml(sourcePageId)}" data-pdf-name="${escapeHtml(p.title || "")}" data-page-number="${escapeHtml(String(p.page_number ?? ""))}">${sourceTagged ? "Tagged" : "Add Tag"}</button>
                <button type="button" class="chip-btn action-note btn-source-note" data-page-id="${escapeHtml(sourcePageId)}" data-pdf-name="${escapeHtml(p.title || "")}" data-page-number="${escapeHtml(String(p.page_number ?? ""))}">${sourceNote ? "Edit Note" : "Add Note"}</button>
              </div>
              ${sourceNote ? `<div class="meta">Note: ${escapeHtml(sourceNote)}</div>` : ""}
            </div>
            <div class="section">
              ${structuralHtml}
            </div>
            <div class="section">
              <h4>Highlights</h4>
              ${highlights || "<div class='meta'>None listed.</div>"}
            </div>
            <div class="section">
              <h4>Why this matters</h4>
              ${why || "<div class='meta'>Contextual support.</div>"}
            </div>
            <div class="section">
              <h4>Citations</h4>
              ${cits || "<div class='meta'>None listed.</div>"}
            </div>
            <div class="section">
              <h4>Snippets</h4>
              ${snips || "<div class='meta'>No snippets.</div>"}
            </div>
            <div class="section">
              <h4>Raw view</h4>
              <a href="${escapeHtml(sourceRecordUrl)}" target="_blank" rel="noopener">Open source record</a>
              &nbsp;|&nbsp;
              <a href="/corpus/page/${p.page_id}?include_text=false" target="_blank" rel="noopener">Developer JSON bundle</a>
            </div>
            <div class="section">
              <button onclick="stepSource(-1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:6px 8px;border-radius:4px;cursor:pointer;margin-right:6px;">Prev source</button>
              <button onclick="stepSource(1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:6px 8px;border-radius:4px;cursor:pointer;">Next source</button>
            </div>
          `;
          if (typeof afterRender === "function") afterRender();
        })
          .catch(() => {
            sourceContent.innerHTML = "<div class='meta'>Could not load source.</div>";
          });
      }

      function pickBestRefFromCitations(citations) {
        if (!citations || !citations.length) return null;
        return citations[0].ref || citations[0];
      }

      function appendPrimaryExcerptBlock(handoffData, anchor) {
        if (!handoffData || !handoffData.page_id) return;
        const blockId = `primary-excerpt-${handoffData.page_id}`;
        if (document.getElementById(blockId)) return;
        const excerptFocus = handoffData.excerpt_focus || handoffData.excerpt_short || "";
        const focusLabel = handoffData.anchor_matched ? "Focused excerpt (matched)" : "Focused excerpt (default)";

        const block = document.createElement("div");
        block.className = "section";
        block.id = blockId;
        block.innerHTML = `
          <h4>Primary excerpt (from ${handoffData.pdf_name} p${handoffData.page_number})</h4>
          <div class="meta">${docTypeLabel(handoffData.doc_type)} · Translation available</div>
          <div class="meta" style="margin-top:6px;"><strong>${focusLabel}</strong></div>
          <div class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(excerptFocus)}</div>
          <div class="meta" style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
            <button id="btn-open-primary-${handoffData.page_id}" style="background:#1f2c4d;color:#b6c8ff;">Open primary source</button>
            <button id="btn-translate-primary-${handoffData.page_id}" style="background:#1f2c4d;color:#b6c8ff;">Translate this text</button>
          </div>
        `;
        sourceContent.appendChild(block);

        document.getElementById(`btn-open-primary-${handoffData.page_id}`).onclick = () => {
          openSource(handoffData.page_id, true);
        };
        document.getElementById(`btn-translate-primary-${handoffData.page_id}`).onclick = () => {
          openHandoff(handoffData.page_id, anchor || "");
        };
      }

      function fetchPrimaryExcerptFromRef(ref, anchor) {
        if (!ref) return;
        const anchorParam = anchor ? `&anchor=${encodeURIComponent(anchor)}` : "";
        fetch(`/corpus/resolve/${encodeURIComponent(ref)}?limit=10${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok || !data.best) return;
            const primary = data.best;
            const handoffAnchor = anchor || "";
            const handoffParam = handoffAnchor ? `?anchor=${encodeURIComponent(handoffAnchor)}` : "";
            fetch(`/corpus/handoff/${primary.page_id}${handoffParam}`)
              .then(r => r.json())
              .then(handoff => {
                if (!handoff.ok) return;
                appendPrimaryExcerptBlock(handoff, handoffAnchor);
              })
              .catch(() => {});
          })
          .catch(() => {});
      }

      function jumpToPrimaryFromCurrentSource() {
        if (!currentSourceMeta) return;
        const ref = pickBestRefFromCitations(currentSourceMeta.citations);
        if (!ref) return;
        const anchorParam = currentSourceMeta.anchor ? `&anchor=${encodeURIComponent(currentSourceMeta.anchor)}` : "";
        fetch(`/corpus/resolve/${encodeURIComponent(ref)}?limit=10${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok || !data.best) return;
            const primary = data.best;
            openSource(primary.page_id, true);
            openHandoff(primary.page_id, primary.snippet || "");
          })
          .catch(() => {});
      }

      function copyToClipboard(text) {
      if (!text) return;
      if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(() => {});
        return;
      }
      const tmp = document.createElement("textarea");
      tmp.value = text;
      document.body.appendChild(tmp);
      tmp.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(tmp);
    }

    function openHandoff(pageId, anchor) {
      openSource(pageId, true, () => {
        if (anchor && currentSourceMeta) {
          currentSourceMeta.anchor = anchor;
        }
        const anchorParam = anchor ? `?anchor=${encodeURIComponent(anchor)}` : "";
        fetch(`/corpus/handoff/${pageId}${anchorParam}`)
          .then(r => r.json())
          .then(data => {
            if (!data.ok) return;

            const block = document.createElement("div");
            block.className = "section";

            const summary = data.handoff_summary || {};
            const keywords = (data.keywords || []).map(k => `<span class="badge">${k}</span>`).join(" ");
            const structural = data.structural_intelligence || summary.structural_intelligence || null;
            const structuralHtml = renderStructuralHeaderV2(structural);

            const sourceRoleCode = summary.source_role_code || "scholarly_commentary";
            const translationAllowed = summary.translation_allowed === true || sourceRoleCode === "primary_text";
            const context = summary.context_capsule || {};
            const contextRows = [
              ["Document type", context.document_type],
              ["Parties involved", context.parties],
              ["Commodity", context.commodity],
              ["Source chain", context.source_chain],
              ["Confidence", context.confidence],
            ].filter(([, value]) => value);
            const contextHtml = translationAllowed
              ? (contextRows.length
                  ? `<ul class="context-list">${contextRows
                      .map(([label, value]) => `<li><strong>${label}:</strong> ${escapeHtml(value)}</li>`)
                      .join("")}</ul>`
                  : "<div class='meta'>Context unavailable.</div>")
              : "<div class='meta'>Context extraction is available only for primary texts. Jump to linked primary text to translate.</div>";
            const gateCopyHtml = translationAllowed
              ? ""
              : `<div class="meta gate-copy">${escapeHtml(summary.translation_gate_copy || "")}</div>`;
            const modes = translationAllowed ? (summary.translation_modes || []) : [];
            const modesHtml = modes.length
              ? modes
                  .map((m) => {
                    const label = m.mode || m.label || "";
                    const status = m.status || "";
                    const cls = status === "available" ? "mode-on" : "mode-off";
                    const suffix = status && status !== "available" ? ` (${status})` : "";
                    return `<span class="badge ${cls}">${escapeHtml(label + suffix)}</span>`;
                  })
                  .join(" ")
              : "<span class='meta'>not configured</span>";
            const translationBlock = translationAllowed
              ? `<div class="meta" style="margin-top:8px;"><strong>Translation scope:</strong> ${escapeHtml(summary.translation_scope || "")}</div>
                 <div class="meta" style="margin-top:6px;"><strong>Translation mode:</strong> ${modesHtml}</div>`
              : "";
            const canJump = !translationAllowed && (data.citations_preview || []).length;
            const jumpHtml = !translationAllowed
              ? (canJump
                  ? `<div class="meta" style="margin-top:6px;"><strong>What you can do:</strong> Jump to linked primary text.</div>
                     <div class="meta" style="margin-top:6px;"><button id="btn-jump-primary-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Jump now</button></div>`
                  : "<div class='meta' style='margin-top:6px;'>No linked primary text available.</div>")
              : "";

            const snippets = data.snippets_structured || [];
            const bestRef = pickBestRefFromCitations(data.citations_preview || data.citations || []);
            const showSnippetList = !translationAllowed && snippets.length;
            const snippetText = showSnippetList
              ? snippets.map((s) => `${s.marker || "Snippet"}: ${s.snippet || ""}`).join("\n")
              : "";
            const snippetsHtml = showSnippetList
              ? `<div class="meta" style="margin-top:8px;"><strong>Evidence snippets</strong></div>` +
                snippets.map((s) => {
                  const snippetText = s.snippet || "";
                  const label = s.marker ? `Snippet about ${s.marker}` : "Snippet";
                  return `
                    <div class="snippet-card">
                      <div class="meta"><strong>${escapeHtml(label)}</strong></div>
                      <div class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(snippetText)}</div>
                      <div class="meta" style="margin-top:6px;">
                        <button class="btn-snippet-jump" data-ref="${bestRef ? encodeURIComponent(bestRef) : ""}" data-anchor="${encodeURIComponent(snippetText)}" style="background:#1f2c4d;color:#b6c8ff;" ${bestRef ? "" : "disabled"}>Open primary</button>
                      </div>
                    </div>
                  `;
                }).join("")
              : "";
            const excerptFocus = data.excerpt_focus || data.excerpt_short || "";
            const excerptFull = data.excerpt_full || "";
            const focusLabel = data.anchor_matched ? "Focused excerpt (matched)" : "Focused excerpt (default)";
            let expanded = false;
            const excerptId = `handoff-excerpt-${pageId}`;
            const excerptHtml = showSnippetList
              ? `${snippetsHtml}<div id="${excerptId}" class="snippet" style="display:none;">${escapeHtml(snippetText)}</div>`
              : `<div class="meta" style="margin-top:8px;"><strong>${focusLabel}</strong></div>
                 <div id="${excerptId}" class="snippet" style="max-height:none; display:block; white-space:pre-wrap;">${escapeHtml(excerptFocus)}</div>`;
            const copyFocusText = showSnippetList ? snippetText : excerptFocus;
            const copyFullText = showSnippetList ? snippetText : excerptFull;
            const toggleStyle = showSnippetList ? "display:none;" : "";

            block.innerHTML = `
              <h4>Translation handoff</h4>
              ${structuralHtml}
              <div class="meta"><strong>Source:</strong> ${data.pdf_name} (p${data.page_number}) ? ${docTypeLabel(data.doc_type)}</div>
              <div class="meta"><strong>Handoff summary:</strong> ${escapeHtml(summary.source_role || "-")} ? ${escapeHtml(summary.doc_type_label || "-")}</div>
              <div class="meta">${escapeHtml(summary.why_opened || "")}</div>
              ${gateCopyHtml}
              <div class="meta"><strong>Signals:</strong> ${keywords || "<span class='meta'>none</span>"}</div>

              <div class="meta" style="margin-top:8px;"><strong>Context</strong></div>
              ${contextHtml}
              ${jumpHtml}
              ${translationBlock}

              <div class="meta" style="margin-top:8px;"><strong>Paper trail:</strong> ${(data.citations_preview || []).join("; ") || "No citations listed"}</div>

              <div class="meta" style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <button id="btn-copy-excerpt-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy excerpt</button>
                <button id="btn-copy-cits-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy citations</button>
                <button id="btn-copy-all-${pageId}" style="background:#1f2c4d;color:#b6c8ff;">Copy all</button>
                <button id="btn-toggle-${pageId}" style="background:#1f2c4d;color:#b6c8ff;${toggleStyle}">Show more</button>
              </div>

              ${excerptHtml}

              <div class="meta" style="margin-top:10px;"><em>${escapeHtml(summary.note || "")}</em></div>
            `;

            sourceContent.appendChild(block);

            const jumpBtn = document.getElementById(`btn-jump-primary-${pageId}`);
            if (jumpBtn) {
              jumpBtn.onclick = () => {
                const ref = pickBestRefFromCitations(data.citations_preview || data.citations || []);
                fetchPrimaryExcerptFromRef(ref, anchor || "");
              };
            }

            const snippetButtons = block.querySelectorAll(".btn-snippet-jump");
            snippetButtons.forEach((btn) => {
              const refEncoded = btn.dataset.ref || "";
              const anchorEncoded = btn.dataset.anchor || "";
              if (!refEncoded) return;
              const ref = decodeURIComponent(refEncoded);
              const anchorText = decodeURIComponent(anchorEncoded || "");
              btn.onclick = () => fetchPrimaryExcerptFromRef(ref, anchorText);
            });

            document.getElementById(`btn-copy-excerpt-${pageId}`).onclick = () => {
              copyToClipboard(expanded ? copyFullText : copyFocusText);
            };
            document.getElementById(`btn-copy-cits-${pageId}`).onclick = () => {
              copyToClipboard((data.citations || []).join("; "));
            };
            document.getElementById(`btn-copy-all-${pageId}`).onclick = () => {
              copyToClipboard(data.copy_payload || "");
            };
            document.getElementById(`btn-toggle-${pageId}`).onclick = () => {
              expanded = !expanded;
              const excerptEl = document.getElementById(excerptId);
              if (!excerptEl) return;
              excerptEl.textContent = expanded ? copyFullText : copyFocusText;
              document.getElementById(`btn-toggle-${pageId}`).textContent = expanded ? "Show less" : "Show more";
            };
          });
      });
    }

    function loadCitation(ref, idx = -1) {
      if (idx >= 0) currentCitationIndex = idx;
      drawerDetail.innerHTML = "<div class='meta'>Loading...</div>";
      fetch(`/corpus/citation?ref=${ref}&limit=5`)
        .then(r => r.json())
        .then(data => {
          if (!data.ok) {
            drawerDetail.innerHTML = `<div class='meta'>${data.error || "Could not load citation."}</div>`;
            return;
          }
          if (!data.results || !data.results.length) {
            drawerDetail.innerHTML = "<div class='meta'>No matches found for this citation.</div>";
            return;
          }
          if (data.evidence_summary) {
            renderEvidenceSummary(data.evidence_summary);
          }
          currentSourceResults = data.results;
          currentSourceIndex = 0;
          const r = data.results[0];
          const srcType = sourceTypeLabel(r.source_role || "scholarly_commentary");
          const structuralHtml = renderStructuralHeaderV2(r.structural_intelligence || null);
          const openLink = r.page_id ? `<a href="#" onclick="openSource('${r.page_id}');return false;">Open source</a>` : `<span style="color:var(--muted)">Source link unavailable</span>`;
          drawerDetail.innerHTML = `
            <div class="meta"><span class="badge">${docTypeLabel(r.doc_type)}</span> <span class="badge">${r.ref_type ? r.ref_type.toUpperCase() : "SOURCE"}</span> <span class="badge">${r.doc_hint || ""}</span> <span class="badge ${srcType.cls}">${srcType.text}</span> <button class="help" type="button" data-tip-title="Citation detail badges" data-tip="These badges summarize document type, citation class, hint label, and source role (primary/wrapper/commentary) for the currently selected citation result.">?</button></div>
            ${structuralHtml}
            <div class="snippet">${markTerm(r.excerpt && r.excerpt.text ? r.excerpt.text : (r.snippet || ""))}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">${r.rank && r.rank.rank_reason ? "Why it matters: " + r.rank.rank_reason : ""}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">${openLink}</div>
            <div class="meta" style="font-size:11px;color:#7ea6d6;">
              <button onclick="stepCitation(-1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:4px 6px;border-radius:4px;cursor:pointer;">Prev</button>
              <button onclick="stepCitation(1)" style="background:#1f2c4d;color:#b6c8ff;border:none;padding:4px 6px;border-radius:4px;cursor:pointer;">Next</button>
            </div>
          `;
        })
        .catch(() => {
          drawerDetail.innerHTML = "<div class='meta'>Citation lookup failed. Try another or open the source page.</div>";
        });
    }

    function stepCitation(delta) {
      if (!currentCitationList.length) return;
      let idx = currentCitationIndex >= 0 ? currentCitationIndex + delta : delta;
      if (idx < 0) idx = currentCitationList.length - 1;
      if (idx >= currentCitationList.length) idx = 0;
      currentCitationIndex = idx;
      const ref = currentCitationList[idx].ref;
      loadCitation(encodeURIComponent(ref), idx);
    }

    function stepSource(delta) {
      if (!currentSourceResults.length) return;
      let idx = currentSourceIndex >= 0 ? currentSourceIndex + delta : delta;
      if (idx < 0) idx = currentSourceResults.length - 1;
      if (idx >= currentSourceResults.length) idx = 0;
      currentSourceIndex = idx;
      const r = currentSourceResults[idx];
      openSource(r.page_id, true);
    }

    const loadDemoAndStatsV2 = () => {
      const demoUrl = `/corpus/demo?limit=${encodeURIComponent(String(demoResultLimit))}`;
      Promise.all([fetch(demoUrl), fetch("/corpus/stats")])
        .then(([d, s]) => Promise.all([d.json(), s.json()]))
        .then(([data, stats]) => {
          if (!data.ok) throw new Error("Demo not ok");
          renderStats(stats);
          renderDemo(data);
        })
        .catch(err => {
          panelsEl.innerHTML = '<div class="panel"><h2>Error</h2><div class="snippet">' + err + '</div></div>';
        });
    };

    if (demoLimitSelectEl) {
      demoLimitSelectEl.value = String(demoResultLimit);
      demoLimitSelectEl.addEventListener("change", () => {
        demoResultLimit = clampDemoLimitV2(demoLimitSelectEl.value || "25");
        demoLimitSelectEl.value = String(demoResultLimit);
        saveDemoLimitV2();
        activeSelectionSource = "auto";
        activeEvidencePageId = null;
        pendingActiveScroll = false;
        loadDemoAndStatsV2();
      });
    }

    const openBrowseSearchV2 = () => {
      const q = (globalSourceSearchEl?.value || "").trim();
      const url = q
        ? `/corpus/browse?q=${encodeURIComponent(q)}&limit=50&page=1`
        : `/corpus/browse?limit=50&page=1`;
      window.open(url, "_blank", "noopener");
    };
    if (openBrowseSearchBtnEl) {
      openBrowseSearchBtnEl.addEventListener("click", openBrowseSearchV2);
    }
    if (globalSourceSearchEl) {
      globalSourceSearchEl.addEventListener("keydown", (evt) => {
        if (evt.key !== "Enter") return;
        evt.preventDefault();
        openBrowseSearchV2();
      });
    }

    loadDemoAndStatsV2();

    // Tooltip logic
    let tipEl = null;
    let tipAnchor = null;
    function closeTip() {
      if (tipEl) tipEl.remove();
      tipEl = null;
      tipAnchor = null;
    }
    function showTip(btn) {
      if (tipAnchor === btn && tipEl) return;
      closeTip();
      tipAnchor = btn;
      const title = btn.dataset.tipTitle || "Info";
      const body = btn.dataset.tip || "";
      tipEl = document.createElement("div");
      tipEl.className = "tooltip";
      tipEl.innerHTML = `
        <div class="tooltip-title">${title}</div>
        <div class="tooltip-body">${body}</div>
      `;
      document.body.appendChild(tipEl);
      const r = btn.getBoundingClientRect();
      const pad = 10;
      let x = r.right + pad;
      let y = r.top;
      const w = tipEl.offsetWidth;
      const h = tipEl.offsetHeight;
      if (x + w > window.innerWidth - 12) x = r.left - w - pad;
      if (y + h > window.innerHeight - 12) y = window.innerHeight - h - 12;
      if (y < 12) y = 12;
      tipEl.style.left = `${x}px`;
      tipEl.style.top = `${y}px`;
    }
    const findTipTarget = (el) => {
      if (!el) return null;
      let node = el.nodeType === 3 ? el.parentNode : el; // text node guard
      return node && node.closest ? node.closest("[data-tip]") : null;
    };
    const isEditableTargetV2 = (target) => {
      if (!target) return false;
      const el = target.nodeType === 1 ? target : target.parentElement;
      if (!el) return false;
      if (el.isContentEditable) return true;
      const tag = String(el.tagName || "").toLowerCase();
      return tag === "input" || tag === "textarea" || tag === "select";
    };
    const focusNavSearchV2 = () => {
      const input = document.getElementById("nav-search-input");
      if (!input) return false;
      input.focus();
      input.select();
      return true;
    };
    const openActiveSourceV2 = () => {
      const pageId = String(activeEvidencePageId || "");
      if (!pageId) return;
      openSource(pageId, true);
    };
    const toggleCitationExplorerV2 = () => {
      const pageId = String(activeEvidencePageId || "");
      if (drawer.classList.contains("open")) {
        closeDrawer();
        return;
      }
      if (!pageId) return;
      openDrawer(pageId, "");
    };

    // Click to pin tooltip
    document.addEventListener("click", (e) => {
      const btn = findTipTarget(e.target);
      if (btn) {
        e.preventDefault();
        showTip(btn);
        return;
      }
      if (tipEl && !e.target.closest(".tooltip")) closeTip();
    });
    // Hover support for stats/cards
    document.addEventListener("mouseenter", (e) => {
      const btn = findTipTarget(e.target);
      if (btn) showTip(btn);
    }, true);
    document.addEventListener("mouseleave", (e) => {
      const btn = findTipTarget(e.target);
      if (!btn) return;
      const related = e.relatedTarget;
      const relTip = findTipTarget(related);
      if (related && (relTip === btn || (related.closest && related.closest(".tooltip")))) return;
      closeTip();
    }, true);
    document.addEventListener("mouseleave", (e) => {
      if (e.target.classList && e.target.classList.contains("tooltip")) {
        closeTip();
      }
    }, true);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeTip();
        return;
      }
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const editable = isEditableTargetV2(e.target);
      if (e.key === "/" && !editable) {
        e.preventDefault();
        focusNavSearchV2();
        return;
      }
      if (editable) return;
      const k = String(e.key || "").toLowerCase();
      if (k === "j") {
        e.preventDefault();
        moveActiveEvidenceV2(1);
        return;
      }
      if (k === "k") {
        e.preventDefault();
        moveActiveEvidenceV2(-1);
        return;
      }
      if (k === "e") {
        e.preventDefault();
        toggleCitationExplorerV2();
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        openActiveSourceV2();
      }
    });
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@corpus_bp.get("/browse")
@json_errors
def browse_ui():
    raw_q = (request.args.get("q", "") or "").strip()
    q_norm = raw_q.lower()
    try:
        page = int(request.args.get("page", "1"))
    except ValueError:
        page = 1
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50
    page = max(1, page)
    limit = max(10, min(limit, 200))
    offset = (page - 1) * limit

    where_sql = ""
    params: list = []
    if q_norm:
        where_sql = "WHERE lower(p.pdf_name) LIKE ? OR lower(p.page_id) LIKE ?"
        like = f"%{q_norm}%"
        params.extend([like, like])

    total_sql = f"""
        SELECT COUNT(*) AS n
        FROM page_registry p
        {where_sql}
    """
    total = scalar(total_sql, tuple(params))

    rows_sql = f"""
        SELECT p.page_id, p.pdf_name, p.page_number, p.text_norm, s.json AS s_json
        FROM page_registry p
        LEFT JOIN social s ON s.page_id = p.page_id
        {where_sql}
        ORDER BY lower(p.pdf_name), CAST(p.page_number AS INTEGER), p.page_id
        LIMIT ? OFFSET ?
    """

    total_pages = max(1, (total + limit - 1) // limit)
    page = min(page, total_pages)
    try:
        goto_item = int(request.args.get("goto_item", "0"))
    except ValueError:
        goto_item = 0
    try:
        goto_page = int(request.args.get("goto_page", "0"))
    except ValueError:
        goto_page = 0
    if goto_item > 0:
        page = min(total_pages, max(1, ((goto_item - 1) // limit) + 1))
    elif goto_page > 0:
        page = min(total_pages, max(1, goto_page))
    offset = (page - 1) * limit

    rows = q(rows_sql, tuple(params + [limit, offset]))
    items = []
    for r in rows:
        soc = safe_json_loads(r["s_json"])
        doc_type = override_doc_type(r["pdf_name"], soc.get("doc_type", "unknown"))
        role = compute_source_role(r["pdf_name"], doc_type, r["text_norm"])
        role_label = get_source_role_label(role)
        items.append(
            {
                "page_id": r["page_id"],
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "doc_type": DOC_LABEL.get(doc_type, "Unclassified page"),
                "source_role": role_label,
            }
        )
    start_row = offset + 1 if total else 0
    end_row = min(offset + len(items), total) if total else 0

    def _browse_url(next_page: int) -> str:
        qp = f"page={next_page}&limit={limit}"
        if raw_q:
            qp += f"&q={quote_plus(raw_q)}"
        return f"/corpus/browse?{qp}"

    prev_url = _browse_url(max(1, page - 1))
    next_url = _browse_url(min(total_pages, page + 1))

    rows_html = "".join(
        [
            (
                "<tr>"
                f"<td>{html.escape(str(i + offset + 1))}</td>"
                f"<td>{html.escape(str(row['pdf_name'] or ''))}</td>"
                f"<td>{html.escape(str(row['page_number'] or '-'))}</td>"
                f"<td>{html.escape(str(row['doc_type'] or 'Unknown'))}</td>"
                f"<td>{html.escape(str(row['source_role'] or 'Unknown'))}</td>"
                f"<td><a href=\"{source_record_url(row['page_id'])}\" target=\"_blank\" rel=\"noopener\">Open record</a></td>"
                "</tr>"
            )
            for i, row in enumerate(items)
        ]
    )
    if not rows_html:
        rows_html = "<tr><td colspan=\"6\">No records found for this filter.</td></tr>"

    page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MIM Corpus Browser</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #0b1021; color: #e7ecf7; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px 20px 22px; }}
    h1 {{ margin: 0 0 8px; color: #7ec7ff; font-size: 24px; }}
    .meta {{ color: #9cb4e6; font-size: 13px; margin: 4px 0; }}
    .toolbar {{ margin-top: 12px; padding: 10px; border: 1px solid #27406d; border-radius: 8px; background: #111a32; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    input, select {{ background: #0f1832; color: #d7e6ff; border: 1px solid #2d4472; border-radius: 6px; padding: 6px 8px; }}
    button, .btn {{ background: #1f2c4d; color: #b6c8ff; border: 1px solid #2d4472; border-radius: 6px; padding: 6px 10px; text-decoration: none; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; border: 1px solid #27406d; border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #21365f; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: #9fd6ff; background: #12213f; }}
    tr:hover td {{ background: #101d39; }}
    a {{ color: #7ec7ff; }}
    .pager {{ margin-top: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MIM Corpus Browser</h1>
    <div class="meta">Browse the full indexed corpus (all tablets/artifacts/pages), not just top demo cards.</div>
    <div class="meta">Total records: {total} | Showing rows {start_row}-{end_row} | Page {page} / {total_pages} | Limit {limit}</div>
    <form class="toolbar" method="get" action="/corpus/browse">
      <label for="q">Search PDF/page id</label>
      <input id="q" name="q" value="{html.escape(raw_q)}" placeholder="e.g., AKT 11a or page id" />
      <label for="limit">Rows</label>
      <select id="limit" name="limit">
        <option value="25" {"selected" if limit == 25 else ""}>25</option>
        <option value="50" {"selected" if limit == 50 else ""}>50</option>
        <option value="100" {"selected" if limit == 100 else ""}>100</option>
        <option value="200" {"selected" if limit == 200 else ""}>200</option>
      </select>
      <input type="hidden" name="page" value="1" />
      <button type="submit">Apply</button>
      <a class="btn" href="/corpus/mim">Back to MIM Demo</a>
    </form>
    <form class="toolbar" method="get" action="/corpus/browse">
      <input type="hidden" name="q" value="{html.escape(raw_q)}" />
      <input type="hidden" name="limit" value="{limit}" />
      <label for="goto_page">Go to page</label>
      <input id="goto_page" name="goto_page" type="number" min="1" max="{total_pages}" placeholder="e.g., 181" />
      <button type="submit">Go</button>
      <label for="goto_item">Go to item #</label>
      <input id="goto_item" name="goto_item" type="number" min="1" max="{max(total,1)}" placeholder="e.g., 93000" />
      <button type="submit">Jump</button>
    </form>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>PDF</th>
          <th>Page</th>
          <th>Doc Type</th>
          <th>Source Role</th>
          <th>Open</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    <div class="pager">
      <a class="btn" href="{html.escape(prev_url)}">Prev</a>
      <a class="btn" href="{html.escape(next_url)}">Next</a>
      <span class="meta">Tip: from each row, open record -> citations -> linked primary.</span>
    </div>
  </div>
</body>
</html>"""
    return Response(page_html, mimetype="text/html")


@corpus_bp.get("/search/deity")
@json_errors
def search_deity_route():
    name = request.args.get("name", "")
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    require_citations = request.args.get("require_citations", "false").lower() == "true"
    limit = int(request.args.get("limit", "20"))
    results = search_deity(name, require_citations=require_citations, limit=limit)
    # rank: doc_type priority then citation count desc
    doc_priority = {"letter": 3, "legal": 3, "commentary": 2, "index": 1, "bibliography": 1, "front_matter": 0}
    results.sort(
        key=lambda r: (
            -(doc_priority.get(r.get("doc_type"), 0)),
            -len(r.get("citations", [])),
        )
    )
    summary = build_evidence_summary(results, top_n=6)
    return jsonify(
        {
            "ok": True,
            "query": {"name": name, "require_citations": require_citations, "limit": limit},
            "count_returned": len(results),
            "evidence_summary": summary,
            "results": results,
        }
    )


@corpus_bp.get("/search/formula")
@json_errors
def search_formula_route():
    marker = request.args.get("marker", "")
    if not marker:
        return jsonify({"ok": False, "error": "marker required"}), 400
    limit = int(request.args.get("limit", "20"))
    results = search_formula(marker, limit=limit)
    doc_priority = {"letter": 3, "legal": 3, "commentary": 2, "index": 1, "bibliography": 1, "front_matter": 0}
    results.sort(
        key=lambda r: (
            -(doc_priority.get(r.get("doc_type"), 0)),
            -len(r.get("formula_snippets", [])),
            -len(r.get("citations", [])),
        )
    )
    summary = build_evidence_summary(results, top_n=6)
    return jsonify(
        {
            "ok": True,
            "query": {"marker": marker, "limit": limit},
            "count_returned": len(results),
            "evidence_summary": summary,
            "results": results,
        }
    )


@corpus_bp.get("/search/institution")
@json_errors
def search_institution_route():
    inst = request.args.get("inst", "")
    if not inst:
        return jsonify({"ok": False, "error": "inst required"}), 400
    limit = int(request.args.get("limit", "20"))
    results = search_institution(inst, limit=limit)
    results.sort(
        key=lambda r: (
            -(1 if r.get("doc_type") in ("letter", "legal") else 0),
            -len(r.get("citations", [])),
        )
    )
    summary = build_evidence_summary(results, top_n=6)
    return jsonify(
        {
            "ok": True,
            "query": {"inst": inst, "limit": limit},
            "count_returned": len(results),
            "evidence_summary": summary,
            "results": results,
        }
    )


@corpus_bp.get("/page/<page_id>/record")
def page_record_route(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return Response("Source record not found.", status=404, mimetype="text/plain")
    soc = bundle.get("social", {}) or {}
    phil = bundle.get("philology", {}) or {}
    doc_type_raw = soc.get("doc_type", "unknown")
    doc_type = override_doc_type(bundle.get("pdf_name"), doc_type_raw)
    source_role = compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm"))
    source_role_label = get_source_role_label(source_role)

    title = html.escape(pretty_title(bundle.get("pdf_name")))
    pdf_name = html.escape(str(bundle.get("pdf_name") or ""))
    page_number = html.escape(str(bundle.get("page_number") or "-"))
    doc_type_label = html.escape(DOC_LABEL.get(doc_type, "Unclassified page"))
    source_role_text = html.escape(source_role_label)
    topics = ", ".join(soc.get("topics", []) or []) or "none"
    institutions = ", ".join(soc.get("institutions", []) or []) or "none"
    citations = phil.get("citations", []) or []
    citations_html = "".join([f"<li>{html.escape(str(c))}</li>" for c in citations[:80]]) or "<li>none</li>"
    text_blob = bundle.get("text_norm") or bundle.get("text_raw") or ""
    text_html = html.escape(text_blob)
    page_json_url = f"/corpus/page/{page_id}?include_text=true"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title} - Source Record</title>
  <style>
    body {{ margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #0b1021; color: #e7ecf7; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 18px 20px 30px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; color: #7ec7ff; }}
    .meta {{ font-size: 13px; color: #9cb4e6; margin: 4px 0; }}
    .panel {{ border: 1px solid #23365c; border-radius: 8px; background: #111a32; padding: 12px; margin-top: 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; color: #7ec7ff; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: Consolas, "Courier New", monospace; font-size: 13px; line-height: 1.45; margin: 0; }}
    a {{ color: #7ec7ff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="meta">PDF: {pdf_name} | page {page_number}</div>
    <div class="meta">Type: {doc_type_label} | source role: {source_role_text}</div>
    <div class="meta">Topics: {html.escape(topics)} | Institutions: {html.escape(institutions)}</div>
    <div class="meta"><a href="{page_json_url}" target="_blank" rel="noopener">Open raw JSON bundle</a></div>
    <div class="panel">
      <h2>Citations</h2>
      <ul>{citations_html}</ul>
    </div>
    <div class="panel">
      <h2>Source Text</h2>
      <pre>{text_html}</pre>
    </div>
  </div>
</body>
</html>"""
    return Response(html_doc, mimetype="text/html")


@corpus_bp.get("/page/<page_id>/compare")
def page_visual_compare_route(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return Response("Source not found.", status=404, mimetype="text/plain")

    pdf_name = bundle.get("pdf_name") or ""
    page_number = bundle.get("page_number") or ""
    text_full = (bundle.get("text_norm") or bundle.get("text_raw") or "").strip()
    excerpt = re.sub(r"\s+", " ", text_full)
    if len(excerpt) > 2600:
        excerpt = excerpt[:2600].rstrip() + "..."

    candidates = find_visual_candidates(pdf_name, page_number, max_results=25)
    try:
        idx = int(request.args.get("idx", "1"))
    except ValueError:
        idx = 1
    if candidates:
        idx = max(1, min(idx, len(candidates)))
        selected = candidates[idx - 1]
    else:
        idx = 1
        selected = {}

    selected_label = selected.get("label") or "No matched visual candidate"
    selected_pub = selected.get("publication_catalog") or selected.get("aliases") or ""
    selected_cdli = selected.get("cdli_id") or ""
    selected_score = selected.get("match_score", 0)
    selected_oare = selected.get("online_transcript") or ""
    selected_aicc = selected.get("aicc_translation") or ""
    selected_url = selected_oare or selected_aicc

    image_pattern = re.compile(r"\.(png|jpe?g|gif|webp)(?:\?|$)", re.IGNORECASE)
    selected_is_image = bool(selected_url and image_pattern.search(selected_url))

    source_assets = build_source_asset_fields(page_id, pdf_name, page_number)
    source_scan_url = (
        source_assets.get("asset_full_url")
        or source_assets.get("pdf_render_url")
        or source_assets.get("asset_thumb_url")
        or ""
    )

    source_preview_html = ""
    if source_scan_url:
        source_preview_html = (
            f'<img src="{html.escape(str(source_scan_url))}" alt="Source preview" '
            'style="max-width:100%;max-height:460px;border-radius:8px;border:1px solid #2b4372;background:#0d1734;" />'
        )
    else:
        source_preview_html = (
            "<div class=\"placeholder\">No local scan image is indexed for this source. "
            "Use source record and citation links below.</div>"
        )

    visual_preview_html = ""
    if selected_url and selected_is_image:
        visual_preview_html = (
            f'<img src="{html.escape(selected_url)}" alt="Visual candidate preview" '
            'style="max-width:100%;max-height:460px;border-radius:8px;border:1px solid #2b4372;background:#0d1734;" />'
        )
    elif selected_url:
        visual_preview_html = (
            f'<iframe src="{html.escape(selected_url)}" loading="lazy" '
            'style="width:100%;height:460px;border:1px solid #2b4372;border-radius:8px;background:#0d1734;"></iframe>'
            '<div class="meta">If the external site blocks embed, use the Open link below.</div>'
        )
    else:
        visual_preview_html = (
            "<div class=\"placeholder\">No direct visual candidate URL available for this source.</div>"
        )

    candidate_rows = ""
    for i, c in enumerate(candidates, start=1):
        cand_label = c.get("label") or "Untitled"
        cand_pub = c.get("publication_catalog") or c.get("aliases") or ""
        cand_cdli = c.get("cdli_id") or ""
        cand_score = c.get("match_score", 0)
        cand_oare = c.get("online_transcript") or ""
        cand_aicc = c.get("aicc_translation") or ""
        cand_open = cand_oare or cand_aicc
        row_cls = "selected" if i == idx else ""
        select_link = f"/corpus/page/{page_id}/compare?idx={i}"
        open_link = (
            f'<a href="{html.escape(cand_open)}" target="_blank" rel="noopener">Open candidate</a>'
            if cand_open
            else '<span class="muted">n/a</span>'
        )
        candidate_rows += (
            f'<tr class="{row_cls}">'
            f"<td>{i}</td>"
            f"<td><a href=\"{html.escape(select_link)}\">Select</a></td>"
            f"<td>{html.escape(cand_label)}</td>"
            f"<td>{html.escape(cand_pub)}</td>"
            f"<td>{html.escape(cand_cdli)}</td>"
            f"<td>{cand_score}</td>"
            f"<td>{open_link}</td>"
            "</tr>"
        )
    if not candidate_rows:
        candidate_rows = (
            '<tr><td colspan="7">No matched candidates found. Use visual search links below.</td></tr>'
        )

    stem = re.sub(r"\.pdf$", "", str(pdf_name), flags=re.IGNORECASE).strip()
    generic_aic = f"https://aicuneiform.com/search?q={quote_plus(stem)}" if stem else ""
    generic_google = (
        f"https://www.google.com/search?q={quote_plus(stem + ' cuneiform tablet image')}" if stem else ""
    )
    source_record = source_record_url(page_id)
    visuals_url = f"/corpus/page/{page_id}/visuals"
    source_bundle_url = f"/corpus/page/{page_id}?include_text=true"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Compare Source vs Visual - {html.escape(str(pdf_name))}</title>
  <style>
    body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif; background:#0b1021; color:#e7ecf7; }}
    .wrap {{ max-width: 1400px; margin:0 auto; padding:16px 20px 24px; }}
    h1 {{ margin:0 0 8px; color:#7ec7ff; font-size:22px; }}
    .meta {{ color:#9cb4e6; font-size:13px; margin:4px 0; }}
    .muted {{ color:#9cb4e6; }}
    .actions a {{
      display:inline-block; margin-right:8px; margin-top:6px; padding:6px 10px;
      border:1px solid #2d4472; border-radius:6px; background:#1f2c4d; color:#d7e6ff; text-decoration:none;
    }}
    .actions a:hover {{ background:#27406d; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap:12px; margin-top:12px; }}
    .panel {{ border:1px solid #27406d; border-radius:8px; background:#111a32; padding:10px; }}
    .panel h2 {{ margin:0 0 8px; color:#9fd6ff; font-size:16px; }}
    .placeholder {{ border:1px dashed #2b4372; border-radius:8px; padding:14px; color:#9cb4e6; background:#0d1734; }}
    .excerpt {{
      margin-top:8px; padding:10px; border:1px solid #2b4372; border-radius:8px;
      background:#0d1734; color:#e7ecf7; font-size:13px; line-height:1.45; max-height:260px; overflow:auto;
      white-space:pre-wrap;
    }}
    table {{ width:100%; border-collapse: collapse; margin-top:10px; }}
    th,td {{ border-bottom:1px solid #21365f; padding:8px; text-align:left; font-size:12px; vertical-align:top; }}
    th {{ color:#9fd6ff; background:#12213f; }}
    tr.selected td {{ background:#102747; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Source vs Visual Compare</h1>
    <div class="meta">Source: {html.escape(str(pdf_name))} (p{html.escape(str(page_number))})</div>
    <div class="meta">Selected candidate: {html.escape(str(selected_label))}</div>
    <div class="actions">
      <a href="{html.escape(source_record)}" target="_blank" rel="noopener">Open source record</a>
      <a href="{html.escape(visuals_url)}" target="_blank" rel="noopener">Open visual candidates list</a>
      <a href="{html.escape(source_bundle_url)}" target="_blank" rel="noopener">Open source JSON bundle</a>
      {f'<a href="{html.escape(generic_aic)}" target="_blank" rel="noopener">Fallback: AICC search</a>' if generic_aic else ''}
      {f'<a href="{html.escape(generic_google)}" target="_blank" rel="noopener">Fallback: web image search</a>' if generic_google else ''}
    </div>
    <div class="grid">
      <section class="panel">
        <h2>Source Context</h2>
        {source_preview_html}
        <div class="meta"><strong>Document:</strong> {html.escape(str(pdf_name))} (p{html.escape(str(page_number))})</div>
        <div class="excerpt">{html.escape(excerpt or "No source excerpt available.")}</div>
      </section>
      <section class="panel">
        <h2>Visual Candidate</h2>
        {visual_preview_html}
        <div class="meta"><strong>Label:</strong> {html.escape(str(selected_label))}</div>
        <div class="meta"><strong>Publication:</strong> {html.escape(str(selected_pub)) or "n/a"}</div>
        <div class="meta"><strong>CDLI:</strong> {html.escape(str(selected_cdli)) or "n/a"} | <strong>Score:</strong> {selected_score}</div>
        <div class="actions">
          {f'<a href="{html.escape(selected_oare)}" target="_blank" rel="noopener">Open OARE</a>' if selected_oare else ''}
          {f'<a href="{html.escape(selected_aicc)}" target="_blank" rel="noopener">Open AICC</a>' if selected_aicc else ''}
          {f'<a href="{html.escape(selected_url)}" target="_blank" rel="noopener">Open selected URL</a>' if selected_url else ''}
        </div>
      </section>
    </div>
    <section class="panel" style="margin-top:12px;">
      <h2>Candidate Switcher</h2>
      <div class="meta">Select another candidate to compare side-by-side.</div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Choose</th>
            <th>Label</th>
            <th>Publication / Alias</th>
            <th>CDLI</th>
            <th>Score</th>
            <th>Open</th>
          </tr>
        </thead>
        <tbody>{candidate_rows}</tbody>
      </table>
    </section>
  </div>
</body>
</html>"""
    return Response(html_doc, mimetype="text/html")


@corpus_bp.get("/page/<page_id>/visuals")
def page_visual_candidates_route(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return Response("Source not found.", status=404, mimetype="text/plain")
    pdf_name = bundle.get("pdf_name") or ""
    page_number = bundle.get("page_number") or ""
    candidates = find_visual_candidates(pdf_name, page_number, max_results=25)
    stem = re.sub(r"\.pdf$", "", str(pdf_name), flags=re.IGNORECASE).strip()
    generic_aic = f"https://aicuneiform.com/search?q={quote_plus(stem)}" if stem else ""
    generic_google = (
        f"https://www.google.com/search?q={quote_plus(stem + ' cuneiform tablet image')}" if stem else ""
    )
    if generic_aic:
        aic_fallback_html = (
            f'<a href="{html.escape(generic_aic)}" target="_blank" rel="noopener">Fallback: AICC search</a>'
        )
    else:
        aic_fallback_html = ""
    if generic_google:
        google_fallback_html = (
            f'<a href="{html.escape(generic_google)}" target="_blank" rel="noopener">Fallback: web image search</a>'
        )
    else:
        google_fallback_html = ""
    rows_html = ""
    for i, c in enumerate(candidates, start=1):
        oare_url = c.get("online_transcript") or ""
        aicc_url = c.get("aicc_translation") or ""
        cdli = c.get("cdli_id") or ""
        label = c.get("label") or "Untitled candidate"
        pub = c.get("publication_catalog") or ""
        aliases = c.get("aliases") or ""
        score = c.get("match_score", 0)
        if oare_url:
            oare_html = f'<a href="{html.escape(oare_url)}" target="_blank" rel="noopener">Open OARE</a>'
        else:
            oare_html = '<span style="color:#9cb4e6">n/a</span>'
        if aicc_url:
            aicc_html = f'<a href="{html.escape(aicc_url)}" target="_blank" rel="noopener">Open AICC</a>'
        else:
            aicc_html = '<span style="color:#9cb4e6">n/a</span>'
        rows_html += (
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(pub or aliases)}</td>"
            f"<td>{html.escape(cdli)}</td>"
            f"<td>{score}</td>"
            f"<td>{oare_html}</td>"
            f"<td>{aicc_html}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html = "<tr><td colspan=\"7\">No direct visual candidates found. Use fallback search links below.</td></tr>"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Visual Candidates - {html.escape(str(pdf_name))}</title>
  <style>
    body {{ margin:0; font-family:'Segoe UI',Arial,sans-serif; background:#0b1021; color:#e7ecf7; }}
    .wrap {{ max-width: 1200px; margin:0 auto; padding:16px 20px 24px; }}
    h1 {{ margin:0 0 8px; color:#7ec7ff; font-size:22px; }}
    .meta {{ color:#9cb4e6; font-size:13px; margin:4px 0; }}
    .panel {{ margin-top: 10px; border:1px solid #27406d; border-radius:8px; background:#111a32; padding:10px; }}
    table {{ width:100%; border-collapse: collapse; }}
    th,td {{ border-bottom:1px solid #21365f; padding:8px; text-align:left; font-size:13px; vertical-align:top; }}
    th {{ color:#9fd6ff; background:#12213f; }}
    a {{ color:#7ec7ff; }}
    .actions a {{ display:inline-block; margin-right:8px; margin-top:6px; padding:6px 10px; border:1px solid #2d4472; border-radius:6px; background:#1f2c4d; text-decoration:none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Visual Candidates</h1>
    <div class="meta">Source: {html.escape(str(pdf_name))} (p{html.escape(str(page_number))})</div>
    <div class="meta">These are likely visual references (OARE/AICC) matched from publication metadata.</div>
    <div class="actions">
      <a href="{html.escape(source_record_url(page_id))}" target="_blank" rel="noopener">Open source record</a>
      <a href="/corpus/page/{html.escape(page_id)}/compare" target="_blank" rel="noopener">Compare source vs visual</a>
      {aic_fallback_html}
      {google_fallback_html}
    </div>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Label</th>
            <th>Publication / Alias</th>
            <th>CDLI</th>
            <th>Score</th>
            <th>OARE</th>
            <th>AICC</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""
    return Response(html_doc, mimetype="text/html")


@corpus_bp.get("/page/<page_id>")
@json_errors
def page_bundle_route(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return jsonify({"ok": False, "error": "not found"}), 404
    include_text = request.args.get("include_text", "false").lower() == "true"
    if not include_text:
        bundle["text_norm"] = None
        bundle["text_raw"] = None

    # cross-linking helpers
    doc_type_raw = bundle.get("social", {}).get("doc_type", "unknown")
    doc_type = override_doc_type(bundle.get("pdf_name"), doc_type_raw)
    bundle["source_role"] = compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm"))
    ref_types = {"index", "bibliography", "front_matter"}
    bundle["is_reference_page"] = doc_type in ref_types
    weight_map = {
        "letter": 1.0,
        "legal": 1.0,
        "commentary": 0.6,
        "index": 0.4,
        "bibliography": 0.4,
        "front_matter": 0.1,
    }
    bundle["evidence_weight"] = weight_map.get(doc_type, 0.0)

    return jsonify({"ok": True, **bundle})

@corpus_bp.get("/page/<page_id>/story")
@json_errors
def page_story_route(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify(build_story(bundle))

@corpus_bp.get("/page/<page_id>/citations")
@json_errors
def page_citations(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return jsonify({"ok": False, "error": "not found"}), 404
    soc = bundle.get("social", {}) or {}
    phil = bundle.get("philology", {}) or {}
    doc_type_raw = soc.get("doc_type", "unknown")
    doc_type = override_doc_type(bundle.get("pdf_name"), doc_type_raw)
    cits = phil.get("citations", []) or []
    rows = []
    for c in cits:
        cleaned = re.sub(r"\b([0-9])g\b", r"\g<1>9", c)
        norm = normalize_ref(cleaned)
        ref_type, doc_hint = classify_ref(norm)
        rows.append(
            {
                "ref": cleaned,
                "ref_norm": norm,
                "ref_type": ref_type,
                "doc_hint": doc_hint,
            }
        )

    return jsonify(
        {
            "ok": True,
            "query": {"page_id": page_id, "limit": len(rows)},
            "source_page": {
                "page_id": page_id,
                "pdf_name": bundle["pdf_name"],
                "page_number": bundle["page_number"],
                "doc_type": doc_type,
                "asset_thumb_url": bundle.get("asset_thumb_url"),
                "asset_full_url": bundle.get("asset_full_url"),
                "pdf_render_url": bundle.get("pdf_render_url"),
                "source_role": compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm")),
                "topics": soc.get("topics", []),
                "institutions": soc.get("institutions", []),
                "evidence_weight": bundle.get("evidence_weight", 0.0),
                "is_reference_page": bundle.get("is_reference_page", False),
                "structural_intelligence": compute_structural_intelligence(
                    bundle.get("text_norm"),
                    source_role=compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm")),
                    internal_score=float(bundle.get("evidence_weight", 0.0)),
                    internal_gap=None,
                ),
            },
            "count_returned": len(rows),
            "citations": rows,
        }
    )


@corpus_bp.get("/resolve/<path:ref>")
@json_errors
def resolve_ref(ref: str):
    limit = int(request.args.get("limit", "10"))
    anchor = normalize_anchor(request.args.get("anchor", ""))
    results = find_citation_matches(ref, limit=limit)
    for r in results:
        text_norm = r.get("text_norm") or ""
        r["_translit_score"] = translit_score(text_norm)
        if anchor:
            _, a_score = best_window_excerpt(text_norm, anchor)
            r["_anchor_score"] = round(a_score, 3)
        else:
            r["_anchor_score"] = 0.0

    results.sort(
        key=lambda r: (
            0 if r.get("source_role") == "primary_text" else 1,
            -float(r.get("_anchor_score", 0.0)),
            -int(r.get("_translit_score", 0)),
            -float(r.get("evidence_weight", 0.0)),
        )
    )
    best = results[0] if results else None
    return jsonify({"ok": True, "ref": ref, "best": best, "all": results})

@corpus_bp.get("/handoff/<page_id>")
@json_errors
def handoff_bundle(page_id: str):
    bundle = get_page_bundle(page_id)
    if not bundle:
        return jsonify({"ok": False, "error": "not found"}), 404
    soc = bundle.get("social", {}) or {}
    phil = bundle.get("philology", {}) or {}
    doc_type_raw = soc.get("doc_type", "unknown")
    doc_type = override_doc_type(bundle.get("pdf_name"), doc_type_raw)

    text = (bundle.get("text_norm") or bundle.get("text_raw") or "").strip()
    anchor = normalize_anchor(request.args.get("anchor", ""))[:200]
    excerpt_full = text
    excerpt_short = text[:900] + ("..." if len(text) > 900 else "")
    excerpt_focus, anchor_score = best_window_excerpt(excerpt_full, anchor, window_chars=1200)
    anchor_matched = anchor_score >= 0.12

    fake_kps = [{"kind": "commentary", "text": text, "label": "raw"}]
    _themes, keywords = extract_themes_and_keywords(fake_kps, text)

    kind_label = DOC_LABEL.get(doc_type, "Unclassified page")
    source_role_code = compute_source_role(bundle.get("pdf_name"), doc_type, bundle.get("text_norm"))
    source_role_label = get_source_role_label(source_role_code)
    translation_allowed = is_translation_allowed(source_role_code)
    translation_gate_copy = (
        "This source discusses documents. It does not contain a document suitable for translation."
    )
    if translation_allowed:
        context_capsule = build_context_capsule(text, doc_type, keywords, phil.get("citations") or [])
        translation_scope = (
            "This translation covers the tablet text only. Envelope text and scholarly commentary are excluded."
        )
        translation_modes = [
            {"mode": "Literal", "status": "planned"},
            {"mode": "Readable", "status": "available"},
            {"mode": "Annotated", "status": "planned"},
        ]
    else:
        context_capsule = None
        translation_scope = None
        translation_modes = []

    structural = compute_structural_intelligence(
        bundle.get("text_norm"),
        source_role=source_role_code,
        internal_score=float(
            {
                "letter": 1.0,
                "legal": 1.0,
                "commentary": 0.6,
                "index": 0.4,
                "bibliography": 0.4,
                "front_matter": 0.1,
            }.get(doc_type, 0.0)
        ),
        internal_gap=None,
    )

    handoff_summary = {
        "source_role": source_role_label,
        "source_role_code": source_role_code,
        "doc_type": doc_type,
        "doc_type_label": kind_label,
        "asset_thumb_url": bundle.get("asset_thumb_url"),
        "asset_full_url": bundle.get("asset_full_url"),
        "pdf_render_url": bundle.get("pdf_render_url"),
        "translation_allowed": translation_allowed,
        "translation_gate_copy": translation_gate_copy,
        "context_capsule": context_capsule,
        "translation_scope": translation_scope,
        "translation_modes": translation_modes,
        "why_opened": "Use the context capsule before translating." if translation_allowed else "Context only.",
        "signals": keywords,
        "paper_trail_count": len(phil.get("citations", []) or []),
        "note": "This panel does not modify sources. It provides a copyable excerpt and traceable citations.",
        "structural_intelligence": structural,
    }

    citations = (phil.get("citations") or [])
    citations_preview = citations[:10]

    copy_payload = (
        f"Source: {bundle.get('pdf_name')} (p{bundle.get('page_number')})\n"
        f"Type: {kind_label} - {source_role_label}\n"
        f"Signals: {', '.join(keywords) if keywords else 'none'}\n"
        f"Citations: {('; '.join(citations_preview)) if citations_preview else 'none'}\n\n"
        f"Excerpt:\n{excerpt_focus}\n"
    )

    return jsonify(
        {
            "ok": True,
            "page_id": page_id,
            "pdf_name": bundle.get("pdf_name"),
            "page_number": bundle.get("page_number"),
            "doc_type": doc_type,
            "asset_thumb_url": bundle.get("asset_thumb_url"),
            "asset_full_url": bundle.get("asset_full_url"),
            "pdf_render_url": bundle.get("pdf_render_url"),
            "topics": soc.get("topics", []),
            "institutions": soc.get("institutions", []),
            "citations": citations,
            "citations_preview": citations_preview,
            "snippets_structured": phil.get("formula_snippets_structured", []) or [],
            "excerpt_short": excerpt_short,
            "excerpt_full": excerpt_full,
            "excerpt_focus": excerpt_focus,
            "anchor_matched": anchor_matched,
            "anchor_score": round(anchor_score, 3),
            "handoff_summary": handoff_summary,
            "structural_intelligence": structural,
            "keywords": keywords,
            "copy_payload": copy_payload,
        }
    )

@corpus_bp.get("/citation")
@json_errors
def citation_search():
    ref = request.args.get("ref", "")
    if not ref:
        return jsonify({"ok": False, "error": "ref required"}), 400
    ref_norm = normalize_ref(ref)
    limit = int(request.args.get("limit", "10"))
    results = find_citation_matches(ref, limit=limit)
    try:
        summary = build_evidence_summary(results, top_n=6)
    except Exception as e:
        summary = {"summary": "", "key_points": [], "evidence": [], "confidence": {"score": 0.0, "level": "Low", "reasons": [str(e)]}, "tags": [], "keywords": []}
    return jsonify(
        {
            "ok": True,
            "query": {"ref": ref, "ref_norm": ref_norm, "limit": limit},
            "count_returned": len(results),
            "evidence_summary": summary,
            "results": results,
        }
    )


# For local testing without the larger app, you can run this file directly.
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    if CORS:
        CORS(app)
    app.register_blueprint(corpus_bp)
    # health endpoint for quick checks
    @app.get("/corpus/health")
    def api_health():
        counts = {}
        for table in ["page_registry", "philology", "social"]:
            counts[table] = q(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        return jsonify({"ok": True, "db": str(DB), "counts": counts})

    app.run(host="127.0.0.1", port=50001, debug=True)




