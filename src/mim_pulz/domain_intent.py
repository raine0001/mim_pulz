from __future__ import annotations

import re

from mim_pulz.preprocess import normalize_text


DOMAIN_LABELS = (
    "letter",
    "legal",
    "economic",
    "administrative",
    "ritual",
    "unknown",
)


_MONEY_MARKERS = (
    "g\u00cdn",
    "k\u00d9.babbar",
    "ma-na",
    "shekel",
    "silver",
    "interest",
    "owes",
)
_LEGAL_MARKERS = (
    "igi",
    "witness",
    "if he has not paid",
    "if there is no profit",
    "dumu",
)
_LETTER_MARKERS = (
    "um-ma",
    "from ",
    "to ",
    "send me",
    "implored me",
)
_ADMIN_MARKERS = (
    "eponymy",
    "month",
    "scribe",
    "seal of",
    "received",
)
_RITUAL_MARKERS = (
    "en.l\u00edl",
    "i\u0161tar",
    "ilu",
    "offering",
    "temple",
)


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    s = text.lower()
    return sum(1 for m in markers if m in s)


def _domain_scores(text: str) -> dict[str, int]:
    s = normalize_text(text or "")
    if not s:
        return {
            "letter": 0,
            "legal": 0,
            "economic": 0,
            "administrative": 0,
            "ritual": 0,
        }

    scores = {
        "letter": _count_markers(s, _LETTER_MARKERS),
        "legal": _count_markers(s, _LEGAL_MARKERS),
        "economic": _count_markers(s, _MONEY_MARKERS),
        "administrative": _count_markers(s, _ADMIN_MARKERS),
        "ritual": _count_markers(s, _RITUAL_MARKERS),
    }

    # Extra nudges from structural signals
    if re.search(r"\b\d+(\.\d+)?\b", s):
        scores["economic"] += 1
    if "igi" in s and "dumu" in s:
        scores["legal"] += 1
    if "um-ma" in s and "a-na" in s:
        scores["letter"] += 1

    return scores


def infer_dialog_domain_with_confidence(text: str) -> tuple[str, float, dict[str, int]]:
    scores = _domain_scores(text)
    label = max(scores, key=scores.get)
    top = int(scores[label])
    if top <= 0:
        return "unknown", 0.0, scores
    sorted_vals = sorted(scores.values(), reverse=True)
    second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 0
    margin = float((top - second) / max(top, 1))
    evidence = min(1.0, top / 3.0)
    confidence = max(0.0, min(1.0, margin * evidence))
    return label, confidence, scores


def infer_dialog_domain(text: str) -> str:
    label, _, _ = infer_dialog_domain_with_confidence(text)
    return label


def infer_dialog_domain_batch(texts: list[str]) -> list[str]:
    return [infer_dialog_domain(t) for t in texts]
