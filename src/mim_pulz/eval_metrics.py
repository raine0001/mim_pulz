from __future__ import annotations

import math
import re

import sacrebleu
from sacrebleu.metrics import CHRF


_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")
_DIGIT_RE = re.compile(r"\d+")

_MEASURE_UNITS = {
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
    "gur",
    "qa",
    "sila",
}

_WITNESS_MARKERS = {
    "witness",
    "witnesses",
    "in the presence",
    "igi",
    "seal",
    "sealed",
}

_DATE_MARKERS = {
    "date",
    "dated",
    "day",
    "month",
    "year",
    "limmu",
    "eponym",
    "eponymy",
    "itu",
    "u4",
}


def corpus_metrics(preds: list[str], gold: list[str]) -> dict[str, float]:
    chrf2 = float(CHRF(word_order=2).corpus_score(preds, [gold]).score)
    bleu = float(sacrebleu.corpus_bleu(preds, [gold]).score)
    combined = float(math.sqrt(max(0.0, bleu) * max(0.0, chrf2)))
    return {
        "bleu": bleu,
        "chrf2": chrf2,
        "combined": combined,
    }


def _extract_digits(text: str) -> set[str]:
    return {m.group(0) for m in _DIGIT_RE.finditer(str(text or ""))}


def _extract_measure_units(text: str) -> set[str]:
    toks = {t.lower() for t in _TOKEN_RE.findall(str(text or ""))}
    return {t for t in toks if t in _MEASURE_UNITS}


def _extract_markers(text: str, markers: set[str]) -> set[str]:
    s = str(text or "").lower()
    return {m for m in markers if m in s}


def _preservation_recall(
    preds: list[str],
    gold: list[str],
    extractor,
) -> dict[str, float]:
    support_items = 0
    support_tokens = 0
    matched_tokens = 0
    for p, g in zip(preds, gold):
        gset = extractor(g)
        if not gset:
            continue
        pset = extractor(p)
        support_items += 1
        support_tokens += len(gset)
        matched_tokens += len(gset.intersection(pset))
    recall = float(matched_tokens / support_tokens) if support_tokens > 0 else 0.0
    return {
        "recall": recall,
        "support_items": float(support_items),
        "support_tokens": float(support_tokens),
        "matched_tokens": float(matched_tokens),
    }


def slot_fidelity_metrics(preds: list[str], gold: list[str]) -> dict[str, float]:
    digit = _preservation_recall(preds, gold, _extract_digits)
    measure = _preservation_recall(preds, gold, _extract_measure_units)
    witness = _preservation_recall(preds, gold, lambda s: _extract_markers(s, _WITNESS_MARKERS))
    date = _preservation_recall(preds, gold, lambda s: _extract_markers(s, _DATE_MARKERS))
    witness_date = _preservation_recall(preds, gold, lambda s: _extract_markers(s, _WITNESS_MARKERS | _DATE_MARKERS))
    return {
        "digit_preservation_recall": float(digit["recall"]),
        "digit_support_items": float(digit["support_items"]),
        "measure_unit_preservation_recall": float(measure["recall"]),
        "measure_unit_support_items": float(measure["support_items"]),
        "witness_marker_preservation_recall": float(witness["recall"]),
        "witness_marker_support_items": float(witness["support_items"]),
        "date_marker_preservation_recall": float(date["recall"]),
        "date_marker_support_items": float(date["support_items"]),
        "witness_date_marker_preservation_recall": float(witness_date["recall"]),
        "witness_date_marker_support_items": float(witness_date["support_items"]),
    }
