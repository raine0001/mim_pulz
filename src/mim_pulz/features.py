from __future__ import annotations
import math
import re
from dataclasses import dataclass

# Cheap markers common in your corpus
RE_NUM = re.compile(r"(\d+(\.\d+)?)")
RE_TOKEN = re.compile(r"[^\s]+")

def _count_substrings(s: str, subs: list[str]) -> int:
    s_low = s.lower()
    return sum(s_low.count(x) for x in subs)

@dataclass(frozen=True)
class ToneFeatures:
    # document-type proxies
    number_density: float
    seal_score: float
    witness_score: float
    date_score: float
    ledger_score: float
    letter_score: float
    legal_score: float

    # power/tone proxies
    imperative_score: float
    deferential_score: float
    urgency_score: float

    def as_vec(self) -> list[float]:
        return [
            self.number_density,
            self.seal_score,
            self.witness_score,
            self.date_score,
            self.ledger_score,
            self.letter_score,
            self.legal_score,
            self.imperative_score,
            self.deferential_score,
            self.urgency_score,
        ]

def extract_tone_features(text: str) -> ToneFeatures:
    s = (text or "").strip()
    if not s:
        return ToneFeatures(*([0.0] * 10))

    toks = RE_TOKEN.findall(s)
    n_tok = max(1, len(toks))

    # numeric / amounts
    n_nums = len(RE_NUM.findall(s))
    number_density = n_nums / n_tok

    # record / authority markers (very common in OA texts)
    seal_hits = _count_substrings(s, ["kišib", "kunuki", "seal"])
    witness_hits = _count_substrings(s, ["igi", "witness", "in the presence"])  # IGI is the big one
    date_hits = _count_substrings(s, ["itu", "limu", "eponymy", "u4", "warḫ", "month"])

    # trade/ledger markers
    ledger_hits = _count_substrings(s, ["kù.babbar", "gín", "ma-na", "mina", "silver", "tin", "urudu", "an.na"])
    # letter markers
    letter_hits = _count_substrings(s, ["um-ma", "qí-bi", "ṭup", "tupp", "té-er-ta", "message", "letter"])
    # legal markers
    legal_hits = _count_substrings(s, ["dí-in", "case", "testimony", "seized", "witness", "oath", "dagger"])

    # normalize-ish
    seal_score = min(1.0, seal_hits / 3.0)
    witness_score = min(1.0, witness_hits / 4.0)
    date_score = min(1.0, date_hits / 4.0)

    ledger_score = min(1.0, (ledger_hits + 2.0 * number_density * n_tok) / 20.0)
    letter_score = min(1.0, letter_hits / 10.0)
    legal_score = min(1.0, (legal_hits + witness_hits + seal_hits) / 15.0)

    # Power/tone proxies (very heuristic, but useful)
    # Imperative-ish: "say to", "send", "do not", "urgent", "let ... come"
    imperative_hits = _count_substrings(s, ["qí-bi", "šé-bi", "li-li-kam", "lá", "send", "do not", "let"])
    imperative_score = min(1.0, imperative_hits / 10.0)

    # Deference-ish: "my lord/master" type proxies (not always present, but keep it)
    defer_hits = _count_substrings(s, ["lord", "master", "my lord", "your servant"])
    deferential_score = min(1.0, defer_hits / 4.0)

    # Urgency proxies
    urg_hits = _count_substrings(s, ["a-pu-tum", "urgent", "the very day", "without delay", "do not hesitate"])
    urgency_score = min(1.0, urg_hits / 5.0)

    return ToneFeatures(
        number_density=number_density,
        seal_score=seal_score,
        witness_score=witness_score,
        date_score=date_score,
        ledger_score=ledger_score,
        letter_score=letter_score,
        legal_score=legal_score,
        imperative_score=imperative_score,
        deferential_score=deferential_score,
        urgency_score=urgency_score,
    )

def doc_type_bucket(f: ToneFeatures) -> str:
    # Pick the dominant “type”
    scores = {
        "ledger": f.ledger_score + 0.6 * f.number_density + 0.4 * f.seal_score,
        "letter": f.letter_score + 0.3 * f.urgency_score + 0.2 * f.imperative_score,
        "legal":  f.legal_score + 0.5 * f.witness_score,
    }
    return max(scores, key=scores.get)

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
