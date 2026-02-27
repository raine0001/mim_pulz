from __future__ import annotations
from collections import defaultdict, Counter
import math
import re

from mim_pulz.preprocess import normalize_text

_src_tok = re.compile(r"[^\s]+")
_en_tok = re.compile(r"[A-Za-z][A-Za-z\-']+")

def tokenize_src(s: str) -> list[str]:
    return _src_tok.findall(normalize_text(s))

def tokenize_en(s: str) -> list[str]:
    return [m.group(0).lower() for m in _en_tok.finditer(normalize_text(s))]

class IBM1:
    """
    Very small IBM Model 1 alignment model.
    Learns p(tgt_word | src_token) from parallel pairs.
    """

    def __init__(self, iters: int = 5, max_en_vocab: int = 50000, min_count: int = 2):
        self.iters = iters
        self.max_en_vocab = max_en_vocab
        self.min_count = min_count
        self.t = {}  # dict[src_token][en_word] = prob

    def fit(self, src_list: list[str], tgt_list: list[str]) -> None:
        pairs = []
        en_counts = Counter()

        for s, t in zip(src_list, tgt_list):
            stoks = tokenize_src(s)
            etoks = tokenize_en(t)
            if not stoks or not etoks:
                continue
            en_counts.update(etoks)
            pairs.append((stoks, etoks))

        # prune EN vocab to common words
        en_vocab = {w for w, c in en_counts.most_common(self.max_en_vocab) if c >= self.min_count}
        if not en_vocab:
            en_vocab = set(w for w, _ in en_counts.most_common(5000))

        # init uniform t(e|f) over co-occurrences
        cooc = defaultdict(set)
        for stoks, etoks in pairs:
            et = [w for w in etoks if w in en_vocab]
            for f in stoks:
                cooc[f].update(et)

        t = {}
        for f, ws in cooc.items():
            if not ws:
                continue
            uni = 1.0 / len(ws)
            t[f] = {w: uni for w in ws}

        # EM iterations
        for _ in range(self.iters):
            count_fe = defaultdict(lambda: defaultdict(float))
            total_f = defaultdict(float)

            for stoks, etoks in pairs:
                et = [w for w in etoks if w in en_vocab]
                if not et:
                    continue
                for e in et:
                    # normalization
                    z = 0.0
                    for f in stoks:
                        z += t.get(f, {}).get(e, 0.0)
                    if z == 0.0:
                        continue
                    for f in stoks:
                        tef = t.get(f, {}).get(e, 0.0)
                        if tef == 0.0:
                            continue
                        c = tef / z
                        count_fe[f][e] += c
                        total_f[f] += c

            # update
            for f, ecnts in count_fe.items():
                denom = total_f[f]
                if denom <= 0:
                    continue
                if f not in t:
                    t[f] = {}
                for e, c in ecnts.items():
                    t[f][e] = c / denom

        self.t = t

    def score_src_tgt(self, src: str, tgt: str) -> float:
        """
        Returns log-likelihood-ish score of tgt words given src tokens.
        Not a true probability (we ignore NULL + length terms), but works for reranking.
        """
        stoks = tokenize_src(src)
        etoks = tokenize_en(tgt)
        if not stoks or not etoks:
            return -1e9

        score = 0.0
        eps = 1e-12
        for e in etoks:
            # sum over src tokens
            s = 0.0
            for f in stoks:
                s += self.t.get(f, {}).get(e, 0.0)
            s = max(s, eps)
            score += math.log(s)
        # normalize by length to reduce bias for long/short
        return score / max(1, len(etoks))
