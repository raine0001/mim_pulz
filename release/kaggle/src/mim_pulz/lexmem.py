from __future__ import annotations
from collections import defaultdict, Counter
import re

from mim_pulz.preprocess import normalize_text

_word_en = re.compile(r"[A-Za-z][A-Za-z\-']+")
_token_src = re.compile(r"[^\s]+")

def src_tokens(s: str) -> list[str]:
    s = normalize_text(s)
    return _token_src.findall(s)

def en_tokens(s: str) -> list[str]:
    s = normalize_text(s)
    return [m.group(0).lower() for m in _word_en.finditer(s)]

class LexicalMemory:
    """
    Builds a simple translation memory:
      src_token -> Counter(english_token)
    Learned purely from training pairs.
    """

    def __init__(self, min_src_count: int = 3, top_en_per_src: int = 8):
        self.min_src_count = min_src_count
        self.top_en_per_src = top_en_per_src
        self.src_counts = Counter()
        self.map: dict[str, list[tuple[str,int]]] = {}

    def fit(self, src_list: list[str], tgt_list: list[str]) -> None:
        tmp: dict[str, Counter] = defaultdict(Counter)

        for src, tgt in zip(src_list, tgt_list):
            stoks = src_tokens(src)
            etoks = en_tokens(tgt)
            if not stoks or not etoks:
                continue
            for st in stoks:
                self.src_counts[st] += 1
                tmp[st].update(etoks)

        # finalize
        out = {}
        for st, cnt in tmp.items():
            if self.src_counts[st] < self.min_src_count:
                continue
            out[st] = cnt.most_common(self.top_en_per_src)
        self.map = out

    def predict_en_keywords(self, src: str, max_keywords: int = 25) -> list[str]:
        """
        For a given src, return a list of likely English keywords based on token map.
        """
        scores = Counter()
        for st in src_tokens(src):
            for en, w in self.map.get(st, []):
                scores[en] += w
        return [w for w, _ in scores.most_common(max_keywords)]
