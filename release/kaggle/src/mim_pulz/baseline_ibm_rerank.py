from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.preprocess import normalize_text
from mim_pulz.ibm1 import IBM1

class DualTfidfIbmRerankTranslator:
    def __init__(self, top_k: int = 80):
        self.top_k = top_k

        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), max_features=250_000)
        self.vec_word = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda s: normalize_text(s).split(),
            preprocessor=normalize_text,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=150_000,
        )

        self.train_char = None
        self.train_word = None
        self.train_src: list[str] = []
        self.train_tgt: list[str] = []

        self.ibm = IBM1(iters=6)

    def fit(
        self,
        train_src: list[str] | None = None,
        train_tgt: list[str] | None = None,
        train_texts: list[str] | None = None,
        train_targets: list[str] | None = None,
    ) -> None:
        # accept either legacy (train_src/train_tgt) or new (train_texts/train_targets) names
        src = train_src if train_src is not None else train_texts
        tgt = train_tgt if train_tgt is not None else train_targets
        if src is None or tgt is None:
            raise ValueError("fit() expects train_src/train_tgt or train_texts/train_targets")

        self.train_src = [normalize_text(s) for s in src]
        self.train_tgt = [str(t) for t in tgt]

        self.train_char = self.vec_char.fit_transform(self.train_src)
        self.train_word = self.vec_word.fit_transform(self.train_src)

        self.ibm.fit(self.train_src, self.train_tgt)

    def predict(self, test_src: list[str], top_k: int | None = None) -> list[str]:
        preds = []

        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k must be positive")

        for q in test_src:
            qn = normalize_text(q)
            q_char = self.vec_char.transform([qn])
            q_word = self.vec_word.transform([qn])

            sims_char = cosine_similarity(q_char, self.train_char)[0]
            sims_word = cosine_similarity(q_word, self.train_word)[0]
            sims = 0.65 * sims_char + 0.35 * sims_word

            cand = sims.argsort()[-k:][::-1]
            q_len = max(1, len(qn))

            best_idx = int(cand[0])
            best_score = -1e9

            for idx in cand:
                idx = int(idx)
                base = float(sims[idx])

                # length sanity on SOURCE
                src_len = max(1, len(self.train_src[idx]))
                len_ratio = min(q_len, src_len) / max(q_len, src_len)

                # alignment score (query src vs candidate tgt)
                align = self.ibm.score_src_tgt(qn, self.train_tgt[idx])

                # combine (tuneable weights)
                score = base + 0.25 * len_ratio + 0.90 * align

                if score > best_score:
                    best_score = score
                    best_idx = idx

            preds.append(self.train_tgt[best_idx])

        return preds
