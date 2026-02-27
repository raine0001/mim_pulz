from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.preprocess import normalize_text
from mim_pulz.features import extract_tone_features, cosine as cos


class ToneAwareDualTfidfTranslator:
    """
    Start with your best approach: char+word TF-IDF retrieval.
    Add tone/intent compatibility between query transliteration and candidate transliteration.
    Choose the best candidate translation (copy, no generation).

    This keeps it Kaggle-safe and avoids hallucinations.
    """

    def __init__(
        self,
        top_k: int = 60,
        alpha_char: float = 0.65,
        alpha_word: float = 0.35,
        w_len: float = 0.30,
        w_tone: float = 0.35,
        diversity_penalty: float = 0.10,
    ):
        self.top_k = top_k
        self.alpha_char = alpha_char
        self.alpha_word = alpha_word
        self.w_len = w_len
        self.w_tone = w_tone
        self.diversity_penalty = diversity_penalty

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
        self.train_tone_vecs: list[list[float]] = []

    def fit(self, train_src: list[str], train_tgt: list[str]) -> None:
        self.train_src = [normalize_text(s) for s in train_src]
        self.train_tgt = [str(t) for t in train_tgt]

        self.train_char = self.vec_char.fit_transform(self.train_src)
        self.train_word = self.vec_word.fit_transform(self.train_src)

        # Precompute tone vectors for training sources
        self.train_tone_vecs = [extract_tone_features(s).as_vec() for s in self.train_src]

    def predict(self, test_src: list[str]) -> list[str]:
        if self.train_char is None or self.train_word is None:
            raise RuntimeError("fit() first")

        preds: list[str] = []
        used_idx: set[int] = set()

        for q in test_src:
            qn = normalize_text(q)
            q_char = self.vec_char.transform([qn])
            q_word = self.vec_word.transform([qn])

            sims_char = cosine_similarity(q_char, self.train_char)[0]
            sims_word = cosine_similarity(q_word, self.train_word)[0]
            sims = self.alpha_char * sims_char + self.alpha_word * sims_word

            cand = sims.argsort()[-self.top_k:][::-1]

            q_len = max(1, len(qn))
            q_tone = extract_tone_features(qn).as_vec()

            best_idx = int(cand[0])
            best_score = -1e9

            for idx in cand:
                idx = int(idx)
                base = float(sims[idx])

                # length sanity (source-side)
                src_len = max(1, len(self.train_src[idx]))
                len_ratio = min(q_len, src_len) / max(q_len, src_len)

                # tone/intent compatibility
                tone_sim = cos(q_tone, self.train_tone_vecs[idx])

                # discourage picking the same training example repeatedly in a batch
                div_pen = self.diversity_penalty if idx in used_idx else 0.0

                score = base + self.w_len * len_ratio + self.w_tone * tone_sim - div_pen

                if score > best_score:
                    best_score = score
                    best_idx = idx

            used_idx.add(best_idx)
            preds.append(self.train_tgt[best_idx])

        return preds
