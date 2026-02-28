from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.preprocess import normalize_text
from mim_pulz.features import extract_tone_features, doc_type_bucket


class GatedDualTfidfTranslator:
    def __init__(self, top_k: int = 80):
        self.top_k = top_k

        self.vec_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 6), max_features=250_000
        )
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
        self.train_bucket: list[str] = []

    def fit(self, train_src: list[str], train_tgt: list[str]) -> None:
        self.train_src = [normalize_text(s) for s in train_src]
        self.train_tgt = [str(t) for t in train_tgt]

        self.train_char = self.vec_char.fit_transform(self.train_src)
        self.train_word = self.vec_word.fit_transform(self.train_src)

        self.train_bucket = [
            doc_type_bucket(extract_tone_features(s)) for s in self.train_src
        ]

    def predict(self, test_src: list[str]) -> list[str]:
        preds = []

        for q in test_src:
            qn = normalize_text(q)

            q_bucket = doc_type_bucket(extract_tone_features(qn))

            q_char = self.vec_char.transform([qn])
            q_word = self.vec_word.transform([qn])

            sims_char = cosine_similarity(q_char, self.train_char)[0]
            sims_word = cosine_similarity(q_word, self.train_word)[0]
            sims = 0.65 * sims_char + 0.35 * sims_word

            cand = sims.argsort()[-self.top_k :][::-1]

            # Gate: pick best candidate within same bucket if possible
            best_idx = None
            for idx in cand:
                idx = int(idx)
                if self.train_bucket[idx] == q_bucket:
                    best_idx = idx
                    break

            # Fallback: no same-bucket found, just take top cosine
            if best_idx is None:
                best_idx = int(cand[0])

            preds.append(self.train_tgt[best_idx])

        return preds
