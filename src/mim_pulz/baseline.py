from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.domain_intent import infer_dialog_domain
from mim_pulz.preprocess import normalize_text


def _tokenize_words(s: str) -> list[str]:
    # cheap tokenization: split on spaces, keep diacritics
    return normalize_text(s).split()


class DualTfidfRerankTranslator:
    """
    Dual-channel retrieval:
      - char_wb TF-IDF for orthography/variants
      - word TF-IDF for structural/template signal
    Then top-k rerank with:
      - combined similarity
      - length ratio sanity
      - diversity penalty to avoid repeating the same train sample everywhere
    """

    def __init__(
        self,
        char_ngram=(3, 6),
        word_ngram=(1, 2),
        alpha_char: float = 0.65,   # weight char channel
        alpha_word: float = 0.35,   # weight word channel
        top_k: int = 30,            # default candidate pool
    ):
        self.alpha_char = alpha_char
        self.alpha_word = alpha_word
        self.top_k = top_k

        self.vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=1,
            max_features=250_000,
        )
        self.vec_word = TfidfVectorizer(
            analyzer="word",
            tokenizer=_tokenize_words,
            preprocessor=normalize_text,
            token_pattern=None,
            ngram_range=word_ngram,
            min_df=1,
            max_features=150_000,
        )

        self.train_char = None
        self.train_word = None
        self.train_targets: list[str] = []
        self.train_src: list[str] = []

    def fit(
        self,
        train_texts: list[str] | None = None,
        train_targets: list[str] | None = None,
        train_src: list[str] | None = None,
        train_tgt: list[str] | None = None,
    ) -> None:
        # allow either train_texts/train_targets (new) or train_src/train_tgt (legacy)
        src = train_texts if train_texts is not None else train_src
        tgt = train_targets if train_targets is not None else train_tgt
        if src is None or tgt is None:
            raise ValueError("fit() expects train_texts/train_targets or train_src/train_tgt")

        train_texts_n = [normalize_text(t) for t in src]
        self.train_src = train_texts_n
        self.train_targets = tgt

        self.train_char = self.vec_char.fit_transform(train_texts_n)
        self.train_word = self.vec_word.fit_transform(train_texts_n)

    def predict(self, test_texts: list[str], top_k: int | None = None) -> list[str]:
        if self.train_char is None or self.train_word is None:
            raise RuntimeError("Model not fit() yet.")

        k = top_k if top_k is not None else self.top_k
        if k <= 0:
            raise ValueError("top_k must be positive")

        test_texts_n = [normalize_text(t) for t in test_texts]
        test_char = self.vec_char.transform(test_texts_n)
        test_word = self.vec_word.transform(test_texts_n)

        sims_char = cosine_similarity(test_char, self.train_char)
        sims_word = cosine_similarity(test_word, self.train_word)
        sims = self.alpha_char * sims_char + self.alpha_word * sims_word

        preds: list[str] = []
        used_idx: set[int] = set()

        for qi, q in enumerate(test_texts_n):
            row = sims[qi]
            cand_idx = row.argsort()[-k:][::-1]

            q_len = max(1, len(q))
            best_idx = int(cand_idx[0])
            best_score = -1e9

            for idx in cand_idx:
                base = float(row[idx])

                # penalize repeating the exact same training example for multiple queries
                diversity_pen = 0.12 if idx in used_idx else 0.0

                # length sanity: prefer similar-sized source texts
                src_len = max(1, len(self.train_src[idx]))
                len_ratio = min(q_len, src_len) / max(q_len, src_len)  # in (0,1]

                # hard penalty if it's absurdly longer/shorter
                absurd_pen = 0.0
                if src_len > 3.0 * q_len:
                    absurd_pen += 0.10
                if q_len > 3.0 * src_len:
                    absurd_pen += 0.10

                score = base + 0.35 * len_ratio - diversity_pen - absurd_pen

                if score > best_score:
                    best_score = score
                    best_idx = int(idx)

            used_idx.add(best_idx)
            preds.append(self.train_targets[best_idx])

        return preds


class DomainAwareDualTfidfTranslator:
    """
    Soft domain separation:
      - global retriever on all data
      - per-domain retrievers when enough samples exist
      - query routed to inferred domain retriever, with global fallback
    """

    def __init__(
        self,
        top_k_global: int = 80,
        top_k_domain: int = 40,
        min_domain_docs: int = 25,
    ):
        self.top_k_global = top_k_global
        self.top_k_domain = top_k_domain
        self.min_domain_docs = min_domain_docs
        self.global_model = DualTfidfRerankTranslator(top_k=top_k_global)
        self.domain_models: dict[str, DualTfidfRerankTranslator] = {}

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

        src_n = [normalize_text(x) for x in src]
        self.global_model.fit(train_texts=src_n, train_targets=tgt)

        bucket_src: dict[str, list[str]] = {}
        bucket_tgt: dict[str, list[str]] = {}
        for s, t in zip(src_n, tgt):
            d = infer_dialog_domain(s)
            bucket_src.setdefault(d, []).append(s)
            bucket_tgt.setdefault(d, []).append(t)

        self.domain_models = {}
        for domain, ds in bucket_src.items():
            if len(ds) < self.min_domain_docs:
                continue
            m = DualTfidfRerankTranslator(top_k=self.top_k_domain)
            m.fit(train_texts=ds, train_targets=bucket_tgt[domain])
            self.domain_models[domain] = m

    def predict(self, test_texts: list[str]) -> list[str]:
        out: list[str] = []
        for q in test_texts:
            qn = normalize_text(q)
            domain = infer_dialog_domain(qn)
            if domain in self.domain_models:
                out.append(self.domain_models[domain].predict([qn], top_k=self.top_k_domain)[0])
            else:
                out.append(self.global_model.predict([qn], top_k=self.top_k_global)[0])
        return out


# Backwards compatibility alias for older scripts/notebooks
TfidfNearestNeighborTranslator = DualTfidfRerankTranslator
