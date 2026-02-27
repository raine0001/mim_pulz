from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mim_pulz.preprocess import normalize_text
from mim_pulz.segment import split_sentences


class SegmentStitchTranslator:
    """
    Stage 1: Retrieve top_k_docs training examples using char+word TF-IDF on transliteration.
    Stage 2: From those docs, gather English sentences and rank them using TF-IDF similarity
             against the *query transliteration* (proxy). Then stitch top sentences.

    This is a pragmatic hack: we don't have alignments, so we use retrieval + sentence selection.
    """

    def __init__(self, top_k_docs: int = 25, top_k_sents: int = 4):
        self.top_k_docs = top_k_docs
        self.top_k_sents = top_k_sents

        # Doc retrieval vectorizers (same idea as before)
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

        # Sentence ranker over EN sentences only (fast)
        self.sent_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=200_000)
        self.sent_matrix = None
        self.sent_texts: list[str] = []
        self.sent_doc_ids: list[int] = []

    def fit(
        self,
        train_src: list[str] | None = None,
        train_tgt: list[str] | None = None,
        train_texts: list[str] | None = None,
        train_targets: list[str] | None = None,
    ) -> None:
        # allow both legacy (train_src/train_tgt) and new (train_texts/train_targets) names
        src = train_src if train_src is not None else train_texts
        tgt = train_tgt if train_tgt is not None else train_targets
        if src is None or tgt is None:
            raise ValueError("fit() expects train_src/train_tgt or train_texts/train_targets")

        self.train_src = [normalize_text(s) for s in src]
        self.train_tgt = [str(t) for t in tgt]

        self.train_char = self.vec_char.fit_transform(self.train_src)
        self.train_word = self.vec_word.fit_transform(self.train_src)

        # Build sentence pool from all training targets
        sent_texts = []
        sent_doc_ids = []
        for doc_id, tgt in enumerate(self.train_tgt):
            for sent in split_sentences(tgt):
                # ignore ultra-short junk
                if len(sent) < 15:
                    continue
                sent_texts.append(sent)
                sent_doc_ids.append(doc_id)

        self.sent_texts = sent_texts
        self.sent_doc_ids = sent_doc_ids
        self.sent_matrix = self.sent_vec.fit_transform(sent_texts)

    def _retrieve_doc_candidates(self, q: str, top_k_docs: int | None = None) -> np.ndarray:
        qn = normalize_text(q)
        q_char = self.vec_char.transform([qn])
        q_word = self.vec_word.transform([qn])

        sims_char = cosine_similarity(q_char, self.train_char)[0]
        sims_word = cosine_similarity(q_word, self.train_word)[0]
        sims = 0.65 * sims_char + 0.35 * sims_word

        k = top_k_docs or self.top_k_docs
        cand = sims.argsort()[-k:][::-1]
        return cand.astype(int)

    def predict(self, test_src: list[str], top_k: int | None = None) -> list[str]:
        if self.sent_matrix is None:
            raise RuntimeError("fit() first")

        k_docs = top_k or self.top_k_docs
        if k_docs <= 0:
            raise ValueError("top_k must be positive")

        outputs: list[str] = []

        # Precompute sentence doc ids as array for masking
        sent_doc_ids = np.array(self.sent_doc_ids, dtype=int)

        for q in test_src:
            cand_docs = self._retrieve_doc_candidates(q, top_k_docs=k_docs)

            # mask sentences belonging to candidate docs
            mask = np.isin(sent_doc_ids, cand_docs)
            if not mask.any():
                # fallback: just return best doc translation
                outputs.append(self.train_tgt[int(cand_docs[0])])
                continue

            # Rank candidate sentences against *English sentence space* using query as weak proxy:
            # We transform query into same space by using sentence vectorizer on query text too.
            # It’s not perfect, but it selects more “generic relevant” sentences.
            q_vec = self.sent_vec.transform([normalize_text(q)])
            sims = cosine_similarity(q_vec, self.sent_matrix)[0]

            # apply mask: invalidate non-candidate sentences
            sims_masked = np.where(mask, sims, -1.0)

            top_sent_idx = sims_masked.argsort()[-self.top_k_sents:][::-1]
            chosen = [self.sent_texts[i] for i in top_sent_idx if sims_masked[i] > 0]

            if not chosen:
                outputs.append(self.train_tgt[int(cand_docs[0])])
            else:
                # stitch with spaces; keep it reasonably sized
                out = " ".join(chosen)
                outputs.append(out)

        return outputs
