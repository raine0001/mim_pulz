from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.preprocess import normalize_text
from utils_manifest import write_json


@dataclass(frozen=True)
class SweepRow:
    family: str
    config: dict
    chrf2: float
    chrf2_reference: float
    chrf2_abs_diff: float


def _split_train_val(n: int, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(n * val_frac)
    return idx[val_size:], idx[:val_size]


def _select_best_idx(
    scores: np.ndarray,
    query_len: int,
    train_lens: np.ndarray,
    top_k: int,
    len_weight: float,
) -> int:
    k = int(max(1, min(top_k, len(scores))))
    cand = np.argpartition(scores, -k)[-k:]
    cand = cand[np.argsort(scores[cand])[::-1]]
    q_len = max(1, int(query_len))
    best_idx = int(cand[0])
    best_score = float("-inf")
    for idx in cand:
        base = float(scores[idx])
        d_len = max(1, int(train_lens[idx]))
        len_ratio = min(q_len, d_len) / max(q_len, d_len)
        score = base + float(len_weight) * len_ratio
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def _score_preds(preds: list[str], gold: list[str]) -> tuple[float, float, float]:
    metric = CHRF(word_order=2)
    s1 = float(metric.corpus_score(preds, [gold]).score)
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return s1, s2, float(abs(s1 - s2))


class CharBM25:
    def __init__(self, ngram_range: tuple[int, int], k1: float, b: float):
        self.ngram_range = ngram_range
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1)
        self.doc_term_csr: sparse.csr_matrix | None = None
        self.doc_term_csc: sparse.csc_matrix | None = None
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0

    def fit(self, docs: list[str]) -> None:
        x = self.vectorizer.fit_transform(docs).astype(np.float32).tocsr()
        self.doc_term_csr = x
        self.doc_term_csc = x.tocsc()
        n_docs = x.shape[0]
        df = np.diff(self.doc_term_csc.indptr).astype(np.float32)
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        self.doc_len = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if n_docs else 0.0

    def score_one(self, query: str) -> np.ndarray:
        if self.doc_term_csr is None or self.doc_term_csc is None or self.idf is None or self.doc_len is None:
            raise RuntimeError("BM25 index is not fitted")
        qv = self.vectorizer.transform([query]).tocsr()
        scores = np.zeros(self.doc_term_csr.shape[0], dtype=np.float32)
        if qv.nnz == 0:
            return scores
        q_terms = qv.indices
        q_counts = qv.data
        for term, qtf in zip(q_terms, q_counts):
            start = self.doc_term_csc.indptr[term]
            end = self.doc_term_csc.indptr[term + 1]
            if start == end:
                continue
            docs = self.doc_term_csc.indices[start:end]
            tfs = self.doc_term_csc.data[start:end]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * (self.doc_len[docs] / max(self.avgdl, 1e-6)))
            term_scores = self.idf[term] * (tfs * (self.k1 + 1.0) / denom)
            q_weight = 1.0 + np.log1p(float(qtf))
            scores[docs] += term_scores * q_weight
        return scores


def run_sweep(
    *,
    competition_dir: Path,
    schema_path: Path,
    seed: int,
    val_frac: float,
) -> dict:
    data = load_deep_past_competition(competition_dir, schema_path)
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col
    df = data.train.copy()
    df[src_col] = df[src_col].astype(str).fillna("").map(normalize_text)
    df[tgt_col] = df[tgt_col].astype(str).fillna("")

    train_idx, val_idx = _split_train_val(len(df), seed=seed, val_frac=val_frac)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()
    train_lens = np.asarray([len(s) for s in train_src], dtype=np.int32)

    tfidf_rows: list[SweepRow] = []
    tfidf_ngrams = [(3, 6), (2, 6), (3, 5), (4, 6)]
    tfidf_topk = [20, 40, 80, 120]
    tfidf_len_weight = [0.0, 0.2, 0.35, 0.5]
    for ngram_range in tfidf_ngrams:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1, max_features=300_000)
        x_train = vec.fit_transform(train_src)
        x_val = vec.transform(val_src)
        sim = cosine_similarity(x_val, x_train)
        q_lens = np.asarray([len(s) for s in val_src], dtype=np.int32)
        for top_k, len_weight in product(tfidf_topk, tfidf_len_weight):
            picked_idx = [
                _select_best_idx(sim[i], int(q_lens[i]), train_lens, top_k=top_k, len_weight=len_weight)
                for i in range(len(val_src))
            ]
            preds = [train_tgt[i] for i in picked_idx]
            s1, s2, diff = _score_preds(preds, val_tgt)
            tfidf_rows.append(
                SweepRow(
                    family="char_tfidf",
                    config={
                        "analyzer": "char_wb",
                        "ngram_range": list(ngram_range),
                        "top_k": int(top_k),
                        "len_weight": float(len_weight),
                    },
                    chrf2=s1,
                    chrf2_reference=s2,
                    chrf2_abs_diff=diff,
                )
            )

    bm25_rows: list[SweepRow] = []
    bm25_ngrams = [(3, 6), (3, 5)]
    bm25_k1 = [1.2, 1.6, 2.0]
    bm25_b = [0.5, 0.75]
    bm25_topk = [20, 40, 80, 120]
    bm25_len_weight = [0.0, 0.2, 0.35]
    q_lens = np.asarray([len(s) for s in val_src], dtype=np.int32)
    for ngram_range, k1, b in product(bm25_ngrams, bm25_k1, bm25_b):
        bm = CharBM25(ngram_range=ngram_range, k1=float(k1), b=float(b))
        bm.fit(train_src)
        bm25_scores = [bm.score_one(q) for q in val_src]
        for top_k, len_weight in product(bm25_topk, bm25_len_weight):
            picked_idx = [
                _select_best_idx(bm25_scores[i], int(q_lens[i]), train_lens, top_k=top_k, len_weight=len_weight)
                for i in range(len(val_src))
            ]
            preds = [train_tgt[i] for i in picked_idx]
            s1, s2, diff = _score_preds(preds, val_tgt)
            bm25_rows.append(
                SweepRow(
                    family="char_bm25",
                    config={
                        "analyzer": "char_wb",
                        "ngram_range": list(ngram_range),
                        "k1": float(k1),
                        "b": float(b),
                        "top_k": int(top_k),
                        "len_weight": float(len_weight),
                    },
                    chrf2=s1,
                    chrf2_reference=s2,
                    chrf2_abs_diff=diff,
                )
            )

    all_rows = tfidf_rows + bm25_rows
    all_rows_sorted = sorted(all_rows, key=lambda r: r.chrf2, reverse=True)
    best_tfidf = max(tfidf_rows, key=lambda r: r.chrf2)
    best_bm25 = max(bm25_rows, key=lambda r: r.chrf2)
    best_overall = all_rows_sorted[0]
    return {
        "seed": int(seed),
        "val_frac": float(val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "best_overall": asdict(best_overall),
        "best_tfidf": asdict(best_tfidf),
        "best_bm25": asdict(best_bm25),
        "top10": [asdict(r) for r in all_rows_sorted[:10]],
        "metric_crosscheck_max_abs_diff": float(max(r.chrf2_abs_diff for r in all_rows)),
        "rows_evaluated": len(all_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieval optimization sweep (char TF-IDF + char BM25) with top_k and length tie-break tuning."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_sweep_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_sweep(
        competition_dir=args.competition_dir,
        schema_path=args.schema,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print("Best overall")
    print(f"family\t{summary['best_overall']['family']}")
    print(f"chrF2\t{summary['best_overall']['chrf2']:.4f}")
    print(f"config\t{summary['best_overall']['config']}")
    print("")
    print("Best char_tfidf")
    print(f"chrF2\t{summary['best_tfidf']['chrf2']:.4f}")
    print(f"config\t{summary['best_tfidf']['config']}")
    print("")
    print("Best char_bm25")
    print(f"chrF2\t{summary['best_bm25']['chrf2']:.4f}")
    print(f"config\t{summary['best_bm25']['config']}")
    print("")
    print(f"Metric crosscheck max abs diff\t{summary['metric_crosscheck_max_abs_diff']:.8f}")
    print(f"Rows evaluated\t{summary['rows_evaluated']}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
