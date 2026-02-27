from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
import re
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
from mim_pulz.domain_intent import infer_dialog_domain_with_confidence
from utils_manifest import write_json


@dataclass(frozen=True)
class EvalRow:
    family: str
    config: dict
    chrf2: float
    chrf2_reference: float
    chrf2_abs_diff: float
    overrides: int = 0
    override_rate: float = 0.0


def _split_idx(n: int, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(n * val_frac)
    return idx[val_size:], idx[:val_size]


def _score(preds: list[str], gold: list[str]) -> tuple[float, float, float]:
    metric = CHRF(word_order=2)
    s1 = float(metric.corpus_score(preds, [gold]).score)
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return s1, s2, float(abs(s1 - s2))


def _normalize_text(text: str, mode: str) -> str:
    s = str(text or "")
    if "strip_punct" in mode:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if "lower" in mode:
        s = s.lower()
    if "collapse_ws" in mode:
        s = " ".join(s.split())
    return s


def _len_score(q_len: int, d_len: int, mode: str) -> float:
    q = max(1, int(q_len))
    d = max(1, int(d_len))
    if mode == "ratio":
        return float(min(q, d) / max(q, d))
    if mode == "logexp":
        return float(np.exp(-abs(np.log((d + 1.0) / (q + 1.0)))))
    raise ValueError(f"Unknown len mode: {mode}")


def _top_idx_row(row: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, len(row))))
    idx = np.argpartition(row, -k)[-k:]
    return idx[np.argsort(row[idx])[::-1]]


class CharBM25:
    def __init__(self, ngram_range: tuple[int, int], k1: float = 1.2, b: float = 0.5):
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
            raise RuntimeError("BM25 not fitted")
        qv = self.vectorizer.transform([query]).tocsr()
        scores = np.zeros(self.doc_term_csr.shape[0], dtype=np.float32)
        if qv.nnz == 0:
            return scores
        for term, qtf in zip(qv.indices, qv.data):
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


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return float(inter / union) if union else 0.0


def run_local_search(
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
    df[src_col] = df[src_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")

    train_idx, val_idx = _split_idx(len(df), seed=seed, val_frac=val_frac)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_tgt = train_df[tgt_col].tolist()
    val_tgt = val_df[tgt_col].tolist()
    raw_train_src = train_df[src_col].tolist()
    raw_val_src = val_df[src_col].tolist()

    normalization_modes = [
        "collapse_ws",
        "collapse_ws_lower",
        "strip_punct_collapse_ws",
        "strip_punct_collapse_ws_lower",
    ]
    ngram_grid = [(2, 5), (3, 5), (3, 6), (4, 7)]
    top_k_grid = [80, 120, 160, 200]
    len_weight_grid = [0.1, 0.2, 0.3, 0.4]
    len_mode_grid = ["ratio", "logexp"]

    first_stage_rows: list[EvalRow] = []
    caches: dict[tuple[str, tuple[int, int]], dict] = {}

    for norm_mode, ngram_range in product(normalization_modes, ngram_grid):
        train_src = [_normalize_text(x, norm_mode) for x in raw_train_src]
        val_src = [_normalize_text(x, norm_mode) for x in raw_val_src]
        train_lens = np.asarray([len(x) for x in train_src], dtype=np.int32)
        val_lens = np.asarray([len(x) for x in val_src], dtype=np.int32)

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=1,
            max_features=300_000,
        )
        x_train = vec.fit_transform(train_src)
        x_val = vec.transform(val_src)
        sim = cosine_similarity(x_val, x_train)
        top200 = np.vstack([_top_idx_row(sim[i], 200) for i in range(sim.shape[0])])

        caches[(norm_mode, ngram_range)] = {
            "train_src": train_src,
            "val_src": val_src,
            "train_lens": train_lens,
            "val_lens": val_lens,
            "sim": sim,
            "top200": top200,
        }

        for top_k, len_weight, len_mode in product(top_k_grid, len_weight_grid, len_mode_grid):
            picked = []
            for i in range(sim.shape[0]):
                cand = top200[i, :top_k]
                q_len = int(val_lens[i])
                best_idx = int(cand[0])
                best_score = float("-inf")
                for idx in cand:
                    length_term = _len_score(q_len=q_len, d_len=int(train_lens[idx]), mode=len_mode)
                    score = float(sim[i, idx] + len_weight * length_term)
                    if score > best_score:
                        best_score = score
                        best_idx = int(idx)
                picked.append(best_idx)
            preds = [train_tgt[j] for j in picked]
            s1, s2, diff = _score(preds, val_tgt)
            first_stage_rows.append(
                EvalRow(
                    family="first_stage",
                    config={
                        "normalization": norm_mode,
                        "ngram_range": list(ngram_range),
                        "top_k": int(top_k),
                        "len_weight": float(len_weight),
                        "len_mode": len_mode,
                    },
                    chrf2=s1,
                    chrf2_reference=s2,
                    chrf2_abs_diff=diff,
                )
            )

    first_stage_rows = sorted(first_stage_rows, key=lambda r: r.chrf2, reverse=True)
    best_first = first_stage_rows[0]
    best_cfg = best_first.config
    best_norm = str(best_cfg["normalization"])
    best_ngram = (int(best_cfg["ngram_range"][0]), int(best_cfg["ngram_range"][1]))
    best_top_k = int(best_cfg["top_k"])
    best_len_weight = float(best_cfg["len_weight"])
    best_len_mode = str(best_cfg["len_mode"])
    cache = caches[(best_norm, best_ngram)]

    # Stage-2 reranker sweep around best first-stage config.
    stage2_rows: list[EvalRow] = []
    stage2_pool_grid = [30, 50, 80]
    stage2_type_grid = ["seq_ratio", "token_overlap", "bm25"]
    stage2_weight_grid = [0.05, 0.10, 0.20, 0.30, 0.40]
    len_weight_local = sorted(set([max(0.0, best_len_weight - 0.1), best_len_weight, best_len_weight + 0.1]))

    train_src_best = cache["train_src"]
    val_src_best = cache["val_src"]
    train_lens_best = cache["train_lens"]
    val_lens_best = cache["val_lens"]
    sim_best = cache["sim"]
    top200_best = cache["top200"]

    bm25 = CharBM25(ngram_range=best_ngram, k1=1.2, b=0.5)
    bm25.fit(train_src_best)
    bm25_scores = np.vstack([bm25.score_one(q) for q in val_src_best])

    for stage2_type, stage2_pool, len_weight in product(stage2_type_grid, stage2_pool_grid, len_weight_local):
        for stage2_weight in stage2_weight_grid:
            picked = []
            for i in range(sim_best.shape[0]):
                cand = top200_best[i, :best_top_k]
                q_len = int(val_lens_best[i])
                pool = cand[: min(stage2_pool, len(cand))]
                base_scores = []
                stage2_raw = []
                for idx in pool:
                    base = float(
                        sim_best[i, idx]
                        + len_weight * _len_score(q_len=q_len, d_len=int(train_lens_best[idx]), mode=best_len_mode)
                    )
                    base_scores.append(base)
                    if stage2_type == "seq_ratio":
                        s2 = SequenceMatcher(None, val_src_best[i], train_src_best[idx]).ratio()
                    elif stage2_type == "token_overlap":
                        s2 = _token_overlap(val_src_best[i], train_src_best[idx])
                    elif stage2_type == "bm25":
                        s2 = float(bm25_scores[i, idx])
                    else:
                        raise ValueError(stage2_type)
                    stage2_raw.append(float(s2))
                s2_arr = np.asarray(stage2_raw, dtype=np.float32)
                s2_min = float(s2_arr.min()) if len(s2_arr) else 0.0
                s2_max = float(s2_arr.max()) if len(s2_arr) else 0.0
                if s2_max > s2_min:
                    s2_norm = (s2_arr - s2_min) / (s2_max - s2_min)
                else:
                    s2_norm = np.zeros_like(s2_arr)
                final = np.asarray(base_scores, dtype=np.float32) + float(stage2_weight) * s2_norm
                best_local = int(np.argmax(final))
                picked.append(int(pool[best_local]))
            preds = [train_tgt[j] for j in picked]
            s1, s2, diff = _score(preds, val_tgt)
            stage2_rows.append(
                EvalRow(
                    family="stage2",
                    config={
                        "normalization": best_norm,
                        "ngram_range": list(best_ngram),
                        "top_k": best_top_k,
                        "len_weight": float(len_weight),
                        "len_mode": best_len_mode,
                        "stage2_type": stage2_type,
                        "stage2_pool": int(stage2_pool),
                        "stage2_weight": float(stage2_weight),
                    },
                    chrf2=s1,
                    chrf2_reference=s2,
                    chrf2_abs_diff=diff,
                )
            )

    stage2_rows = sorted(stage2_rows, key=lambda r: r.chrf2, reverse=True)

    # Domain override tuning on best first-stage base config.
    domain_rows: list[EvalRow] = []
    train_domains = []
    for s in train_src_best:
        d, _, _ = infer_dialog_domain_with_confidence(s)
        train_domains.append(d)
    val_domain_info = [infer_dialog_domain_with_confidence(s) for s in val_src_best]

    bonus_grid = [0.01, 0.02, 0.03, 0.04, 0.06]
    conf_grid = [0.35, 0.40, 0.45, 0.55]
    margin_grid = [0.01, 0.02, 0.05]
    domain_k_grid = [120, 160, 200, 300]

    for domain_bonus, conf_thr, margin, domain_k in product(
        bonus_grid, conf_grid, margin_grid, domain_k_grid
    ):
        picked = []
        override_count = 0
        for i in range(sim_best.shape[0]):
            q_len = int(val_lens_best[i])
            cand_global = top200_best[i, :best_top_k]
            best_global_idx = int(cand_global[0])
            best_global_score = float("-inf")
            for idx in cand_global:
                score = float(
                    sim_best[i, idx]
                    + best_len_weight * _len_score(q_len=q_len, d_len=int(train_lens_best[idx]), mode=best_len_mode)
                )
                if score > best_global_score:
                    best_global_score = score
                    best_global_idx = int(idx)

            chosen_idx = best_global_idx
            chosen_score = best_global_score

            dom_label, dom_conf, _ = val_domain_info[i]
            if dom_label != "unknown" and float(dom_conf) >= float(conf_thr):
                cand_domain = top200_best[i, : min(domain_k, top200_best.shape[1])]
                dom_best_idx = int(cand_domain[0])
                dom_best_score = float("-inf")
                for idx in cand_domain:
                    base = float(
                        sim_best[i, idx]
                        + best_len_weight
                        * _len_score(q_len=q_len, d_len=int(train_lens_best[idx]), mode=best_len_mode)
                    )
                    if train_domains[idx] == dom_label:
                        base += float(domain_bonus)
                    if base > dom_best_score:
                        dom_best_score = base
                        dom_best_idx = int(idx)

                if (
                    dom_best_idx != best_global_idx
                    and train_domains[dom_best_idx] == dom_label
                    and (dom_best_score - best_global_score) >= float(margin)
                ):
                    chosen_idx = dom_best_idx
                    chosen_score = dom_best_score
                    override_count += 1

            _ = chosen_score
            picked.append(chosen_idx)

        preds = [train_tgt[j] for j in picked]
        s1, s2, diff = _score(preds, val_tgt)
        domain_rows.append(
            EvalRow(
                family="domain_override",
                config={
                    "normalization": best_norm,
                    "ngram_range": list(best_ngram),
                    "top_k": best_top_k,
                    "len_weight": best_len_weight,
                    "len_mode": best_len_mode,
                    "domain_bonus": float(domain_bonus),
                    "domain_conf_threshold": float(conf_thr),
                    "domain_margin": float(margin),
                    "domain_candidate_top_k": int(domain_k),
                },
                chrf2=s1,
                chrf2_reference=s2,
                chrf2_abs_diff=diff,
                overrides=int(override_count),
                override_rate=float(override_count / len(val_tgt)),
            )
        )

    domain_rows = sorted(domain_rows, key=lambda r: r.chrf2, reverse=True)

    best_overall = max(
        [best_first] + ([stage2_rows[0]] if stage2_rows else []) + ([domain_rows[0]] if domain_rows else []),
        key=lambda r: r.chrf2,
    )

    return {
        "seed": int(seed),
        "val_frac": float(val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "best_first_stage": asdict(best_first),
        "best_stage2": asdict(stage2_rows[0]) if stage2_rows else None,
        "best_domain_override": asdict(domain_rows[0]) if domain_rows else None,
        "best_overall": asdict(best_overall),
        "top10_first_stage": [asdict(r) for r in first_stage_rows[:10]],
        "top10_stage2": [asdict(r) for r in stage2_rows[:10]],
        "top10_domain_override": [asdict(r) for r in domain_rows[:10]],
        "metric_crosscheck_max_abs_diff": float(
            max(
                [r.chrf2_abs_diff for r in first_stage_rows]
                + [r.chrf2_abs_diff for r in stage2_rows]
                + [r.chrf2_abs_diff for r in domain_rows]
            )
        ),
        "rows_evaluated": int(len(first_stage_rows) + len(stage2_rows) + len(domain_rows)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local search around canonical retrieval winner + stage2 reranker + domain override tuning."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_local_search_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_local_search(
        competition_dir=args.competition_dir,
        schema_path=args.schema,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print("Best first-stage")
    print(f"chrF2\t{summary['best_first_stage']['chrf2']:.4f}")
    print(f"config\t{summary['best_first_stage']['config']}")
    print("")
    print("Best stage2")
    print(f"chrF2\t{summary['best_stage2']['chrf2']:.4f}")
    print(f"config\t{summary['best_stage2']['config']}")
    print("")
    print("Best domain override")
    print(f"chrF2\t{summary['best_domain_override']['chrf2']:.4f}")
    print(f"overrides\t{summary['best_domain_override']['overrides']}")
    print(f"config\t{summary['best_domain_override']['config']}")
    print("")
    print("Best overall")
    print(f"family\t{summary['best_overall']['family']}")
    print(f"chrF2\t{summary['best_overall']['chrf2']:.4f}")
    print(f"config\t{summary['best_overall']['config']}")
    print("")
    print(f"Metric crosscheck max abs diff\t{summary['metric_crosscheck_max_abs_diff']:.8f}")
    print(f"Rows evaluated\t{summary['rows_evaluated']}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
