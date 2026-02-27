from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
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
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator
from utils_manifest import write_json


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


_PUNCT_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2036": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def _normalize_text(text: str, mode: str) -> str:
    s = str(text or "")
    if mode == "keep_all":
        return s
    if "norm_punct" in mode:
        s = s.translate(_PUNCT_TRANSLATE)
    if "strip_punct" in mode:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if "lower" in mode:
        s = s.lower()
    if "collapse_ws" in mode:
        s = " ".join(s.split())
    return s


def _normalize01(v: np.ndarray) -> np.ndarray:
    mn = float(v.min())
    mx = float(v.max())
    if mx > mn:
        return (v - mn) / (mx - mn)
    return np.zeros_like(v, dtype=np.float32)


def _repetition_ratio_3gram(text: str) -> float:
    toks = (text or "").split()
    if len(toks) < 3:
        return 0.0
    grams = [tuple(toks[i : i + 3]) for i in range(len(toks) - 2)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return float(max(c.values()) / len(grams))


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


@dataclass(frozen=True)
class EvalRow:
    family: str
    config: dict
    chrf2: float
    chrf2_reference: float
    chrf2_abs_diff: float
    delta_vs_canonical: float


def _top_idx_row(row: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, len(row))))
    idx = np.argpartition(row, -k)[-k:]
    return idx[np.argsort(row[idx])[::-1]]


def _predict_indices(
    *,
    sim: np.ndarray,
    bm25_scores: np.ndarray,
    top_idx: np.ndarray,
    len_ratio: np.ndarray,
    rep_ratio: np.ndarray,
    top_k: int,
    blend_value: float,
    stage2_pool: int | None,
    len_weight: float,
    scoring_mode: str,
    dynamic_thresholds: tuple[float, float] | None = None,
) -> list[int]:
    picks: list[int] = []
    n_val = sim.shape[0]
    for i in range(n_val):
        cand = top_idx[i, :top_k]
        if dynamic_thresholds is None:
            pool_n = int(max(1, min(int(stage2_pool or top_k), len(cand))))
        else:
            t_high, t_mid = dynamic_thresholds
            s1_top = float(sim[i, cand[0]])
            if s1_top >= t_high:
                pool_n = min(30, len(cand))
            elif s1_top >= t_mid:
                pool_n = min(80, len(cand))
            else:
                pool_n = min(120, len(cand))
            pool_n = int(max(1, pool_n))
        pool = cand[:pool_n]
        s1_raw = sim[i, pool].astype(np.float32)
        s1 = _normalize01(s1_raw)
        s2 = _normalize01(bm25_scores[i, pool].astype(np.float32))
        if scoring_mode == "alpha_blend":
            final = blend_value * s1 + (1.0 - blend_value) * s2 + float(len_weight) * len_ratio[i, pool]
        elif scoring_mode == "additive":
            final = s1_raw + float(len_weight) * len_ratio[i, pool] + blend_value * s2
        else:
            raise ValueError(f"Unknown scoring_mode: {scoring_mode}")
        best_val = float(np.max(final))
        tied = np.where(np.isclose(final, best_val, atol=1e-9))[0]
        if tied.size == 1:
            best_local = int(tied[0])
        else:
            tie_idx = pool[tied]
            tie_rep = rep_ratio[tie_idx]
            best_local = int(tied[int(np.argmin(tie_rep))])
        picks.append(int(pool[best_local]))
    return picks


def run_surgical_sweep(
    *,
    competition_dir: Path,
    schema_path: Path,
    seed: int,
    val_frac: float,
    scoring_mode: str = "both",
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

    canonical = CanonicalRetrievalTranslator(config=CANONICAL_BASELINE_V2)
    canonical.fit(train_src=raw_train_src, train_tgt=train_tgt)
    canonical_preds = canonical.predict(raw_val_src)
    canonical_s1, canonical_s2, canonical_diff = _score(canonical_preds, val_tgt)

    norm_modes = [
        "keep_all",
        "collapse_ws",
        "collapse_ws_lower",
        "collapse_ws_strip_punct",
        "collapse_ws_norm_punct",
    ]
    stage1_top_k_grid = [120, 160, 200]
    stage2_pool_grid = [30, 50, 80, 120]
    alpha_grid = [0.2, 0.35, 0.5, 0.65, 0.8]
    additive_weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    len_weight_grid = [0.0, 0.1, 0.2, 0.3]
    ngram_range = (3, 5)
    max_top_k = max(stage1_top_k_grid)

    norm_cache: dict[str, dict] = {}
    rep_ratio = np.asarray([_repetition_ratio_3gram(t) for t in train_tgt], dtype=np.float32)

    for mode in norm_modes:
        train_src = [_normalize_text(x, mode) for x in raw_train_src]
        val_src = [_normalize_text(x, mode) for x in raw_val_src]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=1, max_features=300_000)
        x_train = vec.fit_transform(train_src)
        x_val = vec.transform(val_src)
        sim = cosine_similarity(x_val, x_train).astype(np.float32)

        top_idx = np.vstack([_top_idx_row(sim[i], max_top_k) for i in range(sim.shape[0])]).astype(np.int32)

        bm25 = CharBM25(ngram_range=ngram_range, k1=1.2, b=0.5)
        bm25.fit(train_src)
        bm25_scores = np.vstack([bm25.score_one(q) for q in val_src]).astype(np.float32)

        train_lens = np.asarray([max(1, len(s)) for s in train_src], dtype=np.int32)
        val_lens = np.asarray([max(1, len(s)) for s in val_src], dtype=np.int32)
        len_ratio = (
            np.minimum(val_lens[:, None], train_lens[None, :]) / np.maximum(val_lens[:, None], train_lens[None, :])
        ).astype(np.float32)

        norm_cache[mode] = {
            "sim": sim,
            "top_idx": top_idx,
            "bm25_scores": bm25_scores,
            "len_ratio": len_ratio,
        }

    # Step A/B preselect top-2 normalizations with one anchor setup.
    probe_rows: list[EvalRow] = []
    for mode in norm_modes:
        c = norm_cache[mode]
        picks = _predict_indices(
            sim=c["sim"],
            bm25_scores=c["bm25_scores"],
            top_idx=c["top_idx"],
            len_ratio=c["len_ratio"],
            rep_ratio=rep_ratio,
            top_k=120,
            blend_value=0.2,
            stage2_pool=80,
            len_weight=0.2,
            scoring_mode="additive",
        )
        preds = [train_tgt[j] for j in picks]
        s1, s2, diff = _score(preds, val_tgt)
        probe_rows.append(
            EvalRow(
                family="normalization_probe",
                config={
                    "normalization": mode,
                    "top_k": 120,
                    "stage2_pool": 80,
                    "scoring_mode": "additive",
                    "stage2_weight": 0.2,
                    "len_weight": 0.2,
                },
                chrf2=s1,
                chrf2_reference=s2,
                chrf2_abs_diff=diff,
                delta_vs_canonical=float(s1 - canonical_s1),
            )
        )
    probe_rows = sorted(probe_rows, key=lambda r: r.chrf2, reverse=True)
    top_norms = [probe_rows[0].config["normalization"], probe_rows[1].config["normalization"]]

    # Main surgical sweep
    main_rows: list[EvalRow] = []
    if scoring_mode == "additive":
        scoring_modes = ["additive"]
    elif scoring_mode == "alpha_blend":
        scoring_modes = ["alpha_blend"]
    else:
        scoring_modes = ["additive", "alpha_blend"]

    for mode in top_norms:
        c = norm_cache[mode]
        for mode_name in scoring_modes:
            blend_grid = additive_weight_grid if mode_name == "additive" else alpha_grid
            for top_k, pool, blend_value, len_weight in product(
                stage1_top_k_grid, stage2_pool_grid, blend_grid, len_weight_grid
            ):
                picks = _predict_indices(
                    sim=c["sim"],
                    bm25_scores=c["bm25_scores"],
                    top_idx=c["top_idx"],
                    len_ratio=c["len_ratio"],
                    rep_ratio=rep_ratio,
                    top_k=top_k,
                    blend_value=blend_value,
                    stage2_pool=pool,
                    len_weight=len_weight,
                    scoring_mode=mode_name,
                )
                preds = [train_tgt[j] for j in picks]
                s1, s2, diff = _score(preds, val_tgt)
                cfg = {
                    "normalization": mode,
                    "top_k": int(top_k),
                    "stage2_pool": int(pool),
                    "scoring_mode": mode_name,
                    "len_weight": float(len_weight),
                }
                if mode_name == "additive":
                    cfg["stage2_weight"] = float(blend_value)
                else:
                    cfg["alpha"] = float(blend_value)
                main_rows.append(
                    EvalRow(
                        family="main_surgical",
                        config=cfg,
                        chrf2=s1,
                        chrf2_reference=s2,
                        chrf2_abs_diff=diff,
                        delta_vs_canonical=float(s1 - canonical_s1),
                    )
                )
    main_rows = sorted(main_rows, key=lambda r: r.chrf2, reverse=True)
    best_main = main_rows[0]

    # Optional dynamic pool gate around best static config.
    dynamic_rows: list[EvalRow] = []
    dyn_thresholds = [(0.60, 0.40), (0.65, 0.45), (0.70, 0.50), (0.55, 0.35), (0.68, 0.42)]
    best_mode = str(best_main.config["normalization"])
    c = norm_cache[best_mode]
    dyn_scoring_mode = str(best_main.config.get("scoring_mode", "additive"))
    dyn_blend_value = float(best_main.config.get("stage2_weight", best_main.config.get("alpha", 0.2)))
    for t_high, t_mid in dyn_thresholds:
        if t_mid >= t_high:
            continue
        picks = _predict_indices(
            sim=c["sim"],
            bm25_scores=c["bm25_scores"],
            top_idx=c["top_idx"],
            len_ratio=c["len_ratio"],
            rep_ratio=rep_ratio,
            top_k=int(best_main.config["top_k"]),
            blend_value=dyn_blend_value,
            stage2_pool=None,
            len_weight=float(best_main.config["len_weight"]),
            scoring_mode=dyn_scoring_mode,
            dynamic_thresholds=(float(t_high), float(t_mid)),
        )
        preds = [train_tgt[j] for j in picks]
        s1, s2, diff = _score(preds, val_tgt)
        dynamic_rows.append(
            EvalRow(
                family="dynamic_pool",
                config={
                    "normalization": best_mode,
                    "top_k": int(best_main.config["top_k"]),
                    "scoring_mode": dyn_scoring_mode,
                    "blend_value": dyn_blend_value,
                    "len_weight": float(best_main.config["len_weight"]),
                    "dynamic_threshold_high": float(t_high),
                    "dynamic_threshold_mid": float(t_mid),
                    "dynamic_pools": [30, 80, 120],
                },
                chrf2=s1,
                chrf2_reference=s2,
                chrf2_abs_diff=diff,
                delta_vs_canonical=float(s1 - canonical_s1),
            )
        )
    dynamic_rows = sorted(dynamic_rows, key=lambda r: r.chrf2, reverse=True)
    best_dynamic = dynamic_rows[0] if dynamic_rows else None

    best_overall = best_main
    if best_dynamic is not None and best_dynamic.chrf2 > best_overall.chrf2:
        best_overall = best_dynamic

    return {
        "seed": int(seed),
        "val_frac": float(val_frac),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "search_space": {
            "ngram_range": [3, 5],
            "stage1_top_k": stage1_top_k_grid,
            "stage2_pool": stage2_pool_grid,
            "alpha": alpha_grid,
            "additive_stage2_weight": additive_weight_grid,
            "len_weight": len_weight_grid,
            "normalizations": norm_modes,
            "dynamic_thresholds": dyn_thresholds,
            "scoring_mode": scoring_mode,
        },
        "canonical_reference": {
            "model_id": CANONICAL_BASELINE_V2.model_id(),
            "chrf2": canonical_s1,
            "chrf2_reference": canonical_s2,
            "chrf2_abs_diff": canonical_diff,
        },
        "normalization_probe_top5": [asdict(r) for r in probe_rows[:5]],
        "selected_top2_normalizations": top_norms,
        "best_main_sweep": asdict(best_main),
        "best_dynamic_pool": asdict(best_dynamic) if best_dynamic else None,
        "best_overall": asdict(best_overall),
        "top10_main_sweep": [asdict(r) for r in main_rows[:10]],
        "top5_dynamic_pool": [asdict(r) for r in dynamic_rows[:5]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Targeted retrieval sweep around canonical config (stage-2 pool/alpha/len/normalization)."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--scoring-mode",
        type=str,
        choices=["additive", "alpha_blend", "both"],
        default="both",
        help="Rerank scoring style for sweep.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_surgical_sweep_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run_surgical_sweep(
        competition_dir=args.competition_dir.resolve(),
        schema_path=args.schema.resolve(),
        seed=args.seed,
        val_frac=args.val_frac,
        scoring_mode=args.scoring_mode,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, out)
    print(f"Wrote {args.json_out}")
    print(
        "Best overall chrF2="
        f"{out['best_overall']['chrf2']:.4f} "
        f"(delta_vs_canonical={out['best_overall']['delta_vs_canonical']:+.4f})"
    )


if __name__ == "__main__":
    main()
