from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.retrieval import CANONICAL_BASELINE_V2, CanonicalRetrievalTranslator, RetrievalConfig
from utils_manifest import write_json


def _split(df, seed: int, val_frac: float):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def _score(preds: list[str], gold: list[str]) -> dict:
    metric = CHRF(word_order=2)
    s1 = float(metric.corpus_score(preds, [gold]).score)
    s2 = float(sacrebleu.corpus_chrf(preds, [gold], word_order=2).score)
    return {
        "chrf2": s1,
        "chrf2_reference": s2,
        "abs_diff": float(abs(s1 - s2)),
    }


def _evaluate(
    cfg: RetrievalConfig,
    train_src: list[str],
    train_tgt: list[str],
    val_src: list[str],
    val_tgt: list[str],
) -> dict:
    model = CanonicalRetrievalTranslator(config=cfg)
    model.fit(train_src=train_src, train_tgt=train_tgt)
    preds, debug_rows = model.predict(val_src, return_debug=True)
    score = _score(preds, val_tgt)
    override_count = int(sum(1 for r in debug_rows if r["override_used"]))
    external_pick_count = int(sum(1 for r in debug_rows if str(r.get("chosen_origin", "")) != "competition"))
    return {
        "config": asdict(cfg),
        "score": score,
        "override_count": override_count,
        "override_rate": float(override_count / len(debug_rows)) if debug_rows else 0.0,
        "external_pick_count": external_pick_count,
        "external_pick_rate": float(external_pick_count / len(debug_rows)) if debug_rows else 0.0,
        "memory_total_rows": int(len(model.train_src_raw)),
        "memory_external_rows": int(model.external_rows_loaded),
    }


def _cfg_from_dict(d: dict) -> RetrievalConfig:
    copied = deepcopy(d)
    if isinstance(copied.get("external_memory_paths"), list):
        copied["external_memory_paths"] = tuple(copied["external_memory_paths"])
    if isinstance(copied.get("external_context_allowlist"), list):
        copied["external_context_allowlist"] = tuple(copied["external_context_allowlist"])
    return RetrievalConfig(**copied)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Focused ORACC retrieval sweep with coordinate search + random refinement."
    )
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--memory-csv",
        type=Path,
        default=PATHS.data_processed / "oracc_evacun_memory.csv",
    )
    parser.add_argument("--memory-source-col", type=str, default="source")
    parser.add_argument("--memory-target-col", type=str, default="target")
    parser.add_argument("--memory-context-col", type=str, default="context")
    parser.add_argument("--memory-origin-col", type=str, default="origin")
    parser.add_argument("--coord-rounds", type=int, default=2)
    parser.add_argument("--random-trials", type=int, default=48)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PATHS.root / "artifacts" / "analysis" / "retrieval_oracc_focused_sweep_seed42.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)

    data = load_deep_past_competition(args.competition_dir, args.schema)
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col

    df = data.train.copy()
    df[src_col] = df[src_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")
    train_df, val_df = _split(df, seed=args.seed, val_frac=args.val_frac)

    train_src = train_df[src_col].tolist()
    train_tgt = train_df[tgt_col].tolist()
    val_src = val_df[src_col].tolist()
    val_tgt = val_df[tgt_col].tolist()

    canonical_no_ext = _evaluate(
        CANONICAL_BASELINE_V2,
        train_src=train_src,
        train_tgt=train_tgt,
        val_src=val_src,
        val_tgt=val_tgt,
    )

    start_cfg_dict = asdict(
        RetrievalConfig(
            analyzer=CANONICAL_BASELINE_V2.analyzer,
            ngram_min=CANONICAL_BASELINE_V2.ngram_min,
            ngram_max=CANONICAL_BASELINE_V2.ngram_max,
            max_features=CANONICAL_BASELINE_V2.max_features,
            top_k=120,
            len_weight=CANONICAL_BASELINE_V2.len_weight,
            len_mode=CANONICAL_BASELINE_V2.len_mode,
            strip_punct=CANONICAL_BASELINE_V2.strip_punct,
            lowercase=CANONICAL_BASELINE_V2.lowercase,
            collapse_whitespace=CANONICAL_BASELINE_V2.collapse_whitespace,
            stage2_type=CANONICAL_BASELINE_V2.stage2_type,
            stage2_pool=50,
            stage2_weight=CANONICAL_BASELINE_V2.stage2_weight,
            stage2_bm25_k1=CANONICAL_BASELINE_V2.stage2_bm25_k1,
            stage2_bm25_b=CANONICAL_BASELINE_V2.stage2_bm25_b,
            enable_domain_override=True,
            domain_candidate_top_k=CANONICAL_BASELINE_V2.domain_candidate_top_k,
            domain_bonus=0.03,
            domain_conf_threshold=0.35,
            domain_margin=0.02,
            external_memory_paths=(str(args.memory_csv.resolve()),),
            external_source_col=args.memory_source_col,
            external_target_col=args.memory_target_col,
            external_context_col=args.memory_context_col,
            external_origin_col=args.memory_origin_col,
            external_memory_limit=20_000,
            external_context_allowlist=("legal", "letter"),
            competition_origin_bonus=0.0,
            external_origin_bonus=-0.02,
            evidence_k=1,
        )
    )

    search_space = {
        "top_k": [80, 100, 120, 140, 160, 200],
        "len_weight": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
        "stage2_pool": [30, 40, 50, 60, 80, 100, 120],
        "stage2_weight": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "external_memory_limit": [5_000, 10_000, 15_000, 20_000, 30_000, 0],
        "external_context_allowlist": [
            (),
            ("legal",),
            ("letter",),
            ("legal", "letter"),
            ("economic", "legal"),
            ("economic", "legal", "letter"),
        ],
        "competition_origin_bonus": [0.0, 0.01, 0.02, 0.03, 0.05],
        "external_origin_bonus": [-0.05, -0.03, -0.02, -0.01, 0.0],
        "enable_domain_override": [True, False],
        "domain_bonus": [0.0, 0.01, 0.02, 0.03, 0.05],
        "domain_conf_threshold": [0.25, 0.35, 0.45, 0.55],
    }

    eval_history: list[dict] = []
    seen_keys: set[str] = set()

    def evaluate_cfg_dict(cfg_dict: dict, tag: str) -> dict:
        cfg = _cfg_from_dict(cfg_dict)
        key = json_key(cfg)
        if key in seen_keys:
            return {"cached": True, "key": key}
        seen_keys.add(key)
        result = _evaluate(cfg, train_src=train_src, train_tgt=train_tgt, val_src=val_src, val_tgt=val_tgt)
        result["delta_vs_no_external"] = float(result["score"]["chrf2"] - canonical_no_ext["score"]["chrf2"])
        result["tag"] = tag
        result["key"] = key
        eval_history.append(result)
        print(
            f"[{len(eval_history):03d}] {tag:<20} chrF2={result['score']['chrf2']:.4f} "
            f"delta={result['delta_vs_no_external']:+.4f} ext_pick={result['external_pick_rate']:.3f}"
        )
        return result

    def json_key(cfg: RetrievalConfig) -> str:
        d = asdict(cfg)
        return str(sorted(d.items(), key=lambda x: x[0]))

    current_cfg = deepcopy(start_cfg_dict)
    best = evaluate_cfg_dict(current_cfg, tag="start")
    if best.get("cached"):
        raise RuntimeError("Unexpected cache hit for initial config.")

    for round_idx in range(args.coord_rounds):
        for param, values in search_space.items():
            local_best = None
            local_best_cfg = None
            for v in values:
                cand = deepcopy(current_cfg)
                cand[param] = v
                # If domain override is disabled, keep domain knobs neutral.
                if param == "enable_domain_override" and not bool(v):
                    cand["domain_bonus"] = 0.0
                    cand["domain_conf_threshold"] = 1.0
                result = evaluate_cfg_dict(cand, tag=f"coord_r{round_idx+1}:{param}")
                if result.get("cached"):
                    continue
                if (local_best is None) or (result["score"]["chrf2"] > local_best["score"]["chrf2"]):
                    local_best = result
                    local_best_cfg = cand
            if local_best_cfg is not None:
                current_cfg = local_best_cfg
                if local_best["score"]["chrf2"] > best["score"]["chrf2"]:
                    best = local_best

    # Random refinement around the discovered region
    params_for_random = list(search_space.keys())
    for i in range(args.random_trials):
        cand = deepcopy(current_cfg)
        mutate_n = random.randint(2, min(5, len(params_for_random)))
        for p in random.sample(params_for_random, mutate_n):
            cand[p] = random.choice(search_space[p])
        if not cand.get("enable_domain_override", True):
            cand["domain_bonus"] = 0.0
            cand["domain_conf_threshold"] = 1.0
        result = evaluate_cfg_dict(cand, tag=f"random_{i+1:02d}")
        if result.get("cached"):
            continue
        if result["score"]["chrf2"] > best["score"]["chrf2"]:
            best = result
            current_cfg = deepcopy(cand)

    ranked = sorted(eval_history, key=lambda x: x["score"]["chrf2"], reverse=True)
    summary = {
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "memory_csv": str(args.memory_csv.resolve()),
        "canonical_no_external": canonical_no_ext,
        "start_config": start_cfg_dict,
        "search_space": search_space,
        "coord_rounds": int(args.coord_rounds),
        "random_trials": int(args.random_trials),
        "best": best,
        "top20": ranked[:20],
        "num_evaluations": len(eval_history),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, summary)

    print("")
    print(f"No-external baseline chrF2\t{canonical_no_ext['score']['chrf2']:.4f}")
    print(f"Best focused ORACC chrF2\t{best['score']['chrf2']:.4f}")
    print(f"Delta vs no-external\t{best['delta_vs_no_external']:.4f}")
    print(f"Evaluations\t{len(eval_history)}")
    print(f"Wrote\t{args.json_out}")


if __name__ == "__main__":
    main()
