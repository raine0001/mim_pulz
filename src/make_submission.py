from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.retrieval import (
    CANONICAL_BASELINE_V2,
    RetrievalConfig,
    make_retrieval_submission,
    make_routed_retrieval_submission,
    make_routed_reranked_retrieval_submission,
)
from mim_pulz.seq2seq_format import DEFAULT_TASK_PREFIX
from routing_engine import resolve_routing_map_path
from utils_manifest import file_sha256, write_json, write_text


CANONICAL_RERANKER_MODELS: dict[str, Path] = {
    "seed42": PATHS.root / "artifacts" / "models" / "routed_linear_reranker_seed42_pairwise_combined.json",
    "seed43": PATHS.root / "artifacts" / "models" / "routed_linear_reranker_seed43_pairwise_combined.json",
    "seed44": PATHS.root / "artifacts" / "models" / "routed_linear_reranker_seed44_pairwise_combined.json",
}


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def _cli_flag_present(*names: str) -> bool:
    argv = set(sys.argv[1:])
    return any(name in argv for name in names)


def _profile_config(memory_profile: str, repo_root: Path) -> RetrievalConfig:
    if memory_profile == "internal":
        return CANONICAL_BASELINE_V2
    if memory_profile == "oracc_best":
        return RetrievalConfig(
            analyzer="char_wb",
            ngram_min=3,
            ngram_max=5,
            max_features=300_000,
            top_k=100,
            len_weight=0.6,
            len_mode="ratio",
            strip_punct=False,
            lowercase=False,
            collapse_whitespace=True,
            stage2_type="bm25",
            stage2_pool=40,
            stage2_weight=0.35,
            stage2_bm25_k1=1.2,
            stage2_bm25_b=0.5,
            enable_domain_override=False,
            domain_candidate_top_k=120,
            domain_bonus=0.01,
            domain_conf_threshold=0.25,
            domain_margin=0.02,
            external_memory_paths=(str((repo_root / "data" / "processed" / "oracc_evacun_memory.csv").resolve()),),
            external_source_col="source",
            external_target_col="target",
            external_context_col="context",
            external_origin_col="origin",
            external_memory_limit=5000,
            external_context_allowlist=("legal", "letter"),
            external_candidate_cap=25,
            external_enable_fallback=False,
            external_internal_top_threshold=-1.0,
            external_internal_gap_threshold=-1.0,
            external_force_contexts=(),
            external_gate_bonus=0.0,
            external_gate_margin=0.0,
            competition_origin_bonus=0.01,
            external_origin_bonus=-0.05,
            enable_uncertainty_adaptation=True,
            uncertainty_high_threshold=0.03,
            uncertainty_closing_threshold=0.02,
            uncertainty_bracket_percentile=80.0,
            uncertainty_internal_top_threshold=0.48,
            uncertainty_internal_gap_threshold=0.01,
            uncertainty_topk_add=20,
            uncertainty_external_cap_add=2,
            uncertainty_topk_boost=0.0,
            uncertainty_external_bonus=0.0,
            uncertainty_stage2_discount=0.0,
            uncertainty_len_discount=0.0,
            uncertainty_triggered_len_ratio_min=0.65,
            uncertainty_skeleton_blend=0.05,
            uncertainty_number_bonus=0.005,
            uncertainty_formula_bonus=0.005,
            uncertainty_slot_bonus=0.0,
            uncertainty_variant_bonus=0.002,
            uncertainty_candidate_uncertain_penalty=0.0,
            enable_skeleton_retrieval_path=True,
            skeleton_candidate_top_k=120,
            evidence_k=3,
        )
    raise ValueError(f"Unknown memory profile: {memory_profile}")


def _pick_value(
    *,
    retrieval_cfg: dict,
    key: str,
    cli_value,
    profile_value,
    cli_flags: tuple[str, ...] = (),
):
    if key in retrieval_cfg:
        return retrieval_cfg[key]
    if cli_flags and _cli_flag_present(*cli_flags):
        return cli_value
    return profile_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic submission and save hash/manifests to artifacts."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["retrieval", "retrieval_routed", "retrieval_routed_reranked", "lora"],
        default="retrieval_routed_reranked",
    )
    parser.add_argument(
        "--memory",
        type=str,
        choices=["internal", "oracc_best"],
        default="oracc_best",
        help="Retrieval memory profile (used by retrieval and routed retrieval methods).",
    )
    parser.add_argument(
        "--routing-map",
        type=Path,
        default=None,
        help="Routing map JSON path for --method retrieval_routed and --method retrieval_routed_reranked.",
    )
    parser.add_argument(
        "--profiles-cache",
        type=Path,
        default=None,
        help="Optional cache for computed test-time structural profiles in retrieval_routed mode.",
    )
    parser.add_argument(
        "--routing-telemetry",
        type=Path,
        default=None,
        help="Optional routing telemetry output path (default: <submission>.routing.json).",
    )
    parser.add_argument(
        "--reranker-canonical",
        type=str,
        choices=sorted(CANONICAL_RERANKER_MODELS.keys()),
        default="seed43",
        help="Canonical reranker seed for --method retrieval_routed_reranked when --reranker-model is not explicitly provided.",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=CANONICAL_RERANKER_MODELS["seed43"],
        help="Linear reranker model JSON for --method retrieval_routed_reranked. Explicit path overrides --reranker-canonical.",
    )
    parser.add_argument(
        "--reranker-internal-top-k",
        type=int,
        default=120,
        help="Candidate pool size from internal memory for reranked routed mode.",
    )
    parser.add_argument(
        "--reranker-oracc-cap",
        type=int,
        default=25,
        help="Candidate cap from ORACC memory for reranked routed mode.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Artifacts run directory.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config to source settings.")
    parser.add_argument("--competition-dir", type=Path, default=PATHS.data_raw / "competition")
    parser.add_argument("--schema", type=Path, default=PATHS.root / "config" / "schema.json")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default="google/mt5-small")
    parser.add_argument("--input-max-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--task-prefix", type=str, default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--use-domain-tag", action="store_true")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--retrieval-top-k", type=int, default=CANONICAL_BASELINE_V2.top_k)
    parser.add_argument("--retrieval-len-weight", type=float, default=CANONICAL_BASELINE_V2.len_weight)
    parser.add_argument("--retrieval-len-mode", type=str, default=CANONICAL_BASELINE_V2.len_mode)
    parser.add_argument(
        "--retrieval-strip-punct",
        dest="retrieval_strip_punct",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-strip-punct",
        dest="retrieval_strip_punct",
        action="store_false",
    )
    parser.set_defaults(retrieval_strip_punct=CANONICAL_BASELINE_V2.strip_punct)
    parser.add_argument(
        "--retrieval-lowercase",
        dest="retrieval_lowercase",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-lowercase",
        dest="retrieval_lowercase",
        action="store_false",
    )
    parser.set_defaults(retrieval_lowercase=CANONICAL_BASELINE_V2.lowercase)
    parser.add_argument(
        "--retrieval-collapse-whitespace",
        dest="retrieval_collapse_whitespace",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-collapse-whitespace",
        dest="retrieval_collapse_whitespace",
        action="store_false",
    )
    parser.set_defaults(retrieval_collapse_whitespace=CANONICAL_BASELINE_V2.collapse_whitespace)
    parser.add_argument("--retrieval-stage2-type", type=str, default=CANONICAL_BASELINE_V2.stage2_type)
    parser.add_argument("--retrieval-stage2-pool", type=int, default=CANONICAL_BASELINE_V2.stage2_pool)
    parser.add_argument("--retrieval-stage2-weight", type=float, default=CANONICAL_BASELINE_V2.stage2_weight)
    parser.add_argument(
        "--retrieval-stage2-bm25-k1",
        type=float,
        default=CANONICAL_BASELINE_V2.stage2_bm25_k1,
    )
    parser.add_argument(
        "--retrieval-stage2-bm25-b",
        type=float,
        default=CANONICAL_BASELINE_V2.stage2_bm25_b,
    )
    parser.add_argument(
        "--retrieval-domain-candidate-top-k",
        type=int,
        default=CANONICAL_BASELINE_V2.domain_candidate_top_k,
    )
    parser.add_argument("--retrieval-domain-bonus", type=float, default=CANONICAL_BASELINE_V2.domain_bonus)
    parser.add_argument(
        "--retrieval-domain-conf-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.domain_conf_threshold,
    )
    parser.add_argument(
        "--retrieval-domain-margin",
        type=float,
        default=CANONICAL_BASELINE_V2.domain_margin,
    )
    parser.add_argument(
        "--retrieval-domain-override",
        dest="retrieval_domain_override",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-domain-override",
        dest="retrieval_domain_override",
        action="store_false",
    )
    parser.set_defaults(retrieval_domain_override=CANONICAL_BASELINE_V2.enable_domain_override)
    parser.add_argument(
        "--retrieval-enable-section-border",
        dest="retrieval_enable_section_border",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-enable-section-border",
        dest="retrieval_enable_section_border",
        action="store_false",
    )
    parser.set_defaults(retrieval_enable_section_border=CANONICAL_BASELINE_V2.enable_section_border)
    parser.add_argument(
        "--retrieval-section-force-closing",
        dest="retrieval_section_force_closing",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-section-force-closing",
        dest="retrieval_section_force_closing",
        action="store_false",
    )
    parser.set_defaults(retrieval_section_force_closing=CANONICAL_BASELINE_V2.section_force_closing)
    parser.add_argument(
        "--retrieval-section-closing-tail-ratio",
        type=float,
        default=CANONICAL_BASELINE_V2.section_closing_tail_ratio,
    )
    parser.add_argument(
        "--retrieval-section-force-min-score",
        type=float,
        default=CANONICAL_BASELINE_V2.section_force_min_score,
    )
    parser.add_argument(
        "--retrieval-section-min-pool",
        type=int,
        default=CANONICAL_BASELINE_V2.section_min_pool,
    )
    parser.add_argument(
        "--retrieval-section-match-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.section_match_bonus,
    )
    parser.add_argument(
        "--retrieval-enable-uncertainty-adaptation",
        dest="retrieval_enable_uncertainty_adaptation",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-enable-uncertainty-adaptation",
        dest="retrieval_enable_uncertainty_adaptation",
        action="store_false",
    )
    parser.set_defaults(
        retrieval_enable_uncertainty_adaptation=CANONICAL_BASELINE_V2.enable_uncertainty_adaptation
    )
    parser.add_argument(
        "--retrieval-uncertainty-high-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_high_threshold,
    )
    parser.add_argument(
        "--retrieval-uncertainty-closing-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_closing_threshold,
    )
    parser.add_argument(
        "--retrieval-uncertainty-bracket-percentile",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_bracket_percentile,
    )
    parser.add_argument(
        "--retrieval-uncertainty-internal-top-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_internal_top_threshold,
    )
    parser.add_argument(
        "--retrieval-uncertainty-internal-gap-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_internal_gap_threshold,
    )
    parser.add_argument(
        "--retrieval-uncertainty-topk-add",
        type=int,
        default=CANONICAL_BASELINE_V2.uncertainty_topk_add,
    )
    parser.add_argument(
        "--retrieval-uncertainty-external-cap-add",
        type=int,
        default=CANONICAL_BASELINE_V2.uncertainty_external_cap_add,
    )
    parser.add_argument(
        "--retrieval-uncertainty-topk-boost",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_topk_boost,
    )
    parser.add_argument(
        "--retrieval-uncertainty-external-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_external_bonus,
    )
    parser.add_argument(
        "--retrieval-uncertainty-stage2-discount",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_stage2_discount,
    )
    parser.add_argument(
        "--retrieval-uncertainty-len-discount",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_len_discount,
    )
    parser.add_argument(
        "--retrieval-uncertainty-triggered-len-ratio-min",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_triggered_len_ratio_min,
    )
    parser.add_argument(
        "--retrieval-uncertainty-skeleton-blend",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_skeleton_blend,
    )
    parser.add_argument(
        "--retrieval-uncertainty-number-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_number_bonus,
    )
    parser.add_argument(
        "--retrieval-uncertainty-formula-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_formula_bonus,
    )
    parser.add_argument(
        "--retrieval-uncertainty-slot-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_slot_bonus,
    )
    parser.add_argument(
        "--retrieval-uncertainty-variant-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_variant_bonus,
    )
    parser.add_argument(
        "--retrieval-uncertainty-candidate-uncertain-penalty",
        type=float,
        default=CANONICAL_BASELINE_V2.uncertainty_candidate_uncertain_penalty,
    )
    parser.add_argument(
        "--retrieval-enable-skeleton-retrieval-path",
        dest="retrieval_enable_skeleton_retrieval_path",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-enable-skeleton-retrieval-path",
        dest="retrieval_enable_skeleton_retrieval_path",
        action="store_false",
    )
    parser.set_defaults(
        retrieval_enable_skeleton_retrieval_path=CANONICAL_BASELINE_V2.enable_skeleton_retrieval_path
    )
    parser.add_argument(
        "--retrieval-skeleton-candidate-top-k",
        type=int,
        default=CANONICAL_BASELINE_V2.skeleton_candidate_top_k,
    )
    parser.add_argument(
        "--retrieval-uncertainty-rare-cutoff",
        type=int,
        default=CANONICAL_BASELINE_V2.uncertainty_rare_cutoff,
    )
    parser.add_argument(
        "--retrieval-memory-csv",
        dest="retrieval_memory_csv",
        action="append",
        default=None,
        help="Optional external retrieval memory table path (repeat for multiple files).",
    )
    parser.add_argument(
        "--retrieval-memory-source-col",
        type=str,
        default=CANONICAL_BASELINE_V2.external_source_col,
    )
    parser.add_argument(
        "--retrieval-memory-target-col",
        type=str,
        default=CANONICAL_BASELINE_V2.external_target_col,
    )
    parser.add_argument(
        "--retrieval-memory-context-col",
        type=str,
        default=CANONICAL_BASELINE_V2.external_context_col,
    )
    parser.add_argument(
        "--retrieval-memory-origin-col",
        type=str,
        default=CANONICAL_BASELINE_V2.external_origin_col,
    )
    parser.add_argument(
        "--retrieval-memory-limit",
        type=int,
        default=CANONICAL_BASELINE_V2.external_memory_limit,
    )
    parser.add_argument(
        "--retrieval-memory-context-allow",
        action="append",
        default=None,
        help="Optional allowlist for external memory contexts (repeatable).",
    )
    parser.add_argument(
        "--retrieval-external-candidate-cap",
        type=int,
        default=CANONICAL_BASELINE_V2.external_candidate_cap,
    )
    parser.add_argument(
        "--retrieval-external-enable-fallback",
        dest="retrieval_external_enable_fallback",
        action="store_true",
    )
    parser.add_argument(
        "--no-retrieval-external-enable-fallback",
        dest="retrieval_external_enable_fallback",
        action="store_false",
    )
    parser.set_defaults(retrieval_external_enable_fallback=CANONICAL_BASELINE_V2.external_enable_fallback)
    parser.add_argument(
        "--retrieval-external-internal-top-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.external_internal_top_threshold,
    )
    parser.add_argument(
        "--retrieval-external-internal-gap-threshold",
        type=float,
        default=CANONICAL_BASELINE_V2.external_internal_gap_threshold,
    )
    parser.add_argument(
        "--retrieval-external-force-context",
        action="append",
        default=None,
        help="Force-enable ORACC fallback for these contexts (repeatable).",
    )
    parser.add_argument(
        "--retrieval-external-gate-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.external_gate_bonus,
    )
    parser.add_argument(
        "--retrieval-external-gate-margin",
        type=float,
        default=CANONICAL_BASELINE_V2.external_gate_margin,
    )
    parser.add_argument(
        "--retrieval-competition-origin-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.competition_origin_bonus,
    )
    parser.add_argument(
        "--retrieval-external-origin-bonus",
        type=float,
        default=CANONICAL_BASELINE_V2.external_origin_bonus,
    )
    parser.add_argument(
        "--retrieval-evidence-k",
        type=int,
        default=CANONICAL_BASELINE_V2.evidence_k,
    )
    parser.add_argument("--verify-determinism", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = PATHS.root

    cfg = {}
    if args.config is not None:
        cfg_path = _resolve(args.config, repo_root)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    run_dir = _resolve(args.run_dir, repo_root) if args.run_dir is not None else None
    adapter_dir = _resolve(args.adapter_dir, repo_root) if args.adapter_dir is not None else None
    tokenizer_dir = _resolve(args.tokenizer_dir, repo_root) if args.tokenizer_dir is not None else None

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.method == "lora":
            if adapter_dir is None:
                cand = run_dir / "adapter"
                if cand.exists():
                    adapter_dir = cand
            if tokenizer_dir is None:
                cand = run_dir / "tokenizer"
                if cand.exists():
                    tokenizer_dir = cand

    if args.method == "lora" and adapter_dir is None:
        from mim_pulz.lora_submission import resolve_default_adapter_dir

        adapter_dir = resolve_default_adapter_dir()

    output_csv = _resolve(args.output, repo_root) if args.output is not None else None
    if output_csv is None:
        if run_dir is not None:
            output_csv = run_dir / "submission.csv"
        else:
            output_csv = PATHS.outputs / "submission.csv"

    model_cfg = cfg.get("model", {})
    gen_cfg = cfg.get("generation", {})
    fmt_cfg = cfg.get("formatting", {})
    retrieval_cfg = cfg.get("retrieval", {})
    seed = int(cfg.get("seed", args.seed))

    result = None
    method_meta = {"method": args.method}
    if args.method in {"retrieval", "retrieval_routed", "retrieval_routed_reranked"}:
        profile_name = str(retrieval_cfg.get("memory_profile", args.memory))
        profile_cfg = _profile_config(profile_name, repo_root)

        if "ngram_range" in retrieval_cfg:
            ngram_range = retrieval_cfg["ngram_range"]
            ngram_min = int(ngram_range[0])
            ngram_max = int(ngram_range[1])
        else:
            ngram_min = int(retrieval_cfg.get("ngram_min", profile_cfg.ngram_min))
            ngram_max = int(retrieval_cfg.get("ngram_max", profile_cfg.ngram_max))

        raw_memory_paths = retrieval_cfg.get("external_memory_paths", retrieval_cfg.get("memory_paths", None))
        if raw_memory_paths is None:
            if _cli_flag_present("--retrieval-memory-csv"):
                raw_memory_paths = args.retrieval_memory_csv or []
            else:
                raw_memory_paths = list(profile_cfg.external_memory_paths)
        if isinstance(raw_memory_paths, str):
            raw_memory_paths = [raw_memory_paths]
        memory_paths = tuple(str(_resolve(Path(p), repo_root)) for p in raw_memory_paths)

        raw_ctx_allow = retrieval_cfg.get("external_context_allowlist", None)
        if raw_ctx_allow is None:
            if _cli_flag_present("--retrieval-memory-context-allow"):
                raw_ctx_allow = args.retrieval_memory_context_allow or []
            else:
                raw_ctx_allow = list(profile_cfg.external_context_allowlist)
        if isinstance(raw_ctx_allow, str):
            raw_ctx_allow = [raw_ctx_allow]
        ctx_allow = tuple(str(x) for x in raw_ctx_allow)

        raw_force_ctx = retrieval_cfg.get("external_force_contexts", None)
        if raw_force_ctx is None:
            if _cli_flag_present("--retrieval-external-force-context"):
                raw_force_ctx = args.retrieval_external_force_context or []
            else:
                raw_force_ctx = list(profile_cfg.external_force_contexts)
        if isinstance(raw_force_ctx, str):
            raw_force_ctx = [raw_force_ctx]
        force_ctx = tuple(str(x) for x in raw_force_ctx)

        rcfg = RetrievalConfig(
            analyzer=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="analyzer",
                    cli_value=None,
                    profile_value=profile_cfg.analyzer,
                )
            ),
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            max_features=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="max_features",
                    cli_value=None,
                    profile_value=profile_cfg.max_features,
                )
            ),
            top_k=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="top_k",
                    cli_value=args.retrieval_top_k,
                    profile_value=profile_cfg.top_k,
                    cli_flags=("--retrieval-top-k",),
                )
            ),
            len_weight=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="len_weight",
                    cli_value=args.retrieval_len_weight,
                    profile_value=profile_cfg.len_weight,
                    cli_flags=("--retrieval-len-weight",),
                )
            ),
            len_mode=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="len_mode",
                    cli_value=args.retrieval_len_mode,
                    profile_value=profile_cfg.len_mode,
                    cli_flags=("--retrieval-len-mode",),
                )
            ),
            strip_punct=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="strip_punct",
                    cli_value=args.retrieval_strip_punct,
                    profile_value=profile_cfg.strip_punct,
                    cli_flags=("--retrieval-strip-punct", "--no-retrieval-strip-punct"),
                )
            ),
            lowercase=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="lowercase",
                    cli_value=args.retrieval_lowercase,
                    profile_value=profile_cfg.lowercase,
                    cli_flags=("--retrieval-lowercase", "--no-retrieval-lowercase"),
                )
            ),
            collapse_whitespace=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="collapse_whitespace",
                    cli_value=args.retrieval_collapse_whitespace,
                    profile_value=profile_cfg.collapse_whitespace,
                    cli_flags=(
                        "--retrieval-collapse-whitespace",
                        "--no-retrieval-collapse-whitespace",
                    ),
                )
            ),
            stage2_type=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="stage2_type",
                    cli_value=args.retrieval_stage2_type,
                    profile_value=profile_cfg.stage2_type,
                    cli_flags=("--retrieval-stage2-type",),
                )
            ),
            stage2_pool=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="stage2_pool",
                    cli_value=args.retrieval_stage2_pool,
                    profile_value=profile_cfg.stage2_pool,
                    cli_flags=("--retrieval-stage2-pool",),
                )
            ),
            stage2_weight=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="stage2_weight",
                    cli_value=args.retrieval_stage2_weight,
                    profile_value=profile_cfg.stage2_weight,
                    cli_flags=("--retrieval-stage2-weight",),
                )
            ),
            stage2_bm25_k1=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="stage2_bm25_k1",
                    cli_value=args.retrieval_stage2_bm25_k1,
                    profile_value=profile_cfg.stage2_bm25_k1,
                    cli_flags=("--retrieval-stage2-bm25-k1",),
                )
            ),
            stage2_bm25_b=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="stage2_bm25_b",
                    cli_value=args.retrieval_stage2_bm25_b,
                    profile_value=profile_cfg.stage2_bm25_b,
                    cli_flags=("--retrieval-stage2-bm25-b",),
                )
            ),
            enable_domain_override=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="enable_domain_override",
                    cli_value=args.retrieval_domain_override,
                    profile_value=profile_cfg.enable_domain_override,
                    cli_flags=("--retrieval-domain-override", "--no-retrieval-domain-override"),
                )
            ),
            domain_candidate_top_k=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="domain_candidate_top_k",
                    cli_value=args.retrieval_domain_candidate_top_k,
                    profile_value=profile_cfg.domain_candidate_top_k,
                    cli_flags=("--retrieval-domain-candidate-top-k",),
                )
            ),
            domain_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="domain_bonus",
                    cli_value=args.retrieval_domain_bonus,
                    profile_value=profile_cfg.domain_bonus,
                    cli_flags=("--retrieval-domain-bonus",),
                )
            ),
            domain_conf_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="domain_conf_threshold",
                    cli_value=args.retrieval_domain_conf_threshold,
                    profile_value=profile_cfg.domain_conf_threshold,
                    cli_flags=("--retrieval-domain-conf-threshold",),
                )
            ),
            domain_margin=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="domain_margin",
                    cli_value=args.retrieval_domain_margin,
                    profile_value=profile_cfg.domain_margin,
                    cli_flags=("--retrieval-domain-margin",),
                )
            ),
            enable_section_border=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="enable_section_border",
                    cli_value=args.retrieval_enable_section_border,
                    profile_value=profile_cfg.enable_section_border,
                    cli_flags=(
                        "--retrieval-enable-section-border",
                        "--no-retrieval-enable-section-border",
                    ),
                )
            ),
            section_force_closing=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="section_force_closing",
                    cli_value=args.retrieval_section_force_closing,
                    profile_value=profile_cfg.section_force_closing,
                    cli_flags=(
                        "--retrieval-section-force-closing",
                        "--no-retrieval-section-force-closing",
                    ),
                )
            ),
            section_force_min_score=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="section_force_min_score",
                    cli_value=args.retrieval_section_force_min_score,
                    profile_value=profile_cfg.section_force_min_score,
                    cli_flags=("--retrieval-section-force-min-score",),
                )
            ),
            section_closing_tail_ratio=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="section_closing_tail_ratio",
                    cli_value=args.retrieval_section_closing_tail_ratio,
                    profile_value=profile_cfg.section_closing_tail_ratio,
                    cli_flags=("--retrieval-section-closing-tail-ratio",),
                )
            ),
            section_min_pool=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="section_min_pool",
                    cli_value=args.retrieval_section_min_pool,
                    profile_value=profile_cfg.section_min_pool,
                    cli_flags=("--retrieval-section-min-pool",),
                )
            ),
            section_match_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="section_match_bonus",
                    cli_value=args.retrieval_section_match_bonus,
                    profile_value=profile_cfg.section_match_bonus,
                    cli_flags=("--retrieval-section-match-bonus",),
                )
            ),
            enable_uncertainty_adaptation=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="enable_uncertainty_adaptation",
                    cli_value=args.retrieval_enable_uncertainty_adaptation,
                    profile_value=profile_cfg.enable_uncertainty_adaptation,
                    cli_flags=(
                        "--retrieval-enable-uncertainty-adaptation",
                        "--no-retrieval-enable-uncertainty-adaptation",
                    ),
                )
            ),
            uncertainty_high_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_high_threshold",
                    cli_value=args.retrieval_uncertainty_high_threshold,
                    profile_value=profile_cfg.uncertainty_high_threshold,
                    cli_flags=("--retrieval-uncertainty-high-threshold",),
                )
            ),
            uncertainty_closing_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_closing_threshold",
                    cli_value=args.retrieval_uncertainty_closing_threshold,
                    profile_value=profile_cfg.uncertainty_closing_threshold,
                    cli_flags=("--retrieval-uncertainty-closing-threshold",),
                )
            ),
            uncertainty_bracket_percentile=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_bracket_percentile",
                    cli_value=args.retrieval_uncertainty_bracket_percentile,
                    profile_value=profile_cfg.uncertainty_bracket_percentile,
                    cli_flags=("--retrieval-uncertainty-bracket-percentile",),
                )
            ),
            uncertainty_internal_top_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_internal_top_threshold",
                    cli_value=args.retrieval_uncertainty_internal_top_threshold,
                    profile_value=profile_cfg.uncertainty_internal_top_threshold,
                    cli_flags=("--retrieval-uncertainty-internal-top-threshold",),
                )
            ),
            uncertainty_internal_gap_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_internal_gap_threshold",
                    cli_value=args.retrieval_uncertainty_internal_gap_threshold,
                    profile_value=profile_cfg.uncertainty_internal_gap_threshold,
                    cli_flags=("--retrieval-uncertainty-internal-gap-threshold",),
                )
            ),
            uncertainty_topk_add=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_topk_add",
                    cli_value=args.retrieval_uncertainty_topk_add,
                    profile_value=profile_cfg.uncertainty_topk_add,
                    cli_flags=("--retrieval-uncertainty-topk-add",),
                )
            ),
            uncertainty_external_cap_add=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_external_cap_add",
                    cli_value=args.retrieval_uncertainty_external_cap_add,
                    profile_value=profile_cfg.uncertainty_external_cap_add,
                    cli_flags=("--retrieval-uncertainty-external-cap-add",),
                )
            ),
            uncertainty_topk_boost=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_topk_boost",
                    cli_value=args.retrieval_uncertainty_topk_boost,
                    profile_value=profile_cfg.uncertainty_topk_boost,
                    cli_flags=("--retrieval-uncertainty-topk-boost",),
                )
            ),
            uncertainty_external_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_external_bonus",
                    cli_value=args.retrieval_uncertainty_external_bonus,
                    profile_value=profile_cfg.uncertainty_external_bonus,
                    cli_flags=("--retrieval-uncertainty-external-bonus",),
                )
            ),
            uncertainty_stage2_discount=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_stage2_discount",
                    cli_value=args.retrieval_uncertainty_stage2_discount,
                    profile_value=profile_cfg.uncertainty_stage2_discount,
                    cli_flags=("--retrieval-uncertainty-stage2-discount",),
                )
            ),
            uncertainty_len_discount=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_len_discount",
                    cli_value=args.retrieval_uncertainty_len_discount,
                    profile_value=profile_cfg.uncertainty_len_discount,
                    cli_flags=("--retrieval-uncertainty-len-discount",),
                )
            ),
            uncertainty_triggered_len_ratio_min=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_triggered_len_ratio_min",
                    cli_value=args.retrieval_uncertainty_triggered_len_ratio_min,
                    profile_value=profile_cfg.uncertainty_triggered_len_ratio_min,
                    cli_flags=("--retrieval-uncertainty-triggered-len-ratio-min",),
                )
            ),
            uncertainty_skeleton_blend=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_skeleton_blend",
                    cli_value=args.retrieval_uncertainty_skeleton_blend,
                    profile_value=profile_cfg.uncertainty_skeleton_blend,
                    cli_flags=("--retrieval-uncertainty-skeleton-blend",),
                )
            ),
            uncertainty_number_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_number_bonus",
                    cli_value=args.retrieval_uncertainty_number_bonus,
                    profile_value=profile_cfg.uncertainty_number_bonus,
                    cli_flags=("--retrieval-uncertainty-number-bonus",),
                )
            ),
            uncertainty_formula_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_formula_bonus",
                    cli_value=args.retrieval_uncertainty_formula_bonus,
                    profile_value=profile_cfg.uncertainty_formula_bonus,
                    cli_flags=("--retrieval-uncertainty-formula-bonus",),
                )
            ),
            uncertainty_slot_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_slot_bonus",
                    cli_value=args.retrieval_uncertainty_slot_bonus,
                    profile_value=profile_cfg.uncertainty_slot_bonus,
                    cli_flags=("--retrieval-uncertainty-slot-bonus",),
                )
            ),
            uncertainty_variant_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_variant_bonus",
                    cli_value=args.retrieval_uncertainty_variant_bonus,
                    profile_value=profile_cfg.uncertainty_variant_bonus,
                    cli_flags=("--retrieval-uncertainty-variant-bonus",),
                )
            ),
            uncertainty_candidate_uncertain_penalty=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_candidate_uncertain_penalty",
                    cli_value=args.retrieval_uncertainty_candidate_uncertain_penalty,
                    profile_value=profile_cfg.uncertainty_candidate_uncertain_penalty,
                    cli_flags=("--retrieval-uncertainty-candidate-uncertain-penalty",),
                )
            ),
            enable_skeleton_retrieval_path=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="enable_skeleton_retrieval_path",
                    cli_value=args.retrieval_enable_skeleton_retrieval_path,
                    profile_value=profile_cfg.enable_skeleton_retrieval_path,
                    cli_flags=(
                        "--retrieval-enable-skeleton-retrieval-path",
                        "--no-retrieval-enable-skeleton-retrieval-path",
                    ),
                )
            ),
            skeleton_candidate_top_k=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="skeleton_candidate_top_k",
                    cli_value=args.retrieval_skeleton_candidate_top_k,
                    profile_value=profile_cfg.skeleton_candidate_top_k,
                    cli_flags=("--retrieval-skeleton-candidate-top-k",),
                )
            ),
            uncertainty_rare_cutoff=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="uncertainty_rare_cutoff",
                    cli_value=args.retrieval_uncertainty_rare_cutoff,
                    profile_value=profile_cfg.uncertainty_rare_cutoff,
                    cli_flags=("--retrieval-uncertainty-rare-cutoff",),
                )
            ),
            external_memory_paths=memory_paths,
            external_source_col=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_source_col",
                    cli_value=args.retrieval_memory_source_col,
                    profile_value=profile_cfg.external_source_col,
                    cli_flags=("--retrieval-memory-source-col",),
                )
            ),
            external_target_col=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_target_col",
                    cli_value=args.retrieval_memory_target_col,
                    profile_value=profile_cfg.external_target_col,
                    cli_flags=("--retrieval-memory-target-col",),
                )
            ),
            external_context_col=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_context_col",
                    cli_value=args.retrieval_memory_context_col,
                    profile_value=profile_cfg.external_context_col,
                    cli_flags=("--retrieval-memory-context-col",),
                )
            ),
            external_origin_col=str(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_origin_col",
                    cli_value=args.retrieval_memory_origin_col,
                    profile_value=profile_cfg.external_origin_col,
                    cli_flags=("--retrieval-memory-origin-col",),
                )
            ),
            external_memory_limit=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_memory_limit",
                    cli_value=args.retrieval_memory_limit,
                    profile_value=profile_cfg.external_memory_limit,
                    cli_flags=("--retrieval-memory-limit",),
                )
            ),
            external_context_allowlist=ctx_allow,
            external_candidate_cap=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_candidate_cap",
                    cli_value=args.retrieval_external_candidate_cap,
                    profile_value=profile_cfg.external_candidate_cap,
                    cli_flags=("--retrieval-external-candidate-cap",),
                )
            ),
            external_enable_fallback=bool(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_enable_fallback",
                    cli_value=args.retrieval_external_enable_fallback,
                    profile_value=profile_cfg.external_enable_fallback,
                    cli_flags=(
                        "--retrieval-external-enable-fallback",
                        "--no-retrieval-external-enable-fallback",
                    ),
                )
            ),
            external_internal_top_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_internal_top_threshold",
                    cli_value=args.retrieval_external_internal_top_threshold,
                    profile_value=profile_cfg.external_internal_top_threshold,
                    cli_flags=("--retrieval-external-internal-top-threshold",),
                )
            ),
            external_internal_gap_threshold=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_internal_gap_threshold",
                    cli_value=args.retrieval_external_internal_gap_threshold,
                    profile_value=profile_cfg.external_internal_gap_threshold,
                    cli_flags=("--retrieval-external-internal-gap-threshold",),
                )
            ),
            external_force_contexts=force_ctx,
            external_gate_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_gate_bonus",
                    cli_value=args.retrieval_external_gate_bonus,
                    profile_value=profile_cfg.external_gate_bonus,
                    cli_flags=("--retrieval-external-gate-bonus",),
                )
            ),
            external_gate_margin=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_gate_margin",
                    cli_value=args.retrieval_external_gate_margin,
                    profile_value=profile_cfg.external_gate_margin,
                    cli_flags=("--retrieval-external-gate-margin",),
                )
            ),
            competition_origin_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="competition_origin_bonus",
                    cli_value=args.retrieval_competition_origin_bonus,
                    profile_value=profile_cfg.competition_origin_bonus,
                    cli_flags=("--retrieval-competition-origin-bonus",),
                )
            ),
            external_origin_bonus=float(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="external_origin_bonus",
                    cli_value=args.retrieval_external_origin_bonus,
                    profile_value=profile_cfg.external_origin_bonus,
                    cli_flags=("--retrieval-external-origin-bonus",),
                )
            ),
            evidence_k=int(
                _pick_value(
                    retrieval_cfg=retrieval_cfg,
                    key="evidence_k",
                    cli_value=args.retrieval_evidence_k,
                    profile_value=profile_cfg.evidence_k,
                    cli_flags=("--retrieval-evidence-k",),
                )
            ),
        )
        if args.method == "retrieval":
            result = make_retrieval_submission(
                output_csv=output_csv,
                competition_dir=_resolve(args.competition_dir, repo_root),
                schema_path=_resolve(args.schema, repo_root),
                config=rcfg,
                verify_determinism=args.verify_determinism,
            )
        elif args.method == "retrieval_routed":
            routing_map_path = resolve_routing_map_path(args.routing_map, repo_root)
            profiles_cache = (
                _resolve(args.profiles_cache, repo_root) if args.profiles_cache is not None else None
            )
            routing_telemetry = (
                _resolve(args.routing_telemetry, repo_root) if args.routing_telemetry is not None else None
            )
            result = make_routed_retrieval_submission(
                output_csv=output_csv,
                competition_dir=_resolve(args.competition_dir, repo_root),
                schema_path=_resolve(args.schema, repo_root),
                base_config=rcfg,
                routing_map_path=routing_map_path,
                profiles_cache_path=profiles_cache,
                routing_telemetry_path=routing_telemetry,
                verify_determinism=args.verify_determinism,
            )
        else:
            routing_map_path = resolve_routing_map_path(args.routing_map, repo_root)
            profiles_cache = (
                _resolve(args.profiles_cache, repo_root) if args.profiles_cache is not None else None
            )
            routing_telemetry = (
                _resolve(args.routing_telemetry, repo_root) if args.routing_telemetry is not None else None
            )
            if _cli_flag_present("--reranker-model"):
                reranker_model = _resolve(args.reranker_model, repo_root)
                reranker_source = "explicit"
            else:
                reranker_model = _resolve(CANONICAL_RERANKER_MODELS[str(args.reranker_canonical)], repo_root)
                reranker_source = f"canonical:{args.reranker_canonical}"
            result = make_routed_reranked_retrieval_submission(
                output_csv=output_csv,
                competition_dir=_resolve(args.competition_dir, repo_root),
                schema_path=_resolve(args.schema, repo_root),
                base_config=rcfg,
                routing_map_path=routing_map_path,
                reranker_model_path=reranker_model,
                reranker_internal_top_k=int(args.reranker_internal_top_k),
                reranker_oracc_cap=int(args.reranker_oracc_cap),
                profiles_cache_path=profiles_cache,
                routing_telemetry_path=routing_telemetry,
                verify_determinism=args.verify_determinism,
            )
        method_meta["retrieval_config"] = rcfg.__dict__
        method_meta["model_id"] = result.model_id
        method_meta["memory_profile"] = profile_name
        method_meta["evidence_json"] = (
            str(result.evidence_json.resolve()) if result.evidence_json is not None else None
        )
        if args.method in {"retrieval_routed", "retrieval_routed_reranked"}:
            method_meta["routing_map_path"] = (
                str(result.routing_map_path.resolve()) if result.routing_map_path is not None else None
            )
            method_meta["routing_map_sha256"] = result.routing_map_sha256
            method_meta["routing_json"] = (
                str(result.routing_json.resolve()) if result.routing_json is not None else None
            )
            method_meta["route_counts"] = result.route_counts
            method_meta["policy_model_ids"] = result.policy_model_ids
            if result.routing_map_path is not None:
                method_meta["routing_map_file_sha256"] = file_sha256(result.routing_map_path)
        if args.method == "retrieval_routed_reranked":
            method_meta["reranker_model_path"] = (
                str(result.reranker_model_path.resolve()) if result.reranker_model_path is not None else None
            )
            method_meta["reranker_model_sha256"] = result.reranker_model_sha256
            method_meta["reranker_internal_top_k"] = int(args.reranker_internal_top_k)
            method_meta["reranker_oracc_cap"] = int(args.reranker_oracc_cap)
            method_meta["reranker_canonical"] = str(args.reranker_canonical)
            method_meta["reranker_model_source"] = reranker_source
    else:
        from mim_pulz.lora_submission import InferenceConfig, make_lora_submission

        raw_max_length = gen_cfg.get("max_length", args.max_length)
        max_length = None if raw_max_length is None else int(raw_max_length)

        if "max_new_tokens" in gen_cfg:
            max_new_tokens = int(gen_cfg["max_new_tokens"])
        else:
            max_new_tokens = int(args.max_new_tokens)

        infer = InferenceConfig(
            base_model=str(model_cfg.get("base_model", args.base_model)),
            input_max_length=int(model_cfg.get("max_source_length", args.input_max_length)),
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            num_beams=int(gen_cfg.get("num_beams", args.num_beams)),
            no_repeat_ngram_size=int(gen_cfg.get("no_repeat_ngram_size", args.no_repeat_ngram_size)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", args.repetition_penalty)),
            early_stopping=bool(gen_cfg.get("early_stopping", args.early_stopping)),
            task_prefix=str(fmt_cfg.get("task_prefix", args.task_prefix)),
            use_domain_tag=bool(fmt_cfg.get("use_domain_tag", args.use_domain_tag)),
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
        )

        result = make_lora_submission(
            output_csv=output_csv,
            competition_dir=_resolve(args.competition_dir, repo_root),
            schema_path=_resolve(args.schema, repo_root),
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            config=infer,
            verify_determinism=args.verify_determinism,
        )
        method_meta["adapter_dir"] = str(adapter_dir.resolve())
        method_meta["tokenizer_dir"] = str(tokenizer_dir.resolve()) if tokenizer_dir else None

    sha_file = output_csv.with_suffix(output_csv.suffix + ".sha256")
    write_text(sha_file, f"{result.output_sha256}\n")

    manifest = {
        "output_csv": str(result.output_csv.resolve()),
        "rows": result.rows,
        "output_sha256": result.output_sha256,
        "metadata_json": str(result.metadata_json.resolve()),
        "sha_file": str(sha_file.resolve()),
        "determinism_verified": bool(args.verify_determinism),
        "method": args.method,
        "method_meta": method_meta,
    }
    if args.method == "lora":
        manifest["adapter_dir"] = str(adapter_dir.resolve())
        manifest["tokenizer_dir"] = str(tokenizer_dir.resolve()) if tokenizer_dir else None
    if args.method in {"retrieval", "retrieval_routed", "retrieval_routed_reranked"} and getattr(result, "evidence_json", None) is not None:
        manifest["evidence_json"] = str(result.evidence_json.resolve())
    if args.method in {"retrieval_routed", "retrieval_routed_reranked"}:
        if getattr(result, "routing_json", None) is not None:
            manifest["routing_json"] = str(result.routing_json.resolve())
        if getattr(result, "routing_map_path", None) is not None:
            manifest["routing_map_path"] = str(result.routing_map_path.resolve())
            manifest["routing_map_sha256"] = result.routing_map_sha256
            manifest["routing_map_file_sha256"] = file_sha256(result.routing_map_path)
        if getattr(result, "route_counts", None) is not None:
            manifest["route_counts"] = result.route_counts
        if getattr(result, "policy_model_ids", None) is not None:
            manifest["policy_model_ids"] = result.policy_model_ids
    if args.method == "retrieval_routed_reranked":
        if getattr(result, "reranker_model_path", None) is not None:
            manifest["reranker_model_path"] = str(result.reranker_model_path.resolve())
        if getattr(result, "reranker_model_sha256", None) is not None:
            manifest["reranker_model_sha256"] = result.reranker_model_sha256
        manifest["reranker_canonical"] = str(args.reranker_canonical)
    manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    write_json(manifest_path, manifest)

    print(f"Submission CSV: {result.output_csv}")
    print(f"Rows: {result.rows}")
    print(f"SHA256: {result.output_sha256}")
    print(f"SHA file: {sha_file}")
    print(f"Metadata: {result.metadata_json}")
    if args.method in {"retrieval_routed", "retrieval_routed_reranked"}:
        if result.routing_map_sha256 is not None:
            print(f"Routing map SHA256: {result.routing_map_sha256}")
        if result.route_counts is not None:
            print(f"Route counts: {result.route_counts}")
        if result.routing_json is not None:
            print(f"Routing telemetry: {result.routing_json}")
    if args.method == "retrieval_routed_reranked":
        print(f"Reranker canonical: {args.reranker_canonical}")
        if result.reranker_model_sha256 is not None:
            print(f"Reranker model SHA256: {result.reranker_model_sha256}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

