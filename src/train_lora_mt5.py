from __future__ import annotations

from dataclasses import asdict
import argparse
import os
from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import sacrebleu
from sacrebleu.metrics import CHRF
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import yaml


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_models import evaluate_models, spam_signature
from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.domain_intent import infer_dialog_domain_batch
from mim_pulz.lora_submission import (
    InferenceConfig,
    LoadedBundle,
    generate_predictions,
    make_lora_submission,
    set_deterministic,
)
from mim_pulz.seq2seq_format import DEFAULT_TASK_PREFIX, clean_target_batch, format_source_batch
from utils_manifest import (
    bytes_sha256,
    command_string,
    copy_file,
    file_sha256,
    git_short_hash,
    new_run_id,
    now_utc_iso,
    write_json,
    write_text,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Config-driven, reproducible LoRA retraining with eval gate and run manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PATHS.root / "configs" / "lora_mt5_small.yaml",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--no-enforce-gate",
        action="store_true",
        help="Do not fail process exit code when gate conditions are not met.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run micro-overfit preflight variants and exit.",
    )
    parser.add_argument(
        "--micro-diagnostic",
        action="store_true",
        help="Run hard micro-overfit diagnostic mode (exact-match/edit-distance/len metrics) and exit.",
    )
    return parser


def _build_bad_words_ids(tokenizer) -> list[list[int]]:
    out = []
    for i in range(100):
        ids = tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False)
        if len(ids) == 1:
            out.append([int(ids[0])])
    return out


def _split_idx(n: int, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_size = int(n * val_frac)
    return idx[val_size:], idx[:val_size]


def _resolve_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def _optional_int(value: object, default: int | None) -> int | None:
    if value is None:
        return default
    return int(value)


def _token_length_profile(tokenizer, texts: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
    )
    lens = np.asarray([len(x) for x in enc["input_ids"]], dtype=np.int32)
    truncated = lens >= max_length
    return lens, truncated


def _build_keep_mask(policy: str, src_trunc: np.ndarray, tgt_trunc: np.ndarray) -> np.ndarray:
    p = policy.lower().strip()
    if p == "keep":
        return np.ones_like(src_trunc, dtype=bool)
    if p == "drop_any":
        return ~(src_trunc | tgt_trunc)
    if p == "drop_target":
        return ~tgt_trunc
    if p == "drop_source":
        return ~src_trunc
    raise ValueError(f"Unknown truncation_policy: {policy}. Use keep|drop_any|drop_target|drop_source.")


def _data_fingerprint(
    competition_dir: Path,
    schema_path: Path,
    train_file: str,
    test_file: str,
    sample_sub_file: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    source_col: str,
    target_col: str,
) -> dict:
    train_path = competition_dir / train_file
    test_path = competition_dir / test_file
    sub_path = competition_dir / sample_sub_file
    return {
        "competition_dir": str(competition_dir.resolve()),
        "schema_path": str(schema_path.resolve()),
        "train_file_sha256": file_sha256(train_path),
        "test_file_sha256": file_sha256(test_path),
        "sample_submission_sha256": file_sha256(sub_path),
        "rows_total": int(len(train_idx) + len(val_idx)),
        "rows_train": int(len(train_idx)),
        "rows_val": int(len(val_idx)),
        "columns_used": {
            "source_col": source_col,
            "target_col": target_col,
        },
        "split": {
            "train_idx_sha256": bytes_sha256(np.asarray(train_idx, dtype=np.int64).tobytes()),
            "val_idx_sha256": bytes_sha256(np.asarray(val_idx, dtype=np.int64).tobytes()),
        },
    }


def _forward_seq2seq_loss(
    model: torch.nn.Module,
    tokenizer,
    source_texts: list[str],
    target_texts: list[str],
    max_src_len: int,
    max_tgt_len: int,
    device: str,
) -> torch.Tensor:
    encoded = tokenizer(
        source_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_src_len,
    )
    labels = tokenizer(
        text_target=target_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tgt_len,
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = labels.to(device)
    out = model(**encoded, labels=labels)
    return out.loss


def _set_dropout_zero(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _text_pair_metrics(preds: list[str], targets: list[str]) -> dict:
    n = len(preds)
    if n == 0:
        return {
            "exact_match_pct": 0.0,
            "avg_char_edit_distance": 0.0,
            "avg_output_length": 0.0,
            "avg_target_length": 0.0,
            "avg_length_ratio_output_to_target": 0.0,
        }
    exact = 0
    dists = []
    out_lens = []
    tgt_lens = []
    for p, t in zip(preds, targets):
        pp = clean_target_batch([p])[0]
        tt = clean_target_batch([t])[0]
        if pp == tt:
            exact += 1
        dists.append(float(_levenshtein_distance(pp, tt)))
        out_lens.append(float(len(pp)))
        tgt_lens.append(float(len(tt)))
    avg_out = float(np.mean(out_lens))
    avg_tgt = float(np.mean(tgt_lens))
    ratio = float(avg_out / avg_tgt) if avg_tgt > 0 else 0.0
    return {
        "exact_match_pct": float(100.0 * exact / n),
        "avg_char_edit_distance": float(np.mean(dists)),
        "avg_output_length": avg_out,
        "avg_target_length": avg_tgt,
        "avg_length_ratio_output_to_target": ratio,
    }


def _chrf2_consistency(preds: list[str], targets: list[str]) -> dict:
    metric = CHRF(word_order=2)
    score_metric = float(metric.corpus_score(preds, [targets]).score)
    score_ref = float(sacrebleu.corpus_chrf(preds, [targets], word_order=2).score)
    return {
        "chrf2_metric": score_metric,
        "chrf2_reference": score_ref,
        "abs_diff": float(abs(score_metric - score_ref)),
    }


def _unicode_audit(texts: list[str], top_n: int = 20) -> dict:
    changed: list[dict] = []
    non_ascii_count = 0
    combining_count = 0
    for i, text in enumerate(texts):
        s = str(text or "")
        for ch in s:
            if ord(ch) > 127:
                non_ascii_count += 1
            if unicodedata.combining(ch):
                combining_count += 1
        nfc = unicodedata.normalize("NFC", s)
        if nfc != s and len(changed) < top_n:
            changed.append({"index": i, "raw": s, "nfc": nfc})
    changed_count = sum(1 for t in texts if unicodedata.normalize("NFC", str(t or "")) != str(t or ""))
    return {
        "rows": len(texts),
        "non_ascii_char_count": int(non_ascii_count),
        "combining_mark_count": int(combining_count),
        "nfc_changed_rows": int(changed_count),
        "nfc_changed_ratio": float((changed_count / len(texts)) if texts else 0.0),
        "changed_examples": changed,
    }


def _collect_eval_snapshot(
    *,
    step: int,
    loss: float,
    micro_sources: list[str],
    micro_targets: list[str],
    model: torch.nn.Module,
    tokenizer,
    infer_cfg: InferenceConfig,
    device: str,
) -> dict:
    bundle = LoadedBundle(
        model=model,
        tokenizer=tokenizer,
        bad_words_ids=_build_bad_words_ids(tokenizer),
        device=device,
    )
    preds = generate_predictions(micro_sources, bundle, infer_cfg)
    preds = clean_target_batch(preds)
    targets = clean_target_batch(micro_targets)
    pair = _text_pair_metrics(preds, targets)
    spam_rate, spam_idx = spam_signature(preds)
    chrf = _chrf2_consistency(preds, targets)
    return {
        "step": int(step),
        "train_loss": float(loss),
        "exact_match_pct": pair["exact_match_pct"],
        "avg_char_edit_distance": pair["avg_char_edit_distance"],
        "avg_output_length": pair["avg_output_length"],
        "avg_target_length": pair["avg_target_length"],
        "avg_length_ratio_output_to_target": pair["avg_length_ratio_output_to_target"],
        "spam_rate": float(spam_rate),
        "spam_count": int(len(spam_idx)),
        "chrf2_metric": chrf["chrf2_metric"],
        "chrf2_reference": chrf["chrf2_reference"],
        "chrf2_abs_diff": chrf["abs_diff"],
    }


def _run_micro_variant(
    *,
    name: str,
    use_lora: bool,
    use_prefix: bool,
    micro_sources: list[str],
    micro_targets: list[str],
    base_model: str,
    tokenizer,
    lora_cfg: LoraConfig,
    max_src_len: int,
    max_tgt_len: int,
    infer_template: InferenceConfig,
    steps: int,
    lr: float,
    max_grad_norm: float,
    loss_ratio_threshold: float,
    spam_threshold: float,
    min_chrf2: float,
    seed: int,
    device: str,
    train_batch_size: int | None = None,
    weight_decay: float = 0.0,
    force_dropout_zero: bool = False,
    log_interval: int | None = None,
    emit_examples: bool = False,
) -> dict:
    set_deterministic(seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    model.config.forced_bos_token_id = None
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    if use_lora:
        model = get_peft_model(model, lora_cfg)
    if force_dropout_zero:
        _set_dropout_zero(model)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.to(device)

    task_prefix = infer_template.task_prefix if use_prefix else ""
    train_domains = infer_dialog_domain_batch(micro_sources) if infer_template.use_domain_tag else None
    src_train = format_source_batch(micro_sources, task_prefix=task_prefix, domains=train_domains)
    tgt_train = clean_target_batch(micro_targets)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"{name}: no trainable parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    mb = int(train_batch_size or len(src_train))
    mb = max(1, min(mb, len(src_train)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(src_train))
    cursor = 0
    log_every = int(log_interval or 0)
    history: list[dict] = []

    model.eval()
    with torch.no_grad():
        start_loss = float(
            _forward_seq2seq_loss(
                model=model,
                tokenizer=tokenizer,
                source_texts=src_train,
                target_texts=tgt_train,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                device=device,
            ).detach().cpu()
        )

    model.train()
    for step in range(1, steps + 1):
        if cursor + mb > len(order):
            order = rng.permutation(len(src_train))
            cursor = 0
        idx = order[cursor : cursor + mb]
        cursor += mb
        batch_src = [src_train[int(i)] for i in idx]
        batch_tgt = [tgt_train[int(i)] for i in idx]
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_seq2seq_loss(
            model=model,
            tokenizer=tokenizer,
            source_texts=batch_src,
            target_texts=batch_tgt,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        if log_every > 0 and step % log_every == 0:
            model.eval()
            with torch.no_grad():
                full_loss = float(
                    _forward_seq2seq_loss(
                        model=model,
                        tokenizer=tokenizer,
                        source_texts=src_train,
                        target_texts=tgt_train,
                        max_src_len=max_src_len,
                        max_tgt_len=max_tgt_len,
                        device=device,
                    ).detach().cpu()
                )
            snap = _collect_eval_snapshot(
                step=step,
                loss=full_loss,
                micro_sources=micro_sources,
                micro_targets=micro_targets,
                model=model,
                tokenizer=tokenizer,
                infer_cfg=InferenceConfig(
                    base_model=infer_template.base_model,
                    input_max_length=infer_template.input_max_length,
                    max_length=infer_template.max_length,
                    max_new_tokens=infer_template.max_new_tokens,
                    num_beams=infer_template.num_beams,
                    no_repeat_ngram_size=infer_template.no_repeat_ngram_size,
                    repetition_penalty=infer_template.repetition_penalty,
                    early_stopping=infer_template.early_stopping,
                    task_prefix=task_prefix,
                    use_domain_tag=infer_template.use_domain_tag,
                    batch_size=infer_template.batch_size,
                    seed=infer_template.seed,
                    device=infer_template.device,
                ),
                device=device,
            )
            history.append(snap)
            print(
                f"[{name}] step={step} loss={snap['train_loss']:.4f} "
                f"exact={snap['exact_match_pct']:.1f}% "
                f"edit={snap['avg_char_edit_distance']:.1f} "
                f"len_ratio={snap['avg_length_ratio_output_to_target']:.3f}"
            )
            model.train()

    model.eval()
    with torch.no_grad():
        end_loss = float(
            _forward_seq2seq_loss(
                model=model,
                tokenizer=tokenizer,
                source_texts=src_train,
                target_texts=tgt_train,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                device=device,
            ).detach().cpu()
        )

    infer_cfg = InferenceConfig(
        base_model=infer_template.base_model,
        input_max_length=infer_template.input_max_length,
        max_length=infer_template.max_length,
        max_new_tokens=infer_template.max_new_tokens,
        num_beams=infer_template.num_beams,
        no_repeat_ngram_size=infer_template.no_repeat_ngram_size,
        repetition_penalty=infer_template.repetition_penalty,
        early_stopping=infer_template.early_stopping,
        task_prefix=task_prefix,
        batch_size=infer_template.batch_size,
        seed=infer_template.seed,
        device=infer_template.device,
    )
    bundle = LoadedBundle(
        model=model,
        tokenizer=tokenizer,
        bad_words_ids=_build_bad_words_ids(tokenizer),
        device=device,
    )
    preds = generate_predictions(micro_sources, bundle, infer_cfg)
    preds = clean_target_batch(preds)
    targets = clean_target_batch(micro_targets)
    chrf = _chrf2_consistency(preds, targets)
    chrf2 = chrf["chrf2_metric"]
    spam_rate, spam_idx = spam_signature(preds)
    pair_metrics = _text_pair_metrics(preds, targets)

    loss_ratio = (end_loss / start_loss) if start_loss > 0 else float("inf")
    loss_pass = loss_ratio < loss_ratio_threshold
    gen_pass = (spam_rate < spam_threshold) and (chrf2 >= min_chrf2)
    variant_pass = bool(loss_pass and gen_pass)

    examples = []
    for i in range(len(micro_sources)):
        examples.append(
            {
                "source": micro_sources[i],
                "target": targets[i],
                "prediction": preds[i],
            }
        )

    out = {
        "name": name,
        "use_lora": use_lora,
        "use_prefix": use_prefix,
        "start_loss": start_loss,
        "end_loss": end_loss,
        "loss_ratio": loss_ratio,
        "loss_ratio_threshold": loss_ratio_threshold,
        "loss_pass": bool(loss_pass),
        "chrf2": chrf2,
        "chrf2_min": min_chrf2,
        "chrf2_reference": chrf["chrf2_reference"],
        "chrf2_abs_diff": chrf["abs_diff"],
        "spam_rate": float(spam_rate),
        "spam_count": int(len(spam_idx)),
        "spam_threshold": spam_threshold,
        "exact_match_pct": pair_metrics["exact_match_pct"],
        "avg_char_edit_distance": pair_metrics["avg_char_edit_distance"],
        "avg_output_length": pair_metrics["avg_output_length"],
        "avg_target_length": pair_metrics["avg_target_length"],
        "avg_length_ratio_output_to_target": pair_metrics["avg_length_ratio_output_to_target"],
        "gen_pass": bool(gen_pass),
        "pass": variant_pass,
        "history": history,
        "examples": examples if emit_examples else examples[:2],
    }

    del optimizer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _run_micro_preflight(
    *,
    cfg: dict,
    run_dir: Path,
    train_df_raw: pd.DataFrame,
    train_src_trunc: np.ndarray,
    train_tgt_trunc: np.ndarray,
    src_col: str,
    tgt_col: str,
    base_model: str,
    tokenizer,
    max_src_len: int,
    max_tgt_len: int,
    infer_template: InferenceConfig,
    seed: int,
    device: str,
) -> dict | None:
    pre_cfg = cfg.get("preflight", {})
    enabled = bool(pre_cfg.get("enabled", False))
    if not enabled:
        return None

    sample_size = int(pre_cfg.get("sample_size", 8))
    steps = int(pre_cfg.get("steps", 120))
    lr = float(pre_cfg.get("learning_rate", 5e-6))
    max_grad_norm = float(pre_cfg.get("max_grad_norm", 1.0))
    loss_ratio_threshold = float(pre_cfg.get("loss_ratio_threshold", 0.6))
    spam_threshold = float(pre_cfg.get("spam_threshold", 0.10))
    full_min_chrf2 = float(pre_cfg.get("full_prefix_on_min_chrf2", 20.0))
    lora_min_chrf2 = float(pre_cfg.get("lora_prefix_on_min_chrf2", 15.0))
    run_control = bool(pre_cfg.get("run_full_prefix_off_control", True))
    enforce = bool(pre_cfg.get("enforce", True))

    clean_mask = ~(train_src_trunc | train_tgt_trunc)
    clean_df = train_df_raw.loc[clean_mask].reset_index(drop=True)
    if len(clean_df) < sample_size:
        raise RuntimeError(
            f"Micro preflight needs {sample_size} clean examples, found {len(clean_df)}."
        )

    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(clean_df), size=sample_size, replace=False))
    micro_df = clean_df.iloc[picked].reset_index(drop=True)
    micro_sources = micro_df[src_col].astype(str).fillna("").tolist()
    micro_targets = clean_target_batch(micro_df[tgt_col].astype(str).fillna("").tolist())

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
    )

    variants = [
        {
            "name": "full_prefix_on",
            "use_lora": False,
            "use_prefix": True,
            "min_chrf2": full_min_chrf2,
            "enforce": True,
        },
        {
            "name": "lora_prefix_on",
            "use_lora": True,
            "use_prefix": True,
            "min_chrf2": lora_min_chrf2,
            "enforce": True,
        },
    ]
    if run_control:
        variants.append(
            {
                "name": "full_prefix_off_control",
                "use_lora": False,
                "use_prefix": False,
                "min_chrf2": 0.0,
                "enforce": False,
            }
        )

    results: list[dict] = []
    for v in variants:
        result = _run_micro_variant(
            name=v["name"],
            use_lora=bool(v["use_lora"]),
            use_prefix=bool(v["use_prefix"]),
            micro_sources=micro_sources,
            micro_targets=micro_targets,
            base_model=base_model,
            tokenizer=tokenizer,
            lora_cfg=lora_cfg,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            infer_template=infer_template,
            steps=steps,
            lr=lr,
            max_grad_norm=max_grad_norm,
            loss_ratio_threshold=loss_ratio_threshold,
            spam_threshold=spam_threshold,
            min_chrf2=float(v["min_chrf2"]),
            seed=seed,
            device=device,
        )
        result["enforce"] = bool(v["enforce"])
        results.append(result)

    failed_required = [r["name"] for r in results if r["enforce"] and (not r["pass"])]
    overall_pass = len(failed_required) == 0
    summary = {
        "enabled": True,
        "enforce": enforce,
        "sample_size": sample_size,
        "steps": steps,
        "learning_rate": lr,
        "max_grad_norm": max_grad_norm,
        "loss_ratio_threshold": loss_ratio_threshold,
        "spam_threshold": spam_threshold,
        "task_prefix": infer_template.task_prefix,
        "overall_pass": overall_pass,
        "failed_required_variants": failed_required,
        "variants": results,
    }
    write_json(run_dir / "micro_preflight.json", summary)

    print("")
    print("Micro preflight variants")
    print("Variant\tLossRatio\tchrF2\tSpam\tPass")
    for r in results:
        print(
            f"{r['name']}\t{r['loss_ratio']:.3f}\t{r['chrf2']:.3f}\t"
            f"{r['spam_rate']:.3f}\t{r['pass']}"
        )
    print(
        "Micro preflight gate\t"
        f"{'PASS' if overall_pass else 'FAIL'}"
        f"{'' if overall_pass else ' (' + ','.join(failed_required) + ')'}"
    )
    return summary


def _run_micro_diagnostic(
    *,
    cfg: dict,
    run_dir: Path,
    train_df_raw: pd.DataFrame,
    train_src_trunc: np.ndarray,
    train_tgt_trunc: np.ndarray,
    src_col: str,
    tgt_col: str,
    base_model: str,
    tokenizer,
    max_src_len: int,
    max_tgt_len: int,
    infer_template: InferenceConfig,
    seed: int,
    device: str,
) -> dict:
    diag_cfg = cfg.get("micro_diagnostic", {})
    sample_size = int(diag_cfg.get("sample_size", 8))
    steps = int(diag_cfg.get("steps", 1500))
    log_interval = int(diag_cfg.get("log_interval", 100))
    batch_size = int(diag_cfg.get("train_batch_size", 1))
    full_lr = float(diag_cfg.get("full_ft_lr", 5e-4))
    lora_lr = float(diag_cfg.get("lora_lr", 1e-3))
    max_grad_norm = float(diag_cfg.get("max_grad_norm", 1.0))
    loss_ratio_threshold = float(diag_cfg.get("loss_ratio_threshold", 0.3))
    exact_match_threshold = float(diag_cfg.get("exact_match_threshold", 50.0))
    enforce = bool(diag_cfg.get("enforce", True))

    clean_mask = ~(train_src_trunc | train_tgt_trunc)
    clean_df = train_df_raw.loc[clean_mask].reset_index(drop=True)
    if len(clean_df) < sample_size:
        raise RuntimeError(
            f"Micro diagnostic needs {sample_size} clean examples, found {len(clean_df)}."
        )

    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(clean_df), size=sample_size, replace=False))
    micro_df = clean_df.iloc[picked].reset_index(drop=True)
    micro_sources = micro_df[src_col].astype(str).fillna("").tolist()
    micro_targets = clean_target_batch(micro_df[tgt_col].astype(str).fillna("").tolist())

    unicode_report = {
        "source_full": _unicode_audit(train_df_raw[src_col].astype(str).fillna("").tolist()),
        "target_full": _unicode_audit(train_df_raw[tgt_col].astype(str).fillna("").tolist()),
        "source_micro": _unicode_audit(micro_sources),
        "target_micro": _unicode_audit(micro_targets),
    }
    write_json(run_dir / "unicode_audit.json", unicode_report)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=0.0,
        target_modules=list(cfg["lora"]["target_modules"]),
    )

    full_result = _run_micro_variant(
        name="full_ft_prefix_on_diag",
        use_lora=False,
        use_prefix=True,
        micro_sources=micro_sources,
        micro_targets=micro_targets,
        base_model=base_model,
        tokenizer=tokenizer,
        lora_cfg=lora_cfg,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        infer_template=infer_template,
        steps=steps,
        lr=full_lr,
        max_grad_norm=max_grad_norm,
        loss_ratio_threshold=loss_ratio_threshold,
        spam_threshold=1.0,
        min_chrf2=0.0,
        seed=seed,
        device=device,
        train_batch_size=batch_size,
        weight_decay=0.0,
        force_dropout_zero=True,
        log_interval=log_interval,
        emit_examples=True,
    )
    lora_result = _run_micro_variant(
        name="lora_prefix_on_diag",
        use_lora=True,
        use_prefix=True,
        micro_sources=micro_sources,
        micro_targets=micro_targets,
        base_model=base_model,
        tokenizer=tokenizer,
        lora_cfg=lora_cfg,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        infer_template=infer_template,
        steps=steps,
        lr=lora_lr,
        max_grad_norm=max_grad_norm,
        loss_ratio_threshold=loss_ratio_threshold,
        spam_threshold=1.0,
        min_chrf2=0.0,
        seed=seed,
        device=device,
        train_batch_size=batch_size,
        weight_decay=0.0,
        force_dropout_zero=True,
        log_interval=log_interval,
        emit_examples=True,
    )

    variants = [full_result, lora_result]
    failed = []
    for r in variants:
        diag_pass = bool(r["loss_ratio"] < loss_ratio_threshold and r["exact_match_pct"] >= exact_match_threshold)
        r["diagnostic_pass"] = diag_pass
        if not diag_pass:
            failed.append(r["name"])

    summary = {
        "enabled": True,
        "enforce": enforce,
        "sample_size": sample_size,
        "steps": steps,
        "log_interval": log_interval,
        "train_batch_size": batch_size,
        "full_ft_lr": full_lr,
        "lora_lr": lora_lr,
        "max_grad_norm": max_grad_norm,
        "loss_ratio_threshold": loss_ratio_threshold,
        "exact_match_threshold": exact_match_threshold,
        "overall_pass": len(failed) == 0,
        "failed_variants": failed,
        "unicode_audit_path": str((run_dir / "unicode_audit.json").resolve()),
        "variants": variants,
    }
    write_json(run_dir / "micro_diagnostic.json", summary)

    print("")
    print("Micro diagnostic summary")
    print("Variant\tLossRatio\tExact%\tAvgEdit\tLenRatio\tchrF2\tPass")
    for r in variants:
        print(
            f"{r['name']}\t{r['loss_ratio']:.3f}\t{r['exact_match_pct']:.1f}\t"
            f"{r['avg_char_edit_distance']:.1f}\t{r['avg_length_ratio_output_to_target']:.3f}\t"
            f"{r['chrf2']:.3f}\t{r['diagnostic_pass']}"
        )

    for r in variants:
        print("")
        print(f"Outputs vs targets: {r['name']}")
        for i, ex in enumerate(r["examples"]):
            print(f"[{i}] TARGET: {ex['target']}")
            print(f"[{i}] PRED  : {ex['prediction']}")

    print("")
    print(
        "Micro diagnostic gate\t"
        f"{'PASS' if summary['overall_pass'] else 'FAIL'}"
        f"{'' if summary['overall_pass'] else ' (' + ','.join(summary['failed_variants']) + ')'}"
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    repo_root = PATHS.root
    cfg_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    set_deterministic(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_prefix = str(cfg.get("run_name", "lora"))
    run_id = args.run_id or new_run_id(repo_root, prefix=run_prefix)
    output_root = _resolve_path(repo_root, cfg["artifacts"]["output_root"])
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dst = run_dir / "config.yaml"
    copy_file(cfg_path, config_dst)

    comp_dir = _resolve_path(repo_root, cfg["data"]["competition_dir"])
    schema_path = _resolve_path(repo_root, cfg["data"]["schema_path"])
    val_frac = float(cfg["data"]["val_frac"])
    truncation_policy = str(cfg["data"].get("truncation_policy", "keep"))
    filter_val = bool(cfg["data"].get("apply_truncation_filter_to_val", False))
    fmt_cfg = cfg.get("formatting", {})
    task_prefix = str(fmt_cfg.get("task_prefix", DEFAULT_TASK_PREFIX))
    use_domain_tag = bool(fmt_cfg.get("use_domain_tag", False))

    base_model = str(cfg["model"]["base_model"])
    use_fast = bool(cfg["model"]["use_fast_tokenizer"])
    max_src_len = int(cfg["model"]["max_source_length"])
    max_tgt_len = int(cfg["model"]["max_target_length"])
    gen_cfg = cfg.get("generation", {})
    gen_beams = int(gen_cfg.get("num_beams", 4))
    gen_max_length = _optional_int(gen_cfg.get("max_length", 96), default=96)
    gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 128))
    gen_no_repeat_ngram = int(gen_cfg.get("no_repeat_ngram_size", 3))
    gen_repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.1))
    gen_early_stopping = bool(gen_cfg.get("early_stopping", True))
    gen_do_sample = bool(gen_cfg.get("do_sample", False))
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)

    data = load_deep_past_competition(comp_dir, schema_path)
    src_col = data.schema.train_text_col
    tgt_col = data.schema.train_target_col
    df = data.train.copy()
    df[src_col] = df[src_col].astype(str).fillna("")
    df[tgt_col] = df[tgt_col].astype(str).fillna("")

    train_idx, val_idx = _split_idx(len(df), seed=seed, val_frac=val_frac)
    train_df_raw = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_src_len, train_src_trunc = _token_length_profile(
        tokenizer, train_df_raw[src_col].tolist(), max_src_len
    )
    train_tgt_len, train_tgt_trunc = _token_length_profile(
        tokenizer, train_df_raw[tgt_col].tolist(), max_tgt_len
    )
    train_keep = _build_keep_mask(truncation_policy, train_src_trunc, train_tgt_trunc)

    val_src_len, val_src_trunc = _token_length_profile(
        tokenizer, val_df[src_col].tolist(), max_src_len
    )
    val_tgt_len, val_tgt_trunc = _token_length_profile(
        tokenizer, val_df[tgt_col].tolist(), max_tgt_len
    )
    val_keep = _build_keep_mask(truncation_policy, val_src_trunc, val_tgt_trunc)

    train_rows_before = len(train_df_raw)
    val_rows_before = len(val_df)
    train_df = train_df_raw.loc[train_keep].reset_index(drop=True)
    if filter_val:
        val_df = val_df.loc[val_keep].reset_index(drop=True)

    if train_df.empty:
        raise RuntimeError(
            "Training dataframe became empty after truncation filtering. "
            "Relax truncation_policy or increase max_source_length/max_target_length."
        )

    fp = _data_fingerprint(
        competition_dir=comp_dir,
        schema_path=schema_path,
        train_file=data.schema.train_file,
        test_file=data.schema.test_file,
        sample_sub_file=data.schema.sample_submission_file,
        train_idx=train_idx,
        val_idx=val_idx,
        source_col=src_col,
        target_col=tgt_col,
    )
    fp["formatting"] = {"task_prefix": task_prefix, "use_domain_tag": use_domain_tag}
    fp["truncation"] = {
        "policy": truncation_policy,
        "max_source_length": max_src_len,
        "max_target_length": max_tgt_len,
        "train_rows_before": train_rows_before,
        "train_rows_after": len(train_df),
        "train_source_truncated_count": int(train_src_trunc.sum()),
        "train_target_truncated_count": int(train_tgt_trunc.sum()),
        "train_any_truncated_count": int((train_src_trunc | train_tgt_trunc).sum()),
        "val_rows_before": val_rows_before,
        "val_rows_after": len(val_df),
        "val_source_truncated_count": int(val_src_trunc.sum()),
        "val_target_truncated_count": int(val_tgt_trunc.sum()),
        "val_any_truncated_count": int((val_src_trunc | val_tgt_trunc).sum()),
        "apply_filter_to_val": filter_val,
        "train_source_len_median": float(np.median(train_src_len)),
        "train_target_len_median": float(np.median(train_tgt_len)),
        "val_source_len_median": float(np.median(val_src_len)),
        "val_target_len_median": float(np.median(val_tgt_len)),
    }
    write_json(run_dir / "data_fingerprint.json", fp)

    infer_template = InferenceConfig(
        base_model=base_model,
        input_max_length=max_src_len,
        max_length=gen_max_length,
        max_new_tokens=gen_max_new_tokens,
        num_beams=gen_beams,
        no_repeat_ngram_size=gen_no_repeat_ngram,
        repetition_penalty=gen_repetition_penalty,
        early_stopping=gen_early_stopping,
        task_prefix=task_prefix,
        use_domain_tag=use_domain_tag,
        batch_size=8,
        seed=seed,
        device=args.device,
    )

    if args.micro_diagnostic:
        diag_summary = _run_micro_diagnostic(
            cfg=cfg,
            run_dir=run_dir,
            train_df_raw=train_df_raw,
            train_src_trunc=train_src_trunc,
            train_tgt_trunc=train_tgt_trunc,
            src_col=src_col,
            tgt_col=tgt_col,
            base_model=base_model,
            tokenizer=tokenizer,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            infer_template=infer_template,
            seed=seed,
            device=args.device,
        )
        diag_enforced = bool(diag_summary["enforce"]) and (not args.no_enforce_gate)
        if diag_enforced and (not bool(diag_summary["overall_pass"])):
            raise SystemExit(4)
        return

    preflight_summary = _run_micro_preflight(
        cfg=cfg,
        run_dir=run_dir,
        train_df_raw=train_df_raw,
        train_src_trunc=train_src_trunc,
        train_tgt_trunc=train_tgt_trunc,
        src_col=src_col,
        tgt_col=tgt_col,
        base_model=base_model,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        infer_template=infer_template,
        seed=seed,
        device=args.device,
    )

    if args.preflight_only:
        if preflight_summary is None:
            print("Micro preflight disabled in config; nothing to run.")
            return
        preflight_enforced = bool(preflight_summary["enforce"]) and (not args.no_enforce_gate)
        if preflight_enforced and (not bool(preflight_summary["overall_pass"])):
            raise SystemExit(3)
        return

    if preflight_summary is not None:
        preflight_enforced = bool(preflight_summary["enforce"]) and (not args.no_enforce_gate)
        if preflight_enforced and (not bool(preflight_summary["overall_pass"])):
            raise SystemExit(3)

    trn_ds = Dataset.from_pandas(train_df[[src_col, tgt_col]])
    val_ds = Dataset.from_pandas(val_df[[src_col, tgt_col]])

    def preprocess(batch):
        domains = infer_dialog_domain_batch(batch[src_col]) if use_domain_tag else None
        sources = format_source_batch(batch[src_col], task_prefix=task_prefix, domains=domains)
        targets = clean_target_batch(batch[tgt_col])
        model_inputs = tokenizer(
            sources,
            max_length=max_src_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_tgt_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    trn_tok = trn_ds.map(
        preprocess,
        batched=True,
        remove_columns=[src_col, tgt_col],
        num_proc=1,
        desc="Tokenizing train",
    )
    val_tok = val_ds.map(
        preprocess,
        batched=True,
        remove_columns=[src_col, tgt_col],
        num_proc=1,
        desc="Tokenizing val",
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    model.config.forced_bos_token_id = None
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"]["dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.generation_config.num_beams = gen_beams
    model.generation_config.max_length = gen_max_length
    model.generation_config.max_new_tokens = gen_max_new_tokens
    model.generation_config.do_sample = gen_do_sample
    model.generation_config.no_repeat_ngram_size = gen_no_repeat_ngram
    model.generation_config.repetition_penalty = gen_repetition_penalty
    model.generation_config.early_stopping = gen_early_stopping
    model.generation_config.bad_words_ids = _build_bad_words_ids(tokenizer)

    metric = CHRF(word_order=2)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, np.ndarray) and preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        pred_text = clean_target_batch(tokenizer.batch_decode(preds, skip_special_tokens=True))
        gold_text = clean_target_batch(tokenizer.batch_decode(labels, skip_special_tokens=True))
        return {"chrf2": metric.corpus_score(pred_text, [gold_text]).score}

    train_cfg = cfg["training"]
    trainer_generation_max_length = gen_max_length if gen_max_length is not None else gen_max_new_tokens
    hf_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy=str(train_cfg["eval_strategy"]),
        save_strategy=str(train_cfg["save_strategy"]),
        logging_strategy="steps",
        logging_steps=int(train_cfg["logging_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        optim=str(train_cfg["optim"]),
        predict_with_generate=True,
        generation_max_length=trainer_generation_max_length,
        generation_num_beams=gen_beams,
        save_total_limit=int(train_cfg["save_total_limit"]),
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=str(train_cfg["metric_for_best_model"]),
        greater_is_better=bool(train_cfg["greater_is_better"]),
        report_to=[],
        seed=seed,
        data_seed=seed,
        no_cuda=(args.device == "cpu"),
    )
    hf_args.generation_config = model.generation_config

    trainer = Seq2SeqTrainer(
        model=model,
        args=hf_args,
        train_dataset=trn_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(
        "Truncation policy: "
        f"{truncation_policy} | "
        f"train_truncated_any={int((train_src_trunc | train_tgt_trunc).sum())}/{train_rows_before} | "
        f"val_truncated_any={int((val_src_trunc | val_tgt_trunc).sum())}/{val_rows_before}"
    )
    print(
        f"Training hyperparams: lr={train_cfg['learning_rate']}, epochs={train_cfg['num_train_epochs']}, "
        f"grad_acc={train_cfg['gradient_accumulation_steps']}"
    )
    print(
        "Formatting/decoding: "
        f"task_prefix={task_prefix!r}, beams={gen_beams}, max_length={gen_max_length}, "
        f"no_repeat_ngram={gen_no_repeat_ngram}, repetition_penalty={gen_repetition_penalty}, "
        f"early_stopping={gen_early_stopping}, use_domain_tag={use_domain_tag}"
    )
    trainer.train()

    adapter_dir = run_dir / "adapter"
    tok_dir = run_dir / "tokenizer"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(tok_dir))

    best_checkpoint = trainer.state.best_model_checkpoint
    write_json(run_dir / "training_args.json", hf_args.to_dict())
    write_json(
        run_dir / "trainer_state_summary.json",
        {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": best_checkpoint,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "log_history": trainer.state.log_history,
        },
    )

    del trainer
    del model
    torch.cuda.empty_cache()

    infer = infer_template
    gate_cfg = cfg["gate"]
    summary = evaluate_models(
        competition_dir=comp_dir,
        schema_path=schema_path,
        adapter_dir=adapter_dir,
        inference=infer,
        seed=seed,
        val_frac=val_frac,
        min_delta_over_baseline=float(gate_cfg["min_delta_over_baseline"]),
        spam_threshold=float(gate_cfg["spam_threshold"]),
    )
    write_json(run_dir / "metrics.json", asdict(summary))

    submission = make_lora_submission(
        output_csv=run_dir / "submission.csv",
        competition_dir=comp_dir,
        schema_path=schema_path,
        adapter_dir=adapter_dir,
        tokenizer_dir=tok_dir,
        config=infer,
        verify_determinism=True,
    )
    write_text(run_dir / "submission.csv.sha256", f"{submission.output_sha256}\n")

    manifest = {
        "run_id": run_id,
        "created_at_utc": now_utc_iso(),
        "git_short_hash": git_short_hash(repo_root),
        "command": command_string(),
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "config": str(config_dst.resolve()),
            "training_script": str(Path(__file__).resolve()),
            "adapter_dir": str(adapter_dir.resolve()),
            "tokenizer_dir": str(tok_dir.resolve()),
            "submission_csv": str(submission.output_csv.resolve()),
            "submission_sha_file": str((run_dir / "submission.csv.sha256").resolve()),
        },
        "hashes": {
            "config_sha256": file_sha256(config_dst),
            "training_script_sha256": file_sha256(Path(__file__).resolve()),
            "submission_sha256": submission.output_sha256,
        },
        "training": {
            "base_model": base_model,
            "seed": seed,
            "best_metric": summary.lora_chrF2,
            "best_checkpoint": best_checkpoint,
        },
        "evaluation_reference": {
            "baseline_model_kind": summary.baseline_model_kind,
            "baseline_model_id": summary.baseline_model_id,
            "baseline_domain_override": {
                "enabled": summary.baseline_use_domain_override,
                "rate": summary.baseline_domain_override_rate,
                "count": summary.baseline_domain_override_count,
            },
            "lora_base_model": summary.lora_base_model,
            "lora_decode": {
                "input_max_length": summary.lora_input_max_length,
                "max_length": summary.lora_max_length,
                "max_new_tokens": summary.lora_max_new_tokens,
                "num_beams": summary.lora_num_beams,
                "no_repeat_ngram_size": summary.lora_no_repeat_ngram_size,
                "repetition_penalty": summary.lora_repetition_penalty,
                "early_stopping": summary.lora_early_stopping,
                "task_prefix": summary.lora_task_prefix,
                "use_domain_tag": summary.lora_use_domain_tag,
            },
        },
        "gate": {
            "pass": summary.gate_pass,
            "reason": summary.gate_reason,
            "min_delta_over_baseline": summary.min_delta_over_baseline,
            "spam_threshold": summary.spam_threshold,
            "spam_rate": summary.spam_rate,
            "delta_chrF2": summary.delta_chrF2,
        },
    }
    if preflight_summary is not None:
        manifest["preflight"] = {
            "path": str((run_dir / "micro_preflight.json").resolve()),
            "overall_pass": bool(preflight_summary["overall_pass"]),
            "failed_required_variants": list(preflight_summary["failed_required_variants"]),
        }
    write_json(run_dir / "manifest.json", manifest)

    print("")
    print(f"Baseline model\t{summary.baseline_model_id}")
    print(f"LoRA base model\t{summary.lora_base_model}")
    print("Model\tchrF2")
    print(f"Baseline\t{summary.baseline_chrF2:.4f}")
    print(f"LoRA\t{summary.lora_chrF2:.4f}")
    print(f"Delta\t{summary.delta_chrF2:.4f}")
    print(f"Spam rate\t{summary.spam_rate:.3f}")
    print(f"Gate\t{'PASS' if summary.gate_pass else 'FAIL'} ({summary.gate_reason})")
    print(f"Submission SHA256\t{submission.output_sha256}")
    print(f"Artifacts\t{run_dir}")

    enforce_gate = bool(gate_cfg["enforce"]) and (not args.no_enforce_gate)
    if enforce_gate and not summary.gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
