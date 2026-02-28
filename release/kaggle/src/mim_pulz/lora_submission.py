from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mim_pulz.config import PATHS
from mim_pulz.data import load_deep_past_competition
from mim_pulz.domain_intent import infer_dialog_domain_batch
from mim_pulz.seq2seq_format import DEFAULT_TASK_PREFIX, clean_text, format_source_batch


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _has_adapter_files(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def resolve_default_adapter_dir() -> Path:
    candidates = [
        PATHS.models / "mt5_lora_v1" / "best_prev_epoch6",
        PATHS.models / "mt5_lora_v1" / "checkpoint-474",
        PATHS.models / "mt5_lora_v1" / "adapter",
        PATHS.models / "mt5_lora_v2" / "adapter",
    ]
    for c in candidates:
        if _has_adapter_files(c):
            return c
    raise FileNotFoundError(
        "No LoRA adapter found. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def _has_tokenizer_files(path: Path) -> bool:
    return (path / "tokenizer_config.json").exists() and (path / "spiece.model").exists()


def resolve_tokenizer_source(adapter_dir: Path, explicit_tokenizer_dir: Path | None) -> str:
    candidates: list[Path] = []
    if explicit_tokenizer_dir is not None:
        candidates.append(explicit_tokenizer_dir)
    candidates.extend(
        [
            adapter_dir,
            adapter_dir.parent / "tokenizer",
            PATHS.models / "mt5_lora_v1" / "tokenizer",
            PATHS.models / "mt5_lora_v2" / "tokenizer",
        ]
    )
    for c in candidates:
        if _has_tokenizer_files(c):
            return str(c)
    return "google/mt5-small"


def _build_bad_words_ids(tokenizer) -> list[list[int]]:
    ids: list[list[int]] = []
    seen: set[int] = set()
    for i in range(100):
        token_id: int | None = None
        encoded = tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False)
        if len(encoded) == 1:
            token_id = int(encoded[0])
        else:
            fallback = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            if fallback is not None and fallback >= 0:
                token_id = int(fallback)
        if token_id is not None and token_id >= 0 and token_id not in seen:
            seen.add(token_id)
            ids.append([token_id])
    return ids


@dataclass(frozen=True)
class InferenceConfig:
    base_model: str = "google/mt5-small"
    input_max_length: int = 256
    max_length: int | None = None
    max_new_tokens: int = 128
    num_beams: int = 1
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0
    early_stopping: bool = False
    task_prefix: str = DEFAULT_TASK_PREFIX
    use_domain_tag: bool = False
    batch_size: int = 8
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LoadedBundle:
    model: torch.nn.Module
    tokenizer: object
    bad_words_ids: list[list[int]]
    device: str


def load_lora_bundle(
    adapter_dir: Path,
    config: InferenceConfig,
    tokenizer_dir: Path | None = None,
) -> LoadedBundle:
    if not _has_adapter_files(adapter_dir):
        raise FileNotFoundError(f"Adapter files not found in: {adapter_dir}")

    tok_src = resolve_tokenizer_source(adapter_dir, tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    model.to(config.device)

    bad_words_ids = _build_bad_words_ids(tokenizer)
    return LoadedBundle(
        model=model,
        tokenizer=tokenizer,
        bad_words_ids=bad_words_ids,
        device=config.device,
    )


def generate_predictions(
    texts: list[str],
    bundle: LoadedBundle,
    config: InferenceConfig,
) -> list[str]:
    preds: list[str] = []
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    with torch.inference_mode():
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i : i + config.batch_size]
            batch_domains = infer_dialog_domain_batch(batch) if config.use_domain_tag else None
            batch = format_source_batch(batch, task_prefix=config.task_prefix, domains=batch_domains)
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.input_max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            gen_kwargs: dict = {
                "do_sample": False,
                "num_beams": config.num_beams,
                "bad_words_ids": bundle.bad_words_ids,
            }
            if config.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
            if config.repetition_penalty and config.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = config.repetition_penalty
            if config.early_stopping:
                gen_kwargs["early_stopping"] = True
            if config.max_length is not None:
                gen_kwargs["max_length"] = config.max_length
            else:
                gen_kwargs["max_new_tokens"] = config.max_new_tokens
            generated = model.generate(**encoded, **gen_kwargs)
            batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            preds.extend(clean_text(x) for x in batch_preds)

    return preds


def read_epoch_chrf2(trainer_state_path: Path) -> list[dict]:
    if not trainer_state_path.exists():
        return []
    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for item in state.get("log_history", []):
        if "eval_chrf2" not in item:
            continue
        out.append(
            {
                "epoch": float(item.get("epoch", 0.0)),
                "global_step": int(item.get("step", 0)),
                "eval_chrf2": float(item["eval_chrf2"]),
                "eval_loss": float(item.get("eval_loss", 0.0)),
            }
        )
    return out


@dataclass(frozen=True)
class SubmissionResult:
    output_csv: Path
    metadata_json: Path
    output_sha256: str
    rows: int


def make_lora_submission(
    output_csv: Path,
    competition_dir: Path,
    schema_path: Path,
    adapter_dir: Path | None = None,
    tokenizer_dir: Path | None = None,
    config: InferenceConfig | None = None,
    verify_determinism: bool = False,
) -> SubmissionResult:
    cfg = config or InferenceConfig()
    set_deterministic(cfg.seed)

    chosen_adapter = adapter_dir or resolve_default_adapter_dir()
    data = load_deep_past_competition(competition_dir, schema_path)
    test_texts = data.test[data.schema.test_text_col].astype(str).fillna("").tolist()

    bundle = load_lora_bundle(
        adapter_dir=chosen_adapter,
        config=cfg,
        tokenizer_dir=tokenizer_dir,
    )
    preds = generate_predictions(test_texts, bundle, cfg)

    if verify_determinism:
        set_deterministic(cfg.seed)
        verify_preds = generate_predictions(test_texts, bundle, cfg)
        if verify_preds != preds:
            raise RuntimeError("Determinism check failed: repeated generation produced different outputs.")

    sub = data.sample_submission.copy()
    sub[data.schema.submission_target_col] = preds

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_csv, index=False)

    digest = sha256(output_csv.read_bytes()).hexdigest()
    metadata = {
        "output_csv": str(output_csv.resolve()),
        "output_sha256": digest,
        "rows": len(sub),
        "adapter_dir": str(chosen_adapter.resolve()),
        "tokenizer_source": resolve_tokenizer_source(chosen_adapter, tokenizer_dir),
        "base_model": cfg.base_model,
        "generation": {
            "input_max_length": cfg.input_max_length,
            "max_length": cfg.max_length,
            "max_new_tokens": cfg.max_new_tokens,
            "num_beams": cfg.num_beams,
            "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
            "repetition_penalty": cfg.repetition_penalty,
            "early_stopping": cfg.early_stopping,
            "task_prefix": cfg.task_prefix,
            "use_domain_tag": cfg.use_domain_tag,
            "do_sample": False,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "device": cfg.device,
        },
        "determinism_verified": verify_determinism,
        "versions": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    metadata_json = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return SubmissionResult(
        output_csv=output_csv,
        metadata_json=metadata_json,
        output_sha256=digest,
        rows=len(sub),
    )
