from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd
import requests


SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mim_pulz.config import PATHS
from mim_pulz.domain_intent import infer_dialog_domain_with_confidence
from utils_manifest import write_json


def _download(url: str, out_path: Path, timeout_sec: int = 180) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout_sec, stream=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return sha256(out_path.read_bytes()).hexdigest()


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _context_for(source_text: str, mode: str, fixed_context: str) -> str:
    if mode == "fixed":
        return fixed_context
    label, _, _ = infer_dialog_domain_with_confidence(source_text)
    return label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ORACC parallel memory (Zenodo) and build retrieval memory CSV."
    )
    parser.add_argument("--record-id", type=str, default="17220688")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PATHS.data_raw / "oracc_evacun_17220688",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PATHS.data_processed / "oracc_evacun_memory.csv",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=PATHS.data_processed / "oracc_evacun_memory.meta.json",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        choices=["infer", "fixed"],
        default="infer",
    )
    parser.add_argument("--fixed-context", type=str, default="oracc")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows after merge/filter.")
    parser.add_argument("--timeout-sec", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = f"https://zenodo.org/records/{args.record_id}/files"

    file_map = {
        "train": ("transcription_train.txt", "english_train.txt"),
        "validation": ("transcription_validation.txt", "english_validation.txt"),
    }

    hashes: dict[str, str] = {}
    rows: list[dict] = []
    split_stats: dict[str, dict] = {}

    for split, (src_name, tgt_name) in file_map.items():
        src_url = f"{base}/{src_name}?download=1"
        tgt_url = f"{base}/{tgt_name}?download=1"
        src_path = args.cache_dir / src_name
        tgt_path = args.cache_dir / tgt_name

        hashes[str(src_path)] = _download(src_url, src_path, timeout_sec=args.timeout_sec)
        hashes[str(tgt_path)] = _download(tgt_url, tgt_path, timeout_sec=args.timeout_sec)

        src_lines = _read_lines(src_path)
        tgt_lines = _read_lines(tgt_path)
        n_aligned = min(len(src_lines), len(tgt_lines))

        kept = 0
        dropped_empty = 0
        for i in range(n_aligned):
            s = str(src_lines[i]).strip()
            t = str(tgt_lines[i]).strip()
            if not s or not t:
                dropped_empty += 1
                continue
            rows.append(
                {
                    "source": s,
                    "target": t,
                    "context": _context_for(s, mode=args.context_mode, fixed_context=args.fixed_context),
                    "origin": f"oracc_evacun:{split}",
                    "split": split,
                    "record_id": str(args.record_id),
                }
            )
            kept += 1
            if args.limit > 0 and len(rows) >= args.limit:
                break

        split_stats[split] = {
            "source_file": src_name,
            "target_file": tgt_name,
            "source_lines": len(src_lines),
            "target_lines": len(tgt_lines),
            "aligned_pairs": n_aligned,
            "kept_pairs": kept,
            "dropped_empty": dropped_empty,
        }
        if args.limit > 0 and len(rows) >= args.limit:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No ORACC memory rows were created.")

    before_dedupe = len(frame)
    frame = frame.drop_duplicates(subset=["source", "target"]).reset_index(drop=True)
    after_dedupe = len(frame)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8")

    meta = {
        "record_id": str(args.record_id),
        "source": "zenodo",
        "base_url": base,
        "output_csv": str(args.output_csv.resolve()),
        "rows_before_dedupe": int(before_dedupe),
        "rows_after_dedupe": int(after_dedupe),
        "context_mode": args.context_mode,
        "fixed_context": args.fixed_context,
        "limit": int(args.limit),
        "split_stats": split_stats,
        "downloaded_file_sha256": hashes,
    }
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.meta_json, meta)

    print(f"Wrote memory CSV: {args.output_csv} ({after_dedupe} rows)")
    print(f"Wrote metadata: {args.meta_json}")


if __name__ == "__main__":
    main()
