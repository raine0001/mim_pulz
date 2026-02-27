from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple

# Paths (hard-set to avoid guessing)
PROJECT_ROOT = Path(r"C:\MIM_pulz")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"

PUBLICATIONS_CSV = DATA_DIR / "raw" / "competition" / "publications.csv"
OUT_REGISTRY = CORPUS_DIR / "page_registry.jsonl"
OUT_ALIASES = CORPUS_DIR / "page_aliases.jsonl"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def strip_bom(s: str) -> str:
    return s.lstrip("\ufeff") if isinstance(s, str) else s


def normalize_text(s: str) -> str:
    """
    Normalize OCR text enough to dedupe reliably without destroying content.
    Keep it conservative.
    """
    if s is None:
        return ""
    # normalize newlines and whitespace
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)  # collapse runs of spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)  # collapse huge blank regions
    s = s.strip()
    return s


def lang_hint(text: str) -> str:
    """
    Very cheap heuristic. Good enough for routing, not scholarship.
    """
    t = text.lower()
    hits = {
        "de": sum(w in t for w in ["und ", " der ", " die ", " das ", "nicht ", "über ", "göttin", "belege"]),
        "en": sum(w in t for w in ["the ", "and ", "from ", "to ", "according to ", "trade", "king"]),
        "tr": sum(w in t for w in ["çalışmaları", "kültür", "bakanlığı", "yılı", "araştırma", "höyük"]),
    }
    best = max(hits.items(), key=lambda kv: kv[1])[0]
    # if everything is basically zero, mark mixed/unknown
    if hits[best] <= 1:
        return "mixed"
    return best


def ocr_quality_hint(text: str) -> str:
    """
    Another cheap heuristic: lots of broken hyphenation / strange spacing / OCR junk.
    """
    if not text:
        return "low"
    bad = 0
    bad += len(re.findall(r"\b[a-zA-Z]{1}\b", text))  # isolated letters
    bad += len(re.findall(r"[|]{2,}", text))
    bad += len(re.findall(r"\b\w+-\n\w+\b", text))  # hyphen linebreaks
    bad_ratio = bad / max(1, len(text) / 500)  # scale by size
    if bad_ratio < 2:
        return "high"
    if bad_ratio < 6:
        return "med"
    return "low"


def make_page_id(pdf_name: str, page_number: str, text_norm: str) -> str:
    content_hash = _sha1(text_norm)
    stable = f"{pdf_name}||{page_number}||{content_hash}"
    return _sha1(stable)


def build_registry(
    input_csv: Path = PUBLICATIONS_CSV,
    out_registry: Path = OUT_REGISTRY,
    out_aliases: Path = OUT_ALIASES,
) -> Tuple[int, int]:
    """
    Returns: (canonical_count, alias_count)
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # key: (pdf_name, page_number, text_hash) -> page_id
    seen: Dict[Tuple[str, str, str], str] = {}
    # key: (pdf_name, page_number) -> first canonical page_id (for duplicate page-number repeats)
    seen_page_pair: Dict[Tuple[str, str], str] = {}

    canonical_written = 0
    alias_written = 0

    with input_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        # normalize headers (handles BOM + stray whitespace)
        if reader.fieldnames:
            reader.fieldnames = [strip_bom(h).strip() for h in reader.fieldnames]
        print("[page_registry] input_csv =", input_csv.resolve())
        print("[page_registry] exists =", input_csv.exists(), "size_bytes =", input_csv.stat().st_size if input_csv.exists() else None)
        print("[page_registry] headers:", reader.fieldnames)

        total = 0
        skipped = 0

        with out_registry.open("w", encoding="utf-8") as reg_f, out_aliases.open("w", encoding="utf-8") as alias_f:
            for row in reader:
                normalized_row = {}
                for k, v in row.items():
                    key_norm = strip_bom(k)
                    if isinstance(key_norm, str):
                        key_norm = key_norm.strip()
                    normalized_row[key_norm] = v
                row = normalized_row
                total += 1
                pdf_name = (row.get("pdf_name") or "").strip()
                page_number = str(row.get("page") or "").strip()
                text_raw = row.get("page_text") or ""
                has_akkadian = str(row.get("has_akkadian") or "").strip().lower() in ("true", "1", "yes")

                if not pdf_name or not page_number:
                    skipped += 1
                    continue

                text_norm = normalize_text(text_raw)
                text_hash = _sha1(text_norm)

                key = (pdf_name, page_number, text_hash)

                if key not in seen:
                    page_id = make_page_id(pdf_name, page_number, text_norm)
                    seen[key] = page_id

                    record = {
                        "page_id": page_id,
                        "pdf_name": pdf_name,
                        "page_number": page_number,
                        "has_akkadian": has_akkadian,
                        "lang_hint": lang_hint(text_norm),
                        "ocr_quality_hint": ocr_quality_hint(text_norm),
                        "text_raw": text_raw,
                        "text_norm": text_norm,
                        "source_csv": input_csv.name,
                    }
                    reg_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    canonical_written += 1

                    # remember first canonical for that (pdf,page)
                    seen_page_pair.setdefault((pdf_name, page_number), page_id)

                else:
                    # Duplicate entry: alias -> canonical
                    canonical_id = seen[key]
                    alias_id = _sha1(f"alias||{pdf_name}||{page_number}||{alias_written}")
                    alias_record = {
                        "alias_id": alias_id,
                        "page_id": canonical_id,
                        "pdf_name": pdf_name,
                        "page_number": page_number,
                        "source_csv": input_csv.name,
                    }
                    alias_f.write(json.dumps(alias_record, ensure_ascii=False) + "\n")
                    alias_written += 1

    print(f"[page_registry] rows total={total} skipped={skipped}")
    print("[page_registry] rows_total:", total, "skipped_missing_fields:", skipped, "canonical:", canonical_written, "aliases:", alias_written)
    return canonical_written, alias_written


if __name__ == "__main__":
    c, a = build_registry()
    print(f"[page_registry] wrote canonical={c} aliases={a}")
