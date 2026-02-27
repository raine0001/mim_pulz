from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.corpus.index_store import init_db, fetch_pages_batch, upsert_payload

DOC_TYPE_HINTS = {
    "letter": ["um-ma", "qA--bi-ma", "a-na", "brother"],
    "legal": ["tablet", "witness", "silver", "mina", "kaspum", "contract", "oath"],
    "commentary": ["line", "gloss", "exegesis", "commentary"],
}

TOPIC_TAGS = {
    "trade": ["mina", "shekel", "silver", "tin", "copper", "merch", "market", "karum", "naruqqum"],
    "debt": ["debt", "owed", "interest", "loan", "obligation"],
    "religion": ["temple", "offering", "god", "goddess", "ritual", "divine"],
    "institutions": ["karum", "wabartum", "limum", "hamu\u0161tum", "naruqqum", "assembly"],
}

INSTITUTIONS = ["karum", "wabartum", "limum", "hamu\u0161tum", "naruqqum"]


def doc_type(text: str) -> str:
    t = text.lower()
    scores = {k: sum(1 for pat in pats if pat in t) for k, pats in DOC_TYPE_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def topic_tags(text: str) -> list[str]:
    t = text.lower()
    tags = []
    for tag, pats in TOPIC_TAGS.items():
        if any(pat in t for pat in pats):
            tags.append(tag)
    return sorted(set(tags))


def institution_tags(text: str) -> list[str]:
    t = text.lower()
    hits = [inst for inst in INSTITUTIONS if inst in t]
    return sorted(set(hits))


def extract_social(text: str, pdf_name: str = "", lang_hint: str = "") -> Dict[str, Any]:
    t_lower = text.lower()
    name_lower = (pdf_name or "").lower()
    lang = (lang_hint or "").lower()

    # Highest precedence: index/bibliography/register
    INDEX_PDF_HINTS = ["index", "indices", "register", "bibliograph", "abk\u00fcrz", "abbreviat", "glossar", "lexicon"]
    if any(h in name_lower for h in INDEX_PDF_HINTS):
        return {"doc_type": "index", "topics": ["reference"], "institutions": []}
    # Textual clue for index/reference pages
    if t_lower.count(":") > 30 and (" p." in t_lower or " n." in t_lower):
        return {"doc_type": "index", "topics": ["reference"], "institutions": []}

    # Archeology report gate
    if (
        lang == "tr"
        or "arastirma" in name_lower
        or "ara\u015ft\u0131rma" in name_lower
        or "kayap" in name_lower
        or "sonu\u00e7lar\u0131" in name_lower
        or "k\u00fclt\u00fcr ve turizm bakanl\u0131\u011f\u0131" in t_lower
    ):
        return {"doc_type": "archaeology_report", "topics": ["archaeology"], "institutions": []}

    # Front matter
    if any(x in t_lower for x in ["digitized by", "issn", "printed", "contents"]):
        return {"doc_type": "front_matter", "topics": [], "institutions": []}

    # Letter / legal heuristics with commentary guard
    commentary_hints = ["iraq", "proceedings", "studies", "symposium", "fs ", "festschrift", "journal"]
    strong_letter = re.search(r"a-na\s+.+qi-bi-ma", t_lower) or re.search(r"from .+ to .+", t_lower)
    if ("um-ma" in t_lower and strong_letter) or strong_letter:
        doc = "letter"
    elif any(h in name_lower for h in commentary_hints):
        doc = "commentary"
    elif re.search(r"\bwitness\b", t_lower) or "seal" in t_lower or "igi" in t_lower:
        doc = "legal"
    else:
        doc = doc_type(text)

    return {"doc_type": doc, "topics": topic_tags(text), "institutions": institution_tags(text)}


DB_PATH = Path("data/corpus/corpus.db")
BATCH = int(os.environ.get("CORPUS_BATCH", "1000"))
MAX_PAGES = int(os.environ.get("CORPUS_MAX_PAGES", "0"))  # 0 = no limit


def main() -> None:
    conn = init_db(DB_PATH)
    processed = 0
    offset = 0

    while True:
        rows = fetch_pages_batch(DB_PATH, limit=BATCH, offset=offset, conn=conn)
        if not rows:
            break

        for page in rows:
            page_id = page["page_id"]
            text = page.get("text_norm") or ""
            payload = extract_social(
                text,
                pdf_name=page.get("pdf_name", ""),
                lang_hint=page.get("lang_hint", ""),
            )
            upsert_payload(DB_PATH, table="social", page_id=page_id, payload=payload, conn=conn)

            processed += 1
            if MAX_PAGES and processed >= MAX_PAGES:
                print(f"[extract_social] stopped early at {processed} (CORPUS_MAX_PAGES)")
                return

        offset += len(rows)
        conn.commit()  # commit per batch
        print(f"[extract_social] processed {processed}")


if __name__ == "__main__":
    main()
