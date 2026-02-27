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

# Citation patterns (ordered by specificity/priority)
CITATION_PATTERNS = [
    # Full: SERIES ROMAN NUMBER (optional letter)
    r"\b(?:BIN|CCT|ICK|VS|EL|TCL|KTK)\s+[IVXLC]+\s+\d+[a-z]?\b",
    # Series that commonly appear as SERIES NUMBER (+ optional letter)
    r"\b(?:AKT|ATHE|JEOL|JCS|AfO|PIHANS|OAAS|CNIP|OBO|CHANE|CRRAI)\s+\d+[a-z]?\b",
    # Kt and kt references (very common)
    r"\bkt\s+[a-z0-9]+/[a-z0-9]+\s*\d+[a-z]?\b",     # kt n/k 1340
    r"\bkt\s+\d{1,3}/k\s*\d+[a-z]?\b",               # kt 94/k 1146
    r"\bKt\s+\d{1,3}[a-z]?\s*[a-z]\s*\d+\b",         # Kt 83k 246 (OCR variants)
]

BROKEN_CITATION_RE = re.compile(
    r"\b(?:BIN|CCT|ICK|VS|EL|TCL|KTK)\s+[IVXLC]+\s*\n\s*\d+[a-z]?\b",
    re.IGNORECASE,
)

DIVINE_NAMES = [
    "assur",
    "istar",
    "samas",
    "nabu",
    "ninlil",
    "marduk",
]

DIVINE_PATTERNS = [
    r"a-sur",
    r"i(?:s|š)tar",
    r"sa-ma(?:s|š)",
    r"an?-?na",
    r"dingir",
    r"i(?:s|š)tar[-\s]?za\.?\.?at",
    r"a[sš]sur",
    r"samas",
    r"shamash",
]

DIVINE_STOPWORDS = {
    "ana",
    "a-na",
    "ina",
    "i-na",
    "u",
    "ù",
    "ša",
    "sa",
    "lu",
    "li",
    "dumu",
    "igi",
    "kù",
    "babbar",
    "ma-na",
    "gin",
}

DIVINE_WHITELIST = {
    "assur",
    "ashur",
    "istar",
    "ishtar",
    "istarzaat",
    "ishtarzaat",
    "shamash",
    "samas",
    "nabu",
    "nabû",
    "enlil",
    "ellil",
    "sin",
    "adad",
}

FORMULA_MARKERS = {
    "um-ma": r"\bum-ma\b",
    "li-tù-la": r"li-t[uù]-la",
    "li-di-a-ni": r"li-di-a-ni",
    "assur": r"\ba-?s[uù]r\b",
    "istar": r"\bi[šs]tar\b",
}

LEGAL_MARKERS = {
    "igi": r"\bigi\b",
    "dumu": r"\bdumu\b",
    "oath": r"\boath\b",
    "swear": r"\bswear\b",
    "tamu": r"tam[uû]",
}

ACCOUNTING_MARKERS = {
    "ku.babbar": r"ku\.?babbar",
    "ma-na": r"ma-?na",
    "gin": r"\bgin\b",
}


def extract_philology(text: str) -> Dict[str, Any]:
    raw_text = text or ""
    flat_text = re.sub(r"\s+", " ", raw_text)

    citations = []
    for pat in CITATION_PATTERNS:
        citations.extend(re.findall(pat, flat_text, flags=re.IGNORECASE))
    citations.extend(BROKEN_CITATION_RE.findall(raw_text))

    def _norm_citation(c: str) -> str:
        c = re.sub(r"\s+", " ", c.strip())
        # Standardize leading token
        c = re.sub(r"^\bkt\b", "Kt", c, flags=re.IGNORECASE)
        # Normalize slash segment
        c = c.replace("/K", "/k")
        # Normalize letter around slash: C/K -> c/k
        c = re.sub(r"\b([A-Za-z])\s*/\s*k\b", lambda m: f"{m.group(1).lower()}/k", c)
        # Normalize Kt C/K 1615 -> Kt c/k 1615
        c = re.sub(r"\bKt\s+([A-Za-z])\s*/\s*k\s+", lambda m: f"Kt {m.group(1).lower()}/k ", c)
        # Common OCR confusions in citation numbers
        c = re.sub(r"\b([0-9])g\b", r"\g<1>9", c)  # 2g -> 29, 3g -> 39
        c = re.sub(r"\b([0-9])l\b", r"\g<1>1", c)  # l -> 1 (rare, cautious)
        return c

    citations = [_norm_citation(c) for c in citations if c]

    def _norm_divine(tok: str) -> str:
        t = tok.lower().strip()
        t = t.replace("š", "s").replace("ṣ", "s").replace("ṭ", "t")
        t = t.replace("ā", "a").replace("ī", "i").replace("ū", "u").replace("ù", "u")
        for ch in ["’", "'", ".", " ", "-"]:
            t = t.replace(ch, "")
        return t

    divs_raw = []
    lowered = raw_text.lower()
    for name in DIVINE_NAMES:
        if name in lowered:
            divs_raw.append(name)
    for pat in DIVINE_PATTERNS:
        divs_raw.extend(re.findall(pat, lowered))

    divs = []
    for d in divs_raw:
        n = _norm_divine(d)
        if not n or n in DIVINE_STOPWORDS:
            continue
        if n in DIVINE_WHITELIST:
            divs.append(n)

    def _snippet(text: str, start: int, end: int, window: int = 80) -> str:
        s = max(0, start - window)
        e = min(len(text), end + window)
        prefix = "..." if s > 0 else ""
        suffix = "..." if e < len(text) else ""
        return prefix + text[s:e] + suffix

    formula_markers = []
    formula_snippets = []
    for label, pat in FORMULA_MARKERS.items():
        for m in re.finditer(pat, raw_text, flags=re.IGNORECASE):
            formula_markers.append(label)
            formula_snippets.append(f"{label}: {_snippet(raw_text, m.start(), m.end())}")

    legal_markers = []
    for label, pat in LEGAL_MARKERS.items():
        if re.search(pat, raw_text, flags=re.IGNORECASE):
            legal_markers.append(label)

    accounting_markers = []
    for label, pat in ACCOUNTING_MARKERS.items():
        if re.search(pat, raw_text, flags=re.IGNORECASE):
            accounting_markers.append(label)

    def _norm_ascii(s: str) -> str:
        import unicodedata

        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()

    return {
        "citations": sorted(set(citations)),
        "divine_names": sorted(set(divs)),
        "formula_markers": sorted(set(formula_markers)),
        "formula_snippets": formula_snippets,
        "legal_markers": sorted(set(legal_markers)),
        "accounting_markers": sorted(set(accounting_markers)),
        "formula_markers_norm": sorted({_norm_ascii(m) for m in formula_markers}),
        "legal_markers_norm": sorted({_norm_ascii(m) for m in legal_markers}),
        "accounting_markers_norm": sorted({_norm_ascii(m) for m in accounting_markers}),
    }


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
            payload = extract_philology(text)
            upsert_payload(DB_PATH, table="philology", page_id=page_id, payload=payload, conn=conn)

            processed += 1
            if MAX_PAGES and processed >= MAX_PAGES:
                print(f"[extract_philology] stopped early at {processed} (CORPUS_MAX_PAGES)")
                return

        offset += len(rows)
        conn.commit()  # commit per batch
        print(f"[extract_philology] processed {processed}")


if __name__ == "__main__":
    main()
