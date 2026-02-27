from __future__ import annotations
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    # keep non-empty, short-trim
    out = [p.strip() for p in parts if p.strip()]
    return out
