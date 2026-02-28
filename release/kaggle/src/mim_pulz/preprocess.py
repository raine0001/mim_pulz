from __future__ import annotations
import re

_whitespace = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    s = _whitespace.sub(" ", s)
    return s
