from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

DB = Path("data/corpus/corpus.db")


def q(sql: str, params: Iterable = ()):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    con.close()
    return rows


def count(table: str) -> int:
    return q(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]


def count_nonempty_citations() -> int:
    return q(
        """
        SELECT COUNT(*) AS n
        FROM philology
        WHERE json NOT LIKE '%"citations":[]%'
          AND json NOT LIKE '%"citations": []%'
        """
    )[0]["n"]


def citation_stats(sample: int = 2000):
    rows = q("SELECT json FROM philology ORDER BY RANDOM() LIMIT ?", (sample,))
    import json as _json

    counts = []
    for r in rows:
        c = _json.loads(r["json"]).get("citations", [])
        counts.append(len(c))
    if not counts:
        print(f"[citation stats] sample=0 (no rows)")
        return
    counts.sort()
    p50 = counts[len(counts) // 2]
    idx90 = max(0, int(len(counts) * 0.9) - 1)
    p90 = counts[idx90]
    print(
        f"[citation stats] sample={len(counts)} "
        f"min={counts[0]} p50={p50} p90={p90} max={counts[-1]}"
    )


def snippet(text: str, term: str, window: int = 120) -> str:
    if not text:
        return ""
    tl = text.lower()
    idx = tl.find(term.lower())
    if idx < 0:
        return (text[:window] + "...") if len(text) > window else text
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def highlight(snippet_text: str, term: str) -> str:
    if not snippet_text or not term:
        return snippet_text
    return snippet_text.replace(term, f"<<{term}>>")


def sample_payload(table: str, limit: int = 5):
    rows = q(f"SELECT page_id, json FROM {table} ORDER BY RANDOM() LIMIT ?", (limit,))
    out = []
    for r in rows:
        try:
            out.append((r["page_id"], json.loads(r["json"])))
        except Exception:
            out.append((r["page_id"], {"_error": "payload not valid json"}))
    return out


def find_citations(limit: int = 10):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, ph.page_id, ph.json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        WHERE ph.json LIKE '%"citations"%'
          AND ph.json NOT LIKE '%"citations":[]%'
          AND ph.json NOT LIKE '%"citations": []%'
        LIMIT ?
        """,
        (limit,),
    )
    return rows


def find_term(term: str, limit: int = 10):
    rows = q(
        """
        SELECT pdf_name, page_number, page_id
        FROM page_registry
        WHERE text_norm LIKE ?
        LIMIT ?
        """,
        (f"%{term}%", limit),
    )
    return rows


def find_divine(name: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, ph.page_id, ph.json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        WHERE lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (f"%{name.lower()}%", limit),
    )
    return rows


def find_deity_with_citations(divine: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.text_norm, ph.json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        WHERE lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (f"%{divine.lower()}%", limit),
    )
    out = []
    for r in rows:
        payload = json.loads(r["json"])
        out.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "citations": payload.get("citations", []),
                "snippet": snippet(r["text_norm"] or "", divine),
            }
        )
    return out


def find_formula_hits(formula: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.text_norm, ph.json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        WHERE lower(ph.json) LIKE ?
        LIMIT ?
        """,
        (f"%{formula.lower()}%", limit),
    )
    out = []
    for r in rows:
        payload = json.loads(r["json"])
        out.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "snippet": snippet(r["text_norm"] or "", formula),
                "formula_snippets": payload.get("formula_snippets", [])[:3],
            }
        )
    return out


def find_institution_pages(inst: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, p.text_norm, ph.json AS ph_json, s.json AS s_json
        FROM social s
        JOIN page_registry p ON p.page_id = s.page_id
        LEFT JOIN philology ph ON ph.page_id = p.page_id
        WHERE lower(s.json) LIKE ?
        LIMIT ?
        """,
        (f"%{inst.lower()}%", limit),
    )
    out = []
    for r in rows:
        ph_payload = json.loads(r["ph_json"]) if r["ph_json"] else {}
        out.append(
            {
                "pdf_name": r["pdf_name"],
                "page_number": r["page_number"],
                "citations": ph_payload.get("citations", []),
                "snippet": snippet(r["text_norm"] or "", inst),
            }
        )
    return out


def find_formula(term: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, ph.json
        FROM philology ph
        JOIN page_registry p ON p.page_id = ph.page_id
        WHERE ph.json LIKE ?
        LIMIT ?
        """,
        (f"%{term}%", limit),
    )
    return rows


def count_doc_types_json(sample_limit: int = 0):
    import json as _json

    sql = "SELECT json FROM social"
    if sample_limit:
        rows = q(sql + " LIMIT ?", (sample_limit,))
    else:
        rows = q(sql)

    counts = {}
    for r in rows:
        payload = _json.loads(r["json"])
        dt = payload.get("doc_type", "missing")
        counts[dt] = counts.get(dt, 0) + 1
    return counts


def find_institution(inst: str, limit: int = 5):
    rows = q(
        """
        SELECT p.pdf_name, p.page_number, s.json
        FROM social s
        JOIN page_registry p ON p.page_id = s.page_id
        WHERE s.json LIKE ?
        LIMIT ?
        """,
        (f"%{inst}%", limit),
    )
    return rows


def main():
    print("[counts]")
    print("page_registry:", count("page_registry"))
    print("philology:", count("philology"))
    print("social:", count("social"))
    print("philology with citations:", count_nonempty_citations())
    citation_stats()

    print("\n[sample philology payloads]")
    for pid, payload in sample_payload("philology", 3):
        print(pid, list(payload.keys()))

    print("\n[sample social payloads]")
    for pid, payload in sample_payload("social", 3):
        print(pid, list(payload.keys()))

    print("\n[pages with citations detected]")
    rows = find_citations(5)
    print("found:", len(rows))
    for r in rows:
        payload = json.loads(r["json"])
        print(f"- {r['pdf_name']} p{r['page_number']} citations={payload.get('citations', [])[:5]}")

    print("\n[quick term search: Istar-ZA.AT]")
    rows = find_term("Istar-ZA.AT", 5)
    for r in rows:
        print(f"- {r['pdf_name']} p{r['page_number']} ({r['page_id']})")

    print("\n[pages with divine Istar-ZA.AT]")
    rows = find_divine("istarzaat", 5)
    for r in rows:
        payload = json.loads(r["json"])
        print(f"- {r['pdf_name']} p{r['page_number']} divine={payload.get('divine_names', [])}")

    print("\n[killer query: deity with citations] istarzaat")
    for r in find_deity_with_citations("istarzaat", 5):
        print(f"- {r['pdf_name']} p{r['page_number']} citations={r['citations'][:3]} snippet={r['snippet'][:140]}")

    print("\n[killer query: formula um-ma]")
    for r in find_formula_hits("um-ma", 5):
        print(f"- {r['pdf_name']} p{r['page_number']} snippet={r['snippet'][:140]} formulas={r['formula_snippets']}")

    print("\n[killer query: institution naruqqum]")
    for r in find_institution_pages("naruqqum", 5):
        print(f"- {r['pdf_name']} p{r['page_number']} citations={r['citations'][:3]} snippet={r['snippet'][:140]}")

    print("\n[pages with formula um-ma]")
    for r in find_formula("um-ma", 5):
        payload = json.loads(r["json"])
        snips = payload.get("formula_snippets", [])
        frags = payload.get("formula_markers", []) or payload.get("formula_fragments", [])
        print("-", r["pdf_name"], "p" + str(r["page_number"]), "fragments=", frags[:10])
        if snips:
            print("  snippet:", highlight(snips[0][:200], "um-ma"))

    print("\n[pages with formula li-tù-la]")
    for r in find_formula("li-tù-la", 5):
        payload = json.loads(r["json"])
        snips = payload.get("formula_snippets", [])
        frags = payload.get("formula_markers", []) or payload.get("formula_fragments", [])
        print("-", r["pdf_name"], "p" + str(r["page_number"]), "fragments=", frags[:10])
        if snips:
            print("  snippet:", highlight(snips[0][:200], "li-tù-la"))

    print("\n[doc_type counts]")
    counts = count_doc_types_json(sample_limit=50000)
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        print(k, ":", counts[k])

    print("\n[pages tagged institution naruqqum]")
    for r in find_institution("naruqqum", 5):
        payload = json.loads(r["json"])
        print("-", r["pdf_name"], "p" + str(r["page_number"]), "doc_type=", payload.get("doc_type"), "institutions=", payload.get("institutions", [])[:10])

    print("\n[pages tagged institution karum]")
    for r in find_institution("karum", 5):
        payload = json.loads(r["json"])
        print("-", r["pdf_name"], "p" + str(r["page_number"]), "doc_type=", payload.get("doc_type"), "institutions=", payload.get("institutions", [])[:10])


if __name__ == "__main__":
    main()
