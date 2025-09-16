#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
from typing import List, Tuple, Dict

SYSTEM_EN = (
    "You are a rigorous assistant for academic use. "
    "Always answer in English. Use the provided context for grounding; "
    "be concise and avoid copying long spans verbatim. "
    "Cite context items as [n], where each [n] corresponds to a numbered reference entry. "
    "If the context is insufficient, clearly say so."
)

def _first_author(authors: str) -> str:
    """
    Extract the first author robustly:
    - Split by ';' or ' and ' which commonly separate authors.
    - Avoid splitting by ',' because many formats use 'Last, First'.
    """
    if not isinstance(authors, str):
        return ""
    s = authors.strip()
    if ";" in s:
        return s.split(";", 1)[0].strip()
    if " and " in s:
        return s.split(" and ", 1)[0].strip()
    # Fallback: return as-is (may include full 'Last, First' for the first author)
    return s

def _format_citation(row: Dict) -> str:
    """
    Build a compact reference header:
    [n] FirstAuthor — Title (Year). DOI: <doi>
    """
    fa = _first_author(row.get("Authors", "") or "")
    title = (row.get("Title", "") or "")[:320]
    year = str(row.get("Year", "") or "").strip()
    doi = str(row.get("DOI", "") or "").strip()
    return f"{fa} — {title} ({year}). DOI: {doi}"

def build_context_block(rows: List[Dict]) -> Tuple[str, Dict[int, int]]:
    """
    Group rows by document (doc_id) so each [n] refers to one document.
    Each entry shows a single reference header and up to two short snippets.
    Returns:
      context_str: human-readable context block
      id_map: mapping from [n] -> doc_id
    """
    grouped: Dict[int, Dict] = {}
    for r in rows:
        did = int(r.get("doc_id", -1))
        if did not in grouped:
            grouped[did] = {"header": _format_citation(r), "snippets": []}
        txt = r.get("text", "") or ""
        if len(txt) > 380:
            txt = txt[:380] + "…"
        grouped[did]["snippets"].append(txt)

    lines, id_map = [], {}
    dids = list(grouped.keys())
    for i, did in enumerate(dids, 1):
        g = grouped[did]
        snippets = g["snippets"][:2]
        line = f"[{i}] {g['header']}\n    - " + "\n    - ".join(snippets)
        lines.append(line)
        id_map[i] = did
    return "\n".join(lines), id_map

def build_messages(query: str, rows=None):
    rows = rows or []
    if rows:
        ctx, _ = build_context_block(rows)
        user = "### Context\n" + ctx + f"\n\n### Question\n{query}\n\n### Answer\n"
    else:
        user = f"### Question\n{query}\n\n### Answer\n"
    return [
        {"role": "system", "content": SYSTEM_EN},
        {"role": "user", "content": user},
    ]

def render_chat_prompt(tokenizer, messages):
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        if not tmpl:
            return None
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

def build_fallback_prompt(query: str, rows=None):
    rows = rows or []
    if rows:
        ctx, _ = build_context_block(rows)
        return f"{SYSTEM_EN}\n\n### Context\n{ctx}\n\n### Question\n{query}\n\n### Answer\n"
    else:
        return f"{SYSTEM_EN}\n\n### Question\n{query}\n\n### Answer\n"

