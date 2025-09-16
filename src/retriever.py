#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, numpy as np, pandas as pd
import faiss
from numpy.linalg import norm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and build a combined text field.
    Expected columns: Title, Abstract, Keywords, Authors, Year, DOI
    """
    df = df.copy()
    for col in ["Title", "Abstract", "Keywords", "Authors", "Year", "DOI"]:
        if col not in df.columns:
            df[col] = ""
    df["text_full"] = df["Title"].fillna('') + ". " + df["Abstract"].fillna('') + " " + df["Keywords"].fillna('')
    return df

def chunk_by_tokens(text: str, tok, max_tok=220, stride=60):
    ids = tok.encode(text or "", add_special_tokens=False)
    chunks, step = [], max(1, max_tok - stride)
    for start in range(0, len(ids), step):
        piece = ids[start:start+max_tok]
        if not piece: break
        chunks.append(tok.decode(piece))
        if start + max_tok >= len(ids): break
    return chunks or [text or ""]

def make_chunks(df_corpus: pd.DataFrame, hf_tok, max_tok=220, stride=60) -> pd.DataFrame:
    records = []
    for i, r in df_corpus.iterrows():
        base = r["text_full"]
        chs = chunk_by_tokens(base, hf_tok, max_tok=max_tok, stride=stride)
        for j, ch in enumerate(chs):
            records.append({
                "doc_id": i,
                "chunk_id": j,
                "text": ch,
                "Title": r["Title"],
                "Abstract": r["Abstract"],
                "Keywords": r["Keywords"],
                "Authors": r["Authors"],
                "Year": r["Year"],
                "DOI": r["DOI"],
            })
    return pd.DataFrame(records)

def build_or_load_embeddings(df_chunks: pd.DataFrame, sbert: SentenceTransformer, cache_npy: str):
    if cache_npy and os.path.exists(cache_npy):
        vecs = np.load(cache_npy)
        vecs = l2_normalize(vecs)
        return vecs, True
    vecs = sbert.encode(df_chunks["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    vecs = l2_normalize(vecs)
    if cache_npy:
        os.makedirs(os.path.dirname(cache_npy), exist_ok=True)
        try: np.save(cache_npy, vecs)
        except Exception: pass
    return vecs, False

def build_ip_index(vecs: np.ndarray):
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index

def _first_author(authors: str) -> str:
    if not isinstance(authors, str): return ""
    s = authors.strip()
    if ";" in s: return s.split(";", 1)[0].strip()
    if " and " in s: return s.split(" and ", 1)[0].strip()
    return s

def _ref_header(row: dict) -> str:
    fa = _first_author(row.get("Authors", "") or "")
    title = (row.get("Title", "") or "")[:320]
    year = str(row.get("Year", "") or "").strip()
    doi = str(row.get("DOI", "") or "").strip()
    return f"{fa} — {title} ({year}). DOI: {doi}"

def mmr(query_vec, cand_vecs, cand_idx, k=8, lambda_=0.72):
    selected, selected_idx = [], []
    sims = (cand_vecs @ query_vec.T).ravel().copy()
    mask = np.ones_like(sims, dtype=bool)
    while len(selected) < min(k, len(cand_idx)):
        if not selected:
            i = int(np.argmax(sims))
            selected.append(cand_vecs[i]); selected_idx.append(cand_idx[i])
            mask[i] = False; sims[i] = -1e9
            continue
        sel_mat = np.stack(selected, axis=0)
        diversity = (cand_vecs @ sel_mat.T).max(axis=1)
        mmr_score = lambda_ * sims + (1 - lambda_) * (-diversity)
        mmr_score[~mask] = -1e9
        i = int(np.argmax(mmr_score))
        selected.append(cand_vecs[i]); selected_idx.append(cand_idx[i])
        mask[i] = False; sims[i] = -1e9
    return selected_idx

def rerank_with_crossencoder(query, rows, cross_encoder, batch_size=32):
    pairs = [(query, r["text"]) for r in rows]
    scores = []
    if not pairs: return rows
    parts = np.array_split(np.arange(len(pairs)), max(1, (len(pairs) + batch_size - 1) // batch_size))
    for idxs in parts:
        batch = [pairs[i] for i in idxs]
        scores.extend(cross_encoder.predict(batch))
    order = np.argsort(-np.array(scores))
    return [rows[i] for i in order]

def retrieve_rows(query: str,
                  faiss_index,
                  embeddings: np.ndarray,
                  df_chunks: pd.DataFrame,
                  sbert: SentenceTransformer,
                  hf_tok,
                  k_initial=40, k_final=8, mmr_lambda=0.72,
                  cross_encoder=None, ce_batch_size=32,
                  max_ctx_tokens=1400, title_max_chars=320, excerpt_max_chars=380):
    """
    Returns a list of row dicts (multiple chunks per document possible).
    Token budget is estimated using the new reference header + one snippet block.
    """
    qv = sbert.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims, idxs = faiss_index.search(qv, k_initial)
    idxs = idxs[0]
    cand_rows = [df_chunks.iloc[i].to_dict() for i in idxs]
    cand_vecs = embeddings[idxs]

    mmr_idx = mmr(qv[0], cand_vecs, idxs, k=max(k_final * 3, 12), lambda_=mmr_lambda)
    cand_rows = [df_chunks.iloc[i].to_dict() for i in mmr_idx]

    if cross_encoder is not None:
        cand_rows = rerank_with_crossencoder(query, cand_rows, cross_encoder, batch_size=ce_batch_size)

    packed, used = [], 0
    for r in cand_rows:
        header = _ref_header(r)
        excerpt = r.get("text", "") or ""
        if len(excerpt) > excerpt_max_chars:
            excerpt = excerpt[:excerpt_max_chars] + "…"
        block = f"{header}\nSnippet: {excerpt}\n"
        cost = len(hf_tok.encode(block))
        if used + cost > max_ctx_tokens:
            break
        packed.append(r)
        used += cost
    return packed

