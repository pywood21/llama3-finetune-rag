#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore

def cosine_similarity_sbert(refs, cands, model_name: str, device: str):
    """
    Compute cosine similarity between reference and candidate texts using a SentenceTransformer.
    Returns a list of cosine similarities for aligned ref/cand pairs.
    """
    sbert = SentenceTransformer(model_name, device=device)
    emb_ref = sbert.encode(refs, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    emb_cand = sbert.encode(cands, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return np.sum(emb_ref * emb_cand, axis=1).tolist()

def compute_bertscore(refs, cands, lang="en"):
    """
    Compute BERTScore (Precision/Recall/F1). Uses 'lang' to select the default model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = bertscore(cands, refs, lang=lang, device=device, verbose=False)
    return P.tolist(), R.tolist(), F1.tolist()

# ---------- Perplexity helpers ----------
@torch.no_grad()
def _sequence_nll(model, tokenizer, text: str, max_ctx: int) -> tuple[float, int]:
    """
    Compute total negative log-likelihood (sum of token-level NLL) and token count for one text.
    Uses a sliding window to handle long sequences. Returns (nll_sum, token_count).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0

    # Tokenize without truncation; we'll chunk manually
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].squeeze(0)
    if input_ids.numel() == 0:
        return 0.0, 0

    # Use model max context if available; otherwise tokenizer limit or a safe default
    context = max_ctx if max_ctx is not None else getattr(model.config, "max_position_embeddings", None)
    if context is None or context <= 0 or context > 32768:
        context = min(getattr(tokenizer, "model_max_length", 8192), 8192)
    # we need at least 2 tokens to compute next-token loss
    window = max(2, min(int(context), input_ids.numel()))

    nll_sum = 0.0
    tok_count = 0

    # Simple sliding windows without overlap for efficiency.
    # If you want higher-fidelity context, you can add overlap (stride < window).
    for start in range(0, input_ids.numel(), window):
        end = min(start + window, input_ids.numel())
        ids = input_ids[start:end]

        # For causal LM NLL: shift inputs by one and predict next token.
        # We compute loss over all positions except the first in the window.
        labels = ids.clone()
        # Mask the first position so that there is no target for it
        labels[0] = -100

        ids = ids.unsqueeze(0).to(model.device)
        labels = labels.unsqueeze(0).to(model.device)

        out = model(input_ids=ids, labels=labels)
        # out.loss is mean over non -100 labels; scale back to token count to aggregate properly
        valid = (labels != -100).sum().item()
        if valid > 0:
            nll_sum += float(out.loss.item()) * valid
            tok_count += valid

    return nll_sum, tok_count

def compute_perplexities(model, tokenizer, texts: list[str], max_ctx: int | None = None) -> list[float]:
    """
    Compute per-text perplexity = exp(total_nll / total_tokens). Empty/invalid texts get NaN.
    """
    ppls = []
    for t in texts:
        nll, cnt = _sequence_nll(model, tokenizer, t, max_ctx=max_ctx)
        if cnt == 0:
            ppls.append(float("nan"))
        else:
            ppls.append(math.exp(nll / cnt))
    return ppls

# ---------- Main entry ----------
def evaluate_text_metrics(
    df: pd.DataFrame,
    eval_sbert_name: str,
    *,
    # Perplexity options:
    ppl_model=None,
    ppl_tokenizer=None,
    ppl_column: str | None = None,
    ppl_max_ctx: int | None = None,
):
    """
    Evaluate generation quality with:
      - CosineSimilarity (SBERT)
      - BERTScore (Precision/Recall/F1)
      - (Optional) Perplexity using a provided causal LM (on a chosen column)
        * Provide ppl_model and ppl_tokenizer (loaded HF model/tokenizer)
        * Set ppl_column to "Answer" or "Generated" (the column to score)
        * ppl_max_ctx optionally limits model context when scoring long texts

    BLEU and ROUGE-L are intentionally omitted.
    """
    df = df.copy()
    for col in ["Answer", "Generated"]:
        if col not in df.columns:
            df[col] = ""
    df["Answer"] = df["Answer"].fillna("").astype(str)
    df["Generated"] = df["Generated"].fillna("").astype(str)

    eval_df = df[(df["Generated"].str.strip() != "") & (df["Generated"] != "[ERROR]")].copy()
    if eval_df.empty:
        eval_df = df.copy()

    refs = eval_df["Answer"].tolist()
    cands = eval_df["Generated"].tolist()

    # (1) Cosine similarity
    cos_sims = cosine_similarity_sbert(
        refs, cands, eval_sbert_name,
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )

    # (2) BERTScore
    P, R, F1 = compute_bertscore(refs, cands, lang="en")

    # Attach metrics to eval subset
    eval_df.loc[:, "CosineSimilarity"] = cos_sims
    eval_df.loc[:, "BERTScore_Precision"] = P
    eval_df.loc[:, "BERTScore_Recall"] = R
    eval_df.loc[:, "BERTScore_F1"] = F1

    # (3) Optional Perplexity
    ppl_col_name = None
    if ppl_model is not None and ppl_tokenizer is not None and ppl_column in ("Answer", "Generated"):
        texts = eval_df[ppl_column].tolist()
        ppls = compute_perplexities(ppl_model, ppl_tokenizer, texts, max_ctx=ppl_max_ctx)
        ppl_col_name = f"Perplexity_{ppl_column}"
        eval_df.loc[:, ppl_col_name] = ppls

    # Merge back to original shape, leaving non-evaluated rows as NaN for metrics
    metric_cols = ["CosineSimilarity", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1"]
    if ppl_col_name:
        metric_cols.append(ppl_col_name)

    out = df.copy()
    for col in metric_cols:
        out[col] = np.nan
    out.loc[eval_df.index, metric_cols] = eval_df[metric_cols].values
    return out

