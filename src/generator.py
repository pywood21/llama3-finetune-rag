#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re, time, torch
from .prompt import build_messages, render_chat_prompt, build_fallback_prompt

RESP_SPLIT_RE = re.compile(r"###\s*Answer[:]?\s*", flags=re.IGNORECASE)

def post_clean(raw_text: str) -> str:
    parts = RESP_SPLIT_RE.split(raw_text, maxsplit=1)
    response = parts[-1] if len(parts) > 1 else raw_text
    response = " ".join(response.strip().split())
    return response.strip()

def build_gen_kwargs(tokenizer, max_new=512, temperature=0.2, top_p=0.9, repetition_penalty=1.15, do_sample=True):
    return dict(
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )

def ensure_ctx_budget(tokenizer, prompt_text: str, model_max_ctx: int, max_new_tokens: int):
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
    if len(ids) + max_new_tokens > model_max_ctx:
        return max(64, model_max_ctx - len(ids) - 16)
    return max_new_tokens

def answer_query(query: str, model, tokenizer, rows=None, gen_cfg=None):
    messages = build_messages(query, rows=rows)
    prompt_text = render_chat_prompt(tokenizer, messages) or build_fallback_prompt(query, rows=rows)
    max_new = ensure_ctx_budget(tokenizer, prompt_text, gen_cfg.model_max_ctx, gen_cfg.max_new_tokens)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=gen_cfg.model_max_ctx - max_new).to(model.device)
    kwargs = build_gen_kwargs(tokenizer, max_new=max_new, temperature=gen_cfg.temperature, top_p=gen_cfg.top_p,
                              repetition_penalty=gen_cfg.repetition_penalty, do_sample=gen_cfg.do_sample)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return post_clean(text)

def safe_generate(query: str, model, tokenizer, retriever_fn, use_rag: bool, gen_cfg, retry_sleep=0.03):
    tries = [
        dict(use_rag_flag=use_rag, k=3, max_ctx_tokens=getattr(gen_cfg, "max_ctx_tokens", 1400), max_new=gen_cfg.max_new_tokens, do_sample=True),
        dict(use_rag_flag=use_rag, k=2, max_ctx_tokens=1200, max_new=min(384, gen_cfg.max_new_tokens), do_sample=True),
        dict(use_rag_flag=use_rag, k=2, max_ctx_tokens=1200, max_new=min(384, gen_cfg.max_new_tokens), do_sample=False),
        dict(use_rag_flag=False, k=0, max_ctx_tokens=0, max_new=320, do_sample=False),
    ]
    last_err = None
    for t in tries:
        try:
            rows = retriever_fn(query, k_final=t["k"], max_ctx_tokens=t["max_ctx_tokens"]) if t["use_rag_flag"] else None
            return answer_query(query, model, tokenizer, rows=rows, gen_cfg=gen_cfg), {"status": "ok", "use_rag": t["use_rag_flag"]}
        except Exception as e:
            last_err = str(e)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            time.sleep(retry_sleep)
    return "[ERROR]", {"status": "fail", "error": last_err}

