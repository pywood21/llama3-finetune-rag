#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataclasses import dataclass
from typing import Optional

@dataclass
class Paths:
    corpus_csv: str = "<PATH-TO>/dataset_all.csv"        # must include at least: Title, Abstract, Keywords, Authors, Year, DOI
    qa_csv: str      = "<PATH-TO>/qa_set.csv"            # must include: Question (Answer optional)
    output_dir: str  = "outputs"
    emb_cache_npy: str = "outputs/faiss_mpnet_embeds_chunks.npy"
    faiss_index_path: Optional[str] = None

@dataclass
class Models:
    base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_adapter_path: str = "<PATH-TO>/checkpoint-XXXX"
    retriever_sbert: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    eval_sbert: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    cross_encoder: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

@dataclass
class RetrieverCfg:
    chunk_max_tok: int = 220
    chunk_stride: int = 60
    k_initial: int = 40
    k_final: int = 8
    mmr_lambda: float = 0.72
    ce_batch_size: int = 32
    max_ctx_tokens: int = 1400
    title_max_chars: int = 320
    excerpt_max_chars: int = 380

@dataclass
class GenCfg:
    model_max_ctx: int = 8192
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True

@dataclass
class Flags:
    load_in_4bit: bool = True
    use_bf16_if_available: bool = True

@dataclass
class Experiment:
    selected_model: str = "base_rag"   # choices: "base", "ft", "base_rag", "ft_rag"
    seed: int = 42

