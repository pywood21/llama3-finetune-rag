#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def prefer_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

def load_backbone(model_id: str, load_in_4bit: bool, use_bf16_if_available: bool):
    dtype = torch.bfloat16 if (use_bf16_if_available and prefer_bf16()) else torch.float16
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    return model, tok

def merge_lora(base_model, lora_adapter_path: str):
    ft = PeftModel.from_pretrained(base_model, lora_adapter_path)
    try:
        ft = ft.merge_and_unload(safe_merge=True)
    except TypeError:
        ft = ft.merge_and_unload()
    ft.eval()
    return ft

