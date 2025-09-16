#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora(model, lora_cfg) -> torch.nn.Module:
    """
    Prepare a quantized model and apply LoRA adapters.
    """
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    model = get_peft_model(model, peft_cfg)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model

