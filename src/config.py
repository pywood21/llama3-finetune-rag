#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_id: str = "meta-llama/Meta-Llama-3-8B"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # use "bfloat16" if your GPU supports BF16
    device_map: str = "auto"
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False
    padding_side: str = "right"

@dataclass
class LoRAConfigLite:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class DataConfig:
    train_path: str = "<PATH-TO>/finetune_data.jsonl"  # user should provide a valid path
    text_field: str = "text"
    max_length: int = 1024
    add_eos_token: bool = True
    test_size: float = 0.1
    seed: int = 27
    num_proc: Optional[int] = None  # set e.g., 4 on Linux; keep None on Windows/Jupyter

@dataclass
class TrainConfig:
    output_dir: str = "outputs/llama3_finetuned"
    logging_dir: str = "outputs/logs"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 5
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True  # set bf16=True if supported and desired
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    report_to: str = "tensorboard"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    seed: int = 27

