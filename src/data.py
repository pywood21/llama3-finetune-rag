#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Dict
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
import platform

PROMPT_TPL_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

PROMPT_TPL_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

def _format_prompt(example: Dict, add_eos: bool, eos_token: str) -> Dict:
    """
    Convert a record with keys {instruction, input, output} into a single text field.
    """
    instruction = str(example.get("instruction", "")).replace("###", "### ")
    output = str(example.get("output", ""))
    input_text = str(example.get("input", "")).strip()

    if input_text:
        text = PROMPT_TPL_WITH_INPUT.format(instruction=instruction, input=input_text, output=output)
    else:
        text = PROMPT_TPL_NO_INPUT.format(instruction=instruction, output=output)

    if add_eos and eos_token and not text.endswith(eos_token):
        text = text + eos_token
    return {"text": text}

def _tokenize(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_attention_mask=True,
        )
    return _fn

def load_and_prepare_dataset(data_cfg, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    """
    Load JSONL dataset and return tokenized train/test splits.
    The JSONL file should contain fields: instruction, input (optional), output.
    """
    ds = load_dataset("json", data_files=data_cfg.train_path, split="train")
    use_proc = data_cfg.num_proc if (data_cfg.num_proc and platform.system() != "Windows") else None

    ds = ds.map(
        lambda ex: _format_prompt(ex, data_cfg.add_eos_token, tokenizer.eos_token or ""),
        num_proc=use_proc,
    )

    tokenized = ds.map(
        _tokenize(tokenizer, data_cfg.max_length),
        batched=True,
        num_proc=use_proc,
        remove_columns=ds.column_names,
    )

    split = tokenized.train_test_split(test_size=data_cfg.test_size, seed=data_cfg.seed)
    return DatasetDict(train=split["train"], test=split["test"])

