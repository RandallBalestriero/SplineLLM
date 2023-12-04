

import re
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
)

from tqdm import tqdm

generation_params = {
    "do_sample": False,
    "max_new_tokens": 100,
}


def load_generative_model_and_tokenizer(
    model_name_or_path: str, model_name_tokenizer: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer


def generative_model_inference(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    generation_params: dict,
) -> str:

    gen_params = {k: v for k, v in generation_params.items() if k != "stop"}

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    logits = model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        **gen_params,
    )
    response = tokenizer.decode(logits[0].tolist()[len(input_ids[0]) :])
    return response
