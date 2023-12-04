from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
import datasets
import torch
from typing import Dict, List
from collections import defaultdict


torch.set_grad_enabled(False)

if __name__ == "__main__":
    category = "asian_2"
    dataset_file_path = f"/home/ubuntu/polytope/toxicity_id_eval/{category}.csv"
    raw_dataset = datasets.load_dataset(
        "csv", data_files=dataset_file_path, split="train"
    )
    # raw_dataset = raw_dataset.remove_columns(["ID", "Unnamed: 5", "Unnamed: 6"])  #

    ## Use the below line after first
    #raw_dataset = raw_dataset.remove_columns(["ID", "Context Length"])  #

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    context_lengths = []
    outputs_attn = []
    for data in raw_dataset:
        tokenized_prompt = tokenizer(data["Prompt"], return_tensors="pt")
        outputs = model(**tokenized_prompt)
        context_lengths.append(tokenized_prompt.input_ids.numel())
        outputs_attn.append(outputs.attentions)

    raw_dataset = raw_dataset.add_column("ID", outputs_attn)
    raw_dataset = raw_dataset.add_column("Context Length", context_lengths)
    raw_dataset.to_csv(dataset_file_path)
