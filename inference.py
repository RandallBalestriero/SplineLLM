import torch
import modeling_llama 
from transformers import AutoTokenizer
import transformers
from tqdm import tqdm
import os
import pickle
import numpy as np
import pandas as pd

from data_utils import get_dataset

import argparse


@torch.no_grad()
def forward(model, tokenizer, df_data, data_path_out, add_meta_data):
    polytope_stats = []
    raw_text = []
    inference_time = []
    model.eval()
    with torch.no_grad():
       for _, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Generating Llama2 Response"):
            prompt = row["prompt"]
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            output = model(input_ids=input_ids)
            raw_text.append(prompt)
            polytope_stats.append(output.regions_stats)
            inference_time.append(output.inference_time)
    output_path =os.path.join(data_path_out, 'statistics.csv')
    if add_meta_data:
        df = pd.DataFrame({'prompt': raw_text, 'stats': polytope_stats, 'inference_time': inference_time, 'toxic': df_data['toxic'], 
                           'severe_toxic': df_data['severe_toxic'], 'obscene': df_data['obscene'], 'threat': df_data['threat'], 
                           'insult': df_data['insult'], 'identity_hate': df_data['identity_hate']})  
    else:
        df = pd.DataFrame({'prompt': raw_text, 'stats': polytope_stats, 'inference_time': inference_time})
    df.to_csv(output_path, index=False)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
    )
    args = parser.parse_args()

    if args.model == 'meta':
        model_name ="meta-llama/Llama-2-7b-chat-hf" 
    elif args.model == 'mistral':
        model_name = "mistralai/Mistral-7B-v0.1"
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
    data_names = ['dollyQA' , "toxic_pile",'toxicity', 'full_jigsaw',  "FreeLaw","PubMed Abstracts", "DM Mathematics", "USPTO Backgrounds", "Github"] 
    

    for data_name in data_names:
        add_meta_data = False
        if data_name == 'full_jigsaw':
            add_meta_data = True
        data_path_out = f'/home/ubuntu/polytope/{data_name}'
        os.makedirs(data_path_out, exist_ok=True)
        dataset = get_dataset(data_name)
        forward(model, tokenizer, dataset, data_path_out, add_meta_data)








