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



@torch.no_grad()
def forward(model, tokenizer, df_data, data_path_out):
    polytope_stats = []
    raw_text = []
    model.eval()
    with torch.no_grad():
       for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Generating Llama2 Response"):
            if idx>99:
                break
            prompt = row["prompt"]
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            output = model(input_ids=input_ids)
            raw_text.append(prompt)
            polytope_stats.append(output.regions_stats)
    output_path =os.path.join(data_path_out, 'imshow_statistics.csv')
    df = pd.DataFrame({'prompt': raw_text, 'stats': polytope_stats})
    df.to_csv(output_path, index=False)
    
    
    
if __name__ == "__main__":
    model = modeling_llama.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
        
    data_names = ['toxic_pile'] #['jigsaw_insult', "toxic_pile", "toxicity", "knowledge", 'hotel', "FreeLaw","PubMed Abstracts", "DM Mathematics", "USPTO Backgrounds", "Github"] #"toxicity", "knowledge", 'hotel',
    

    for data_name in data_names:
        data_path_out = f'/home/ubuntu/polytope_hyperplane_side/{data_name}'
        os.makedirs(data_path_out, exist_ok=True)
        dataset = get_dataset(data_name)
        forward(model, tokenizer, dataset, data_path_out)








