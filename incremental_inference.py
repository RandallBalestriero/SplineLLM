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

    polytope_sequence = []
    tokens_sequence = []
    data_idx = []
    model.eval()
    
    with torch.no_grad():
       for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Generating Llama2 Response"):
            if idx>10:
                break
            prompt = row["prompt"]
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            input_len = input_ids.size(-1)
            for k in range(1, input_len):
                tokens = []
                input_ = input_ids[0][:k].reshape(1,-1)
                output = model(input_ids=input_)
                for token in input_ids[0][:k]:
                    tokens.append(tokenizer.decode(token))
                polytope_sequence.append(output.regions_stats)
                tokens_sequence.append(tokens)
                data_idx.append(idx)
                
    output_path =os.path.join(data_path_out, 'incemental_stats.csv')
    df = pd.DataFrame({'sample_id': data_idx, 'tokens': tokens_sequence, 'stats': polytope_sequence})
    df.to_csv(output_path, index=False)
    
    
    
if __name__ == "__main__":
    model = modeling_llama.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
        
    data_names = ['toxic_uspto', 'toxic_pubmeb', 'toxic_github', 'toxic_freelaw', 'toxic_maths'] #['jigsaw_insult', "toxic_pile", "toxicity", "knowledge", 'hotel', "FreeLaw","PubMed Abstracts", "DM Mathematics", "USPTO Backgrounds", "Github"] #"toxicity", "knowledge", 'hotel',
    
    for data_name in data_names:
        print('DATA NAME:', data_name)
        data_path_out = f'/home/ubuntu/polytope_incremental/{data_name}'
        os.makedirs(data_path_out, exist_ok=True)
        dataset = get_dataset(data_name)
        forward(model, tokenizer, dataset, data_path_out)








