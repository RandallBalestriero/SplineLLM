import pandas as pd
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
from statistic_analysis import load_data
from typing import Tuple, List


def get_dataset(data_name):
    if "toxicity" == data_name:
        data_path = (
            "/home/ubuntu/finetuning_benchmark_datasets_large/toxicity_dataset.csv"
        )
        return pd.read_csv(data_path, nrows=20000)

    elif "knowledge" == data_name:
        data_path = (
            "/home/ubuntu/finetuning_benchmark_datasets_large/forgetting_eval.csv"
        )
        return pd.read_csv(data_path, nrows=20000)

    elif "dollyQA" == data_name:
        dataset = load_dataset("databricks/databricks-dolly-15k")
        return pd.DataFrame({"prompt": dataset["train"]["instruction"]})

    elif "hotel" == data_name:
        data_path = (
            "/home/ubuntu/finetuning_benchmark_datasets_large/hotel_data_10k_train.csv"
        )
        return pd.read_csv(data_path, nrows=20000)

    elif "toxic_pile" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_pile.csv"
        return pd.read_csv(data_path)

    elif "toxic_uspto" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_USPTO Backgrounds.csv"
        return pd.read_csv(data_path)

    elif "toxic_pubmeb" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_PubMed Abstracts.csv"
        return pd.read_csv(data_path)

    elif "toxic_github" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_Github.csv"
        return pd.read_csv(data_path)

    elif "toxic_freelaw" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_FreeLaw.csv"
        return pd.read_csv(data_path)

    elif "toxic_maths" == data_name:
        data_path = (
            "/home/ubuntu/finetuning_benchmark_datasets_large/toxic_DM Mathematics.csv"
        )
        return pd.read_csv(data_path)

    elif "jigsaw_insult" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/jigsaw-toxic-comment-train-processed-seqlen128.csv"
        data = pd.read_csv(data_path)
        data = data.rename(columns={"comment_text": "prompt"})
        return data[data["insult"] == 1][:20000]

    elif "full_jigsaw" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/jigsaw-toxic-comment-train-processed-seqlen128.csv"
        data = pd.read_csv(data_path)
        data = data.rename(columns={"comment_text": "prompt"})
        return data
    
    elif "jigsaw_testset" == data_name:
        data_path = "/home/ubuntu/finetuning_benchmark_datasets_large/testset_jigsaw.csv"
        data = pd.read_csv(data_path)
        data = data.rename(columns={"comment_text": "prompt"})
        return data
    else:
        data_pile = load_pile(data_name)
        return pd.DataFrame(data_pile, columns=["prompt"])


def load_pile(meta_data):
    raw_dataset = load_dataset(
        "monology/pile-uncopyrighted", "default", split="validation[:20000]"
    )
    raw_dataset = raw_dataset.filter(
        lambda example: example["meta"]["pile_set_name"] == meta_data
    )
    return raw_dataset["text"]


def get_toxic_clean_data_benchmarch() -> Tuple[List[str], List[str]]:
    data_names_clean = [
        "DM Mathematics",
        "dollyQA",
        "FreeLaw",
        "Github",
        "PubMed Abstracts",
        "USPTO Backgrounds",
    ]
    data_names_toxic = ["toxicity", "toxic_pile"]

    jigsaw_path = "/home/ubuntu/polytope/full_jigsaw/statistics.csv"
    clean_data = []  # list(load_data(jigsaw_path, jigsaw_subset= "clean")[1])
    toxic_data = list(load_data(jigsaw_path, jigsaw_subset="toxic")[1])


    for name in data_names_clean:
        clean_data += list(
            pd.read_csv(f"/home/ubuntu/polytope/{name}/statistics.csv")["prompt"].values
        )
    for name in data_names_toxic:
        toxic_data += list(
            pd.read_csv(f"/home/ubuntu/polytope/{name}/statistics.csv")["prompt"].values
        )
    return toxic_data, clean_data


def mix_pile_toxic():
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", use_fast=True
    )

    dataset_toxic = get_dataset("toxicity")
    n_toxic_samples = len(dataset_toxic["prompt"])
    data_names = [
        "FreeLaw",
        "PubMed Abstracts",
        "DM Mathematics",
        "USPTO Backgrounds",
        "Github",
    ]
    for data_name in data_names:
        dataset_list = []
        print(f"Dataset Name: {data_name}")
        df_data = get_dataset(data_name)
        print(df_data)

        for _, row in tqdm(
            df_data.iterrows(), total=len(df_data), desc="Generating Llama2 Response"
        ):
            idx_sample_toxic = np.random.randint(n_toxic_samples)
            toxic_sample = dataset_toxic["prompt"][idx_sample_toxic]
            tokenized_toxic_sample = tokenizer.encode(
                toxic_sample, return_tensors="pt", max_length=1024, truncation=True
            )[0]
            len_toxic_sample = len(tokenized_toxic_sample)

            # adding black space after each sentence
            max_pos_toxic_sample = 1024 - (len_toxic_sample + 1)
            sample_position_toxic = np.random.randint(max_pos_toxic_sample)
            input_ids = tokenizer.encode(
                row["prompt"], return_tensors="pt", max_length=1024, truncation=True
            )

            # adding to random position toxic_sentence in the tokenized format
            mix_input_sample = torch.cat(
                [
                    input_ids[0][:sample_position_toxic],
                    tokenized_toxic_sample[1:],
                    torch.tensor([259]),
                    input_ids[0][sample_position_toxic:],
                ],
                0,
            )
            dataset_list.append(tokenizer.decode(mix_input_sample))
        df_mix_data = pd.DataFrame({"prompt": dataset_list})
        df_mix_data.to_csv(
            f"/home/ubuntu/finetuning_benchmark_datasets_large/toxic_{data_name}.csv",
            index=False,
        )


if __name__ == "__main__":
    # mix_pile_toxic()
    get_toxic_clean_data_benchmarch()
