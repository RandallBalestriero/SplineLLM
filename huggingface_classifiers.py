from transformers import pipeline
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import time
from data_utils import get_toxic_clean_data_benchmarch


def classify_dataset(classification_pipeline, raw_dataset, classifier_name):
    output = []
    timing = []
    start_t = time.time()
    for out in tqdm(
        classification_pipeline(
            KeyDataset(raw_dataset, "prompt"), batch_size=1, truncation=True
        ),
        total=len(raw_dataset),
    ):
        end_t = time.time()
        output.append(out)
        timing.append(end_t - start_t)
        start_t = end_t

    output_dataset = datasets.Dataset.from_list(output)
    output_dataset = output_dataset.rename_columns(
        {"label": f"{classifier_name}_label", "score": f"{classifier_name}_score"}
    )
    output_dataset = output_dataset.add_column(f"{classifier_name}_timing", timing)
    return datasets.concatenate_datasets([raw_dataset, output_dataset], axis=1)



if __name__ == "__main__":
    # classifier_names = ["HateBert", "ToxRoberta"]
    # classifiers = ["tomh/toxigen_hatebert", "tomh/toxigen_roberta"]
    # tokenizers = ["bert-base-cased", "tomh/toxigen_roberta"]
    classifier_names = [
        "ToxRoberta",
        "martin-ha",
        "s-nlp",
        "citizenlab",
        "unitary",
        "nicholasKluge",
    ]
    classifiers = [
        "tomh/toxigen_roberta",
        "martin-ha/toxic-comment-model",
        "s-nlp/roberta_toxicity_classifier",
        "citizenlab/distilbert-base-multilingual-cased-toxicity",
        "unitary/toxic-bert",
        "nicholasKluge/ToxicityModel",
    ]
    tokenizers = [
        "tomh/toxigen_roberta",
        "martin-ha/toxic-comment-model",
        "s-nlp/roberta_toxicity_classifier",
        "citizenlab/distilbert-base-multilingual-cased-toxicity",
        "unitary/toxic-bert",
        "nicholasKluge/ToxicityModel",
    ]
    toxicity_classifiers = [
        pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")
        for model, tokenizer in zip(classifiers, tokenizers)
    ]

    hf_datasets = get_toxic_clean_data_benchmarch()
    for dataset_itr, str_dataset in enumerate(hf_datasets):
        dataset = datasets.Dataset.from_dict({"prompt": str_dataset})
        for classifier_itr, pipe in enumerate(toxicity_classifiers):
            dataset = classify_dataset(pipe, dataset, classifier_names[classifier_itr])

        dataset.to_csv(f"/home/ubuntu/polytope/toxicity_scored_{dataset_itr}.csv")
