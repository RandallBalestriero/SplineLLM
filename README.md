# Code repository for the paper *Characterizing Large Language Model Geometry Solves Toxicity Detection and Generation*

The codebase allows to reproduce our figures and tables. 

| Model | Toxicity detection AUC | Latency |
| ----- | ------- | ---------------------- |
|Spline-Llama2-7B (linear) | 99.18 | 0.061s|
|Spline-Llama2-7B (3 layers, RF) | 94.68 | 0.005s|
|Spline-Mistral-7B (linear) | 98.45 | 0.066s |
|Spline-Mistral-7B (3 layers, RF) |  93.73 | 0.006s|

We encourage reproducibility and hope that any issue that may arise in the process will be raised in this repository to ensure that the codebase is stable and useful to everyone.

## Spline features extraction

The procedure to extract our proposed spline features is as follows:
- as per the Source Code 1 from the paper, add the spline statistics in the `modelling_llama.py` file from the `Transformers` package. We provide in this repo the file we employed for reference.
- make sure to save those statistics as part of the dataclass used to collect the LLM outputs and retreive it for all the layers at the end of the forward pass
- use that code for Llama2 and Mistral. Although we did not explore other architectures, we beleive our derived features to be equally informative in other settings.


## Intrinsic dimension experiment

ToDo


## Generating the figures

- all the figures about T-SNE, table of results, semi-supervised learning, and ablation (looking at peformance of our features per-layer) are provided in the `statistic_analysis.py` file. In there, you will find a few utilities such as:
  -  `unsupervised_embedding` used to obtain the T-SNE
  - `solve_jigsaw` to solve the jigsaw challenge
  - `run_sup_semisup_experiments` for the supervised (and semi-supervised) experiments
  - `compare_huggingface_classifiers` to obtain the result of the HuggingFace models for the table in page 1
  
  their usage should be self-explanatory from our use-cases and mostly involve splitting the dataset into a train and test set, and then calling `sklearn` classifiers and reporting/plotting test performances

- for the figures about the intrinsice dimension generation and the tables, the code is provided in the `toxicity_id_eval` folder. There you will find some examples of prompts we produced in `.csv` files and the `id_plotting.py` file which can be used to generate the various figures and tables