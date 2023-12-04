# Code repository for the paper **Characterizing Large Language Model Geometry Solves Toxicity Detection and Generation**

The codebase allows to reproduce our figures and tables. 


*To provide a practical and principled answer, we propose to characterize LLMs from a geometric perspective. We obtain in closed form (i) the intrinsic dimension in which the Multi-Head Attention embeddings are constrained to exist and (ii) the partition and per-region affine mappings of the per-layer feedforward networks. Our results are informative, do not rely on approximations, and are actionable. First, we show that, motivated by our geometric interpretation, we can break Llama$2$'s RLHF by controlling its embedding's intrinsic dimension through informed prompt manipulation. Second, we derive $7$ interpretable spline features that can be extracted from any (pre-trained) LLM layer, providing a rich abstract representation of their inputs. Those features alone ($224$ for Mistral-7B and Llama$2$-7B) are sufficient to help solve toxicity detection, infer the domain of the prompt, and even solve the Jigsaw challenge, which aims to characterize the type of toxicity of various prompts. Our results demonstrate how, even in large-scale regimes, exact theoretical results can answer practical questions in language models.*


- Our study enables toxicity detection from pre-trained LLM using only 224 extracted features:

    | Model | Toxicity detection AUC | Latency |
    | ----- | ------- | ---------------------- |
    |Spline-Llama2-7B (linear) | 99.18 | 0.061s|
    |Spline-Llama2-7B (3 layers, RF) | 94.68 | 0.005s|
    |Spline-Mistral-7B (linear) | 98.45 | 0.066s |
    |Spline-Mistral-7B (3 layers, RF) |  93.73 | 0.006s|
- Domain separation, again using 224 extracted features:

    | Model |classifier| DM Math. | FreeLaw | Github | PubMed | USPTO | dollyQA | jigsaw(clean) |
    | --   | ---| ---       | ---     | ---   | ---    | ---   | --- | --- |
    |Spline-Mistral-7B|RandomForest | 100.00 | 99.77 | 99.24 | 99.37 | 98.25 | 97.73 | 94.62 |
    | Spline-Mistral-7B|LogisticRegression | 100.00 | 99.82 | 99.76 | 99.86 | 99.79 | 99.14 | 98.68 |
    |Spline-Mistral-7B|LogisticRegression (1\% labels) | 99.97 | 99.25 | 98.09 | 97.47 | 94.83 | 94.45 | 89.87 |
    |Spline-Llama2-7B| RandomForest | 99.98 | 99.86 | 99.29 | 99.73 | 98.89 | 98.88 | 97.63 |
    |Spline-Llama2-7B|LogisticRegression | 100.00 | 99.87 | 99.76 | 99.92 | 99.92 | 99.63 | 99.33 |
    |Spline-Llama2-7B|LogisticRegression (1\% labels) | 99.31 | 99.60 | 98.60 | 99.32 | 98.21 | 98.18 | 96.11 |

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