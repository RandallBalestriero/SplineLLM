import numpy as np

import pandas as pd
from sklearn.manifold import TSNE

import seaborn as sns

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import ast
from typing import Dict, Tuple, List
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
import seaborn
from matplotlib import rc
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import warnings
import os

# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ["PYTHONWARNINGS"] = "ignore"


plt.style.use("ggplot")


def clustering(data, labels, method, by=None):
    y = LabelEncoder().fit_transform(labels)
    if by == None:
        chunks = [data.reshape((len(data), -1))]
        index = ["all"]
    elif by == "layer":
        chunks = [data[:, i] for i in range(data.shape[1])]
        index = [f"layer {1+i}" for i in range(data.shape[1])]
    elif by == "feature":
        chunks = [data[:, :, i] for i in range(data.shape[2])]
        index = [f"feature {1+i}" for i in range(data.shape[2])]
    scores = pd.DataFrame(
        np.empty((len(chunks), 6)),
        columns=[
            "homogeneity",
            "completeness",
            "v-score",
            "fowlkes-mallows score",
            "adj. rand score",
            "normalize MI",
        ],
        index=index,
    )
    for c, chunk in tqdm(enumerate(chunks), desc="Chunks"):
        X = sklearn.preprocessing.StandardScaler().fit_transform(chunk)
        y_kmeans = method.fit_predict(X)
        scores.iloc[c, :3] = homogeneity_completeness_v_measure(y, y_kmeans)
        scores.iloc[c, 3] = fowlkes_mallows_score(y, y_kmeans)
        scores.iloc[c, 4] = adjusted_rand_score(y, y_kmeans)
        scores.iloc[c, 5] = normalized_mutual_info_score(y, y_kmeans)
    return scores


def fit_sup(data, labels, model, by=None, n_folds=5, pct_unlabeled=0, test_size=0.3):
    multioutput = len(labels.shape) > 1 and labels.shape[1] > 1
    if not multioutput:
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels).flatten()
        names = encoder.classes_
    else:
        y = labels
        names = [str(i) for i in range(y.shape[1])]
        # if semisup we still fit one dim at a time, so skipping multioutput
        if pct_unlabeled == 0:
            model = MultiOutputClassifier(model)
    if by == None:
        chunks = [data.reshape((len(data), -1))]
        index = ["all"]
    elif by == "layer":
        chunks = [data[:, i] for i in range(data.shape[1])]
        index = [f"layer {1+i}" for i in range(data.shape[1])]
    elif by == "layer_progressive":
        chunks = [
            data[:, : i + 1].reshape((len(data), -1)) for i in range(data.shape[1])
        ]
        index = [f"layer 1->{1+i}" for i in range(data.shape[1])]
    elif by == "feature":
        chunks = [data[:, :, i] for i in range(data.shape[2])]
        index = [f"feature {1+i}" for i in range(data.shape[2])]
    print(f"Fitting inputs: X (shape={data.shape}), y (shape={y.shape})")
    scores = np.zeros((len(chunks), len(names)))
    scores = pd.DataFrame(scores, columns=names, index=index)
    scaler = StandardScaler()

    # create data frame to keep track of sizes in train/test sets (and semisup)
    columns = list(names) + ["Total"]
    if pct_unlabeled:
        nrows = 3
        index = ["labeled train set", "unlabeled train set", "test set"]
        k_best = int(len(chunks[0]) * (1 - test_size) * pct_unlabeled) // 10
        model = SelfTrainingClassifier(model, criterion="k_best", k_best=k_best)
    else:
        nrows = 2
        index = ["train set", "test set"]
    sizes = np.zeros((nrows, len(names) + 1))
    sizes = pd.DataFrame(sizes, columns=columns, index=index)

    for c, chunk in tqdm(enumerate(chunks), total=len(chunks), disable=by is None):
        print(f"\tchunk={c}")
        for fold in range(n_folds):
            print(f"\t\t{fold=}")
            stratify = y if not multioutput else None
            X_train, X_test, y_train, y_test = train_test_split(
                chunk, y, test_size=test_size, random_state=fold, stratify=stratify
            )
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # perform fitting and prediction
            if pct_unlabeled:
                stratify = y_train if not multioutput else None
                r = range(len(y_train))
                _, undo = train_test_split(
                    r, test_size=pct_unlabeled, random_state=fold, stratify=stratify
                )
                cp_y_train = y_train.copy()
                cp_y_train[undo] = -1
                if multioutput:
                    yhat = []
                    for i in range(y.shape[1]):
                        model.fit(X_train, cp_y_train[:, i])
                        yhat.append(model.predict_proba(X_test))
                else:
                    yhat = model.fit(X_train, cp_y_train).predict_proba(X_test)
            else:
                yhat = model.fit(X_train, y_train).predict_proba(X_test)

            # perform evaluation metric computation
            if multioutput:
                for i in range(len(yhat)):
                    scores.iloc[c, i] += roc_auc_score(y_test[:, i], yhat[i][:, 1])
            else:
                if len(names) == 2:
                    yhat = yhat[:, 1]
                scores.iloc[c, :] += roc_auc_score(
                    y_test,
                    yhat,
                    multi_class="ovr" if len(names) > 2 else "raise",
                    average=None,
                )
    for k in range(len(names)):
        sizes.iloc[-1, k] = np.count_nonzero(y_test == k)
        if pct_unlabeled:
            sizes.iloc[0, k] = np.count_nonzero(cp_y_train == k)
            sizes.iloc[1, k] = np.count_nonzero((y_train == k) & (cp_y_train == -1))
        else:
            sizes.iloc[0, k] = np.count_nonzero(y_train == k)
    sizes.iloc[:, -1] = sizes.iloc[:, :-1].sum(1)
    return scores * 100 / n_folds, sizes.astype("int")


def unsupervised_embedding(X, y, exp_name):
    cmap = plt.cm.tab10.colors

    pbar = tqdm(range(2), total=2)

    pbar.set_description(f"Computing PCA")
    X_pca = PCA(n_components=2, random_state=0).fit_transform(X)
    pbar.update(1)
    pbar.set_description(f"Computing TSNE")
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
    pbar.update(1)
    pbar.close()

    encoder = LabelEncoder()
    colors = encoder.fit_transform(y)
    for x, name in zip([X_pca, X_tsne], ["PCA", "T-SNE"]):
        fig, ax = plt.subplots(figsize=(7, 7))
        for k in range(len(encoder.classes_)):
            ax.scatter(
                x[colors == k, 0],
                x[colors == k, 1],
                color=cmap[k],
                edgecolor="gray",
                linewidth=1,
                label=encoder.classes_[k],
                alpha=0.2,
            )
        leg = plt.legend(ncol=2)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"./figures/{exp_name}_{name}.png")
        plt.close()


def run_sup_semisup_experiments(X, y, exp_name, n_folds: int = 5, pcts=None):
    if pcts is not None:
        pcts = np.array(pcts)
    else:
        pcts = []
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    print(f"Running sup/semisup experiment with X:{X.shape}, y:{y.shape}, {n_folds=}")
    """
    We define some models that will be used for the supervised and semi-supervised experiments
    """
    RF = RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, min_samples_leaf=100
    )
    GBDT = GradientBoostingClassifier(n_estimators=30)
    LR = LogisticRegression(class_weight="balanced", n_jobs=-1, max_iter=400)

    """
    Semi supervised experiments that is here done only on the RF model
    """
    kwargs = dict(annot=True, fmt=".1f")
    kwargs["annot_kws"] = {"fontsize": 11, "color": "k"}
    fig, ax = plt.subplots(figsize=(9, 8))
    semisup_scores = []
    semisup_sizes = []
    all_sup_scores = []
    for m in [RF, LR]:
        name = type(m).__name__
        for pct in tqdm(pcts, desc=f"Semisup {type(m).__name__}"):
            sc, sz = fit_sup(X, y, model=m, pct_unlabeled=pct, n_folds=n_folds)
            sz.index = sz.index + f" ({100 - 100*pct}\%)"
            sc.index = [f"{name} ({100 - 100*pct}\%)"]
            semisup_scores.append(sc)
            semisup_sizes.append(sz)
        seaborn.heatmap(
            pd.concat(semisup_scores[-len(pcts) :]), ax=ax, cmap=cmap, **kwargs
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        ax.set_yticklabels(
            [f"{v:.2f}" for v in np.round(100 - pcts * 100, 2)], rotation=0
        )
        ax.set_ylabel("% of training set being labeled", weight="bold")
        plt.subplots_adjust(0.14, 0.15, 1, 0.99)
        plt.savefig(f"./figures/{exp_name}_semisup_{name}.png")
        plt.close()

        """
        Supervised experiments that is done on all the defined models
        """
        kwargs = dict(annot=True, fmt=".1f")
        kwargs["annot_kws"] = {"fontsize": 11, "color": "k"}

        if "DecisionTreeClassifier" == name:
            name += f"(depth={m.max_depth})"
        name = name.replace("Classifier", "")
        print(f"\t-Fitting {name}")
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        for ax, by in zip(axs, ["layer", "layer_progressive", "feature"]):
            print(f"\t\t-Group: {by}")
            scores, _ = fit_sup(X, y, model=m, by=by, n_folds=n_folds)
            seaborn.heatmap(scores, ax=ax, cmap=cmap, cbar=False, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        plt.tight_layout()
        # plt.subplots_adjust(0.06, 0.13, 1, 0.995, 0.15, 0.15)
        plt.savefig(f"./figures/{exp_name}_classification_{name}.png")
        plt.close()
        print(f"\t\t-Group: All")
        all_sup_scores.append(fit_sup(X, y, model=m)[0])
        all_sup_scores[-1].index = [name]
    with open(f"./figures/table_{exp_name}_semisup_acc.txt", "w") as f:
        f.write(str(pd.concat(semisup_scores).to_latex(float_format="{:.2f}".format)))
    with open(f"./figures/table_{exp_name}_semisup_sizes.txt", "w") as f:
        f.write(str(pd.concat(semisup_sizes).to_latex(float_format="{:.2f}".format)))
    with open(f"./figures/table_{exp_name}_sup_acc.txt", "w") as f:
        f.write(str(pd.concat(all_sup_scores).to_latex(float_format="{:.2f}".format)))


def load_data(path: str, jigsaw_subset: str = "clean") -> Tuple[np.ndarray, List[str]]:
    """load the statistics and prompts saved using another script.

    Args:
        datasets (str): the path to .csv
        jigsaw_subset (str): must be one of 'clean' or 'toxic'

    Returns:
        Tuple[np.ndarray, List[str]]: the loaded data in a numpy format
    """

    raw_data = pd.read_csv(path)
    if "jigsaw" in str(path):
        y = raw_data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].values
        toxic = y.any(1)
        if "clean" in jigsaw_subset:
            valid = np.flatnonzero(~toxic)
        elif jigsaw_subset == "toxic":
            valid = np.flatnonzero(toxic)
        raw_data = raw_data.loc[valid]
    prompts = raw_data["prompt"].values
    stats = raw_data["stats"].apply(ast.literal_eval)
    stats = np.stack(stats.values)
    return stats, prompts


def solve_jigsaw(backbone, n_folds):
    columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    print("Loading csv")
    raw_data = pd.read_csv(
        f"../Downloads/polytope_{backbone}/full_jigsaw/statistics.csv"
    )
    y = raw_data[columns].values
    X = np.stack(raw_data["stats"].apply(ast.literal_eval).values)
    X = X.reshape((len(X), -1))
    model = LogisticRegression(
        class_weight="balanced", n_jobs=-1, max_iter=500, random_state=0
    )
    df1, _ = fit_sup(
        X, y, model=model, n_folds=n_folds, pct_unlabeled=0.98, test_size=0.2
    )
    df1.columns = columns
    df1.index = ["2% labels"]
    df2, _ = fit_sup(
        X, y, model=model, n_folds=n_folds, pct_unlabeled=0.95, test_size=0.2
    )
    df2.columns = columns
    df2.index = ["5% labels"]
    df3, _ = fit_sup(
        X, y, model=model, n_folds=n_folds, pct_unlabeled=0.9, test_size=0.2
    )
    df3.columns = columns
    df3.index = ["10% labels"]
    df4, _ = fit_sup(
        X, y, model=model, n_folds=n_folds, pct_unlabeled=0.0, test_size=0.2
    )
    df4.columns = columns
    df4.index = ["100% labels"]
    df = pd.concat([df1, df2, df3, df4])
    df["avg."] = df.mean(1)
    with open(f"./figures/table_{backbone}_jigsaw_acc.txt", "w") as f:
        f.write(str(df.to_latex(float_format="{:.2f}".format)))


def compare_huggingface_classifiers(n_folds: int = 5, test_size: float = 0.3):
    models = [
        "martin-ha",
        "ToxRoberta",
        "nicholasKluge",
        "unitary",
        "s-nlp",
        "citizenlab",
    ]

    all_times = []
    for name in ["dollyQA", "toxicity", "toxic_pile"]:
        times = (
            pd.read_csv(f"../Downloads/polytope_mistral/{name}/statistics.csv")
            .iloc[1:]["inference_time"]
            .apply(lambda x: np.array(list(ast.literal_eval(x).values())).flatten())
        )
        all_times.append(times)
    print("Mistral time", np.concatenate(all_times, axis=0).mean(0))

    clean_data = pd.read_csv("../Downloads/toxicity_scored_clean_timed.csv")
    toxic_data = pd.read_csv("../Downloads/toxicity_scored_toxic_timed.csv")
    clean_data["ToxRoberta_label"].loc[
        clean_data["ToxRoberta_label"] == "LABEL_1"
    ] = "toxic"
    clean_data["ToxRoberta_label"].loc[
        clean_data["ToxRoberta_label"] == "LABEL_0"
    ] = "non-toxic"
    data = pd.concat([clean_data, toxic_data], 0, ignore_index=True)
    labels = np.concatenate([np.zeros(len(clean_data)), np.ones(len(toxic_data))])
    print(clean_data.columns)
    for model in models:
        auc = 0
        for fold in range(n_folds):
            _, test = train_test_split(
                range(len(labels)),
                test_size=test_size,
                random_state=fold,
                stratify=labels,
            )
            auc += roc_auc_score(
                labels[test], data.iloc[test][f"{model}_label"] == "toxic"
            )
        print(
            model,
            auc / n_folds,
            data[f"{model}_timing"].mean(),
        )


if __name__ == "__main__":
    rc("axes", linewidth=2)
    rc("font", weight="bold")
    rc("font", size=16)
    rc("axes", labelsize=18)

    # compare_huggingface_classifiers()
    # asdf
    # solve_jigsaw("mistral", 3)
    # solve_jigsaw("llama7b", 3)
    # asdf

    n_folds = 5
    for backbone in ["mistral", "llama7b"]:
        datasets = {
            "FreeLaw": f"../Downloads/polytope_{backbone}/FreeLaw/statistics.csv",
            "PubMed": f"../Downloads/polytope_{backbone}/PubMed Abstracts/statistics.csv",
            "DM Math.": f"../Downloads/polytope_{backbone}/DM Mathematics/statistics.csv",
            "USPTO": f"../Downloads/polytope_{backbone}/USPTO Backgrounds/statistics.csv",
            "Github": f"../Downloads/polytope_{backbone}/Github/statistics.csv",
            "dollyQA": f"../Downloads/polytope_{backbone}/dollyQA/statistics.csv",
            # "jigsaw_clean": f"../Downloads/polytope_{backbone}/full_jigsaw/statistics.csv",
        }
        X = []
        y = []
        prompts = []
        pbar = tqdm(datasets.items(), total=len(datasets))
        for name, path in pbar:
            pbar.set_description(f"Loading {name} from {path}")
            X_, prompts_ = load_data(path, jigsaw_subset="clean_small")
            y.append(np.asarray([name] * len(X_)))
            prompts.extend(prompts_)
            X.append(X_)
        X = np.concatenate(X)
        y = np.concatenate(y)

        # run_sup_semisup_experiments(
        #     X,
        #     y,
        #     exp_name=f"nontoxicdatasets_{backbone}",
        #     n_folds=n_folds,
        #     pcts=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        # )
        unsupervised_embedding(
            np.reshape(X, (len(X), -1)), y, exp_name=f"nontoxicdatasets_{backbone}"
        )

    # datasets = {
    #     "toxic_pile": f"../Downloads/polytope_{backbone}/toxic_pile/statistics.csv",
    #     "toxigen": f"../Downloads/polytope_{backbone}/toxicity/statistics.csv",
    #     "FreeLaw": f"../Downloads/polytope_{backbone}/FreeLaw/statistics.csv",
    #     "PubMed": f"../Downloads/polytope_{backbone}/PubMed Abstracts/statistics.csv",
    #     "DM Math.": f"../Downloads/polytope_{backbone}/DM Mathematics/statistics.csv",
    #     "USPTO": f"../Downloads/polytope_{backbone}/USPTO Backgrounds/statistics.csv",
    #     "dollyQA": f"../Downloads/polytope_{backbone}/dollyQA/statistics.csv",
    #     "Github": f"../Downloads/polytope_{backbone}/Github/statistics.csv",
    #     "jigsaw_toxic": f"../Downloads/polytope_{backbone}/full_jigsaw/statistics.csv",
    # }
    # X = []
    # y = []
    # prompts = []
    # pbar = tqdm(datasets.items(), total=len(datasets))
    # for name, path in pbar:
    #     pbar.set_description(f"Loading {name} from {path}")
    #     if "jigsaw_clean" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="clean")
    #     elif "jigsaw_toxic" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="toxic")
    #     elif "jigsaw_very_toxic" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="very_toxic")
    #     else:
    #         X_, prompts_ = load_data(path)
    #     y.append(
    #         np.asarray(
    #             [
    #                 "toxic"
    #                 if name
    #                 in [
    #                     "toxigen",
    #                     "toxic_pile",
    #                     "jigsaw_toxic",
    #                     "jigsaw_very_toxic",
    #                 ]
    #                 else "clean"
    #             ]
    #             * len(X_)
    #         )
    #     )
    #     prompts.extend(prompts_)
    #     X.append(X_)
    #     print(X_.shape, y[-1].shape)
    # X = np.concatenate(X)
    # y = np.concatenate(y)

    # run_sup_semisup_experiments(
    #     X,
    #     y,
    #     exp_name=f"toxicseparation_nojigsawclean_{backbone}",
    #     n_folds=n_folds,
    #     pcts=[0.8, 0.9, 0.95],
    # )
    # unsupervised_embedding(
    #     np.reshape(X, (len(X), -1)),
    #     y,
    #     exp_name=f"toxicseparation_nojigsawclean_{backbone}",
    # )

    # datasets = {
    #     "toxic_pile": f"../Downloads/polytope_{backbone}/toxic_pile/statistics.csv",
    #     "toxigen": f"../Downloads/polytope_{backbone}/toxicity/statistics.csv",
    #     "FreeLaw": f"../Downloads/polytope_{backbone}/FreeLaw/statistics.csv",
    #     "PubMed": f"../Downloads/polytope_{backbone}/PubMed Abstracts/statistics.csv",
    #     "DM Math.": f"../Downloads/polytope_{backbone}/DM Mathematics/statistics.csv",
    #     "USPTO": f"../Downloads/polytope_{backbone}/USPTO Backgrounds/statistics.csv",
    #     "dollyQA": f"../Downloads/polytope_{backbone}/dollyQA/statistics.csv",
    #     "Github": f"../Downloads/polytope_{backbone}/Github/statistics.csv",
    #     "jigsaw_clean": f"../Downloads/polytope_{backbone}/full_jigsaw/statistics.csv",
    #     "jigsaw_toxic": f"../Downloads/polytope_{backbone}/full_jigsaw/statistics.csv",
    # }
    # X = []
    # y = []
    # prompts = []
    # pbar = tqdm(datasets.items(), total=len(datasets))
    # for name, path in pbar:
    #     pbar.set_description(f"Loading {name} from {path}")
    #     if "jigsaw_clean" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="clean")
    #     elif "jigsaw_toxic" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="toxic")
    #     elif "jigsaw_very_toxic" == name:
    #         X_, prompts_ = load_data(path, jigsaw_subset="very_toxic")
    #     else:
    #         X_, prompts_ = load_data(path)
    #     y.append(
    #         np.asarray(
    #             [
    #                 "toxic"
    #                 if name
    #                 in [
    #                     "toxigen",
    #                     "toxic_pile",
    #                     "jigsaw_toxic",
    #                     "jigsaw_very_toxic",
    #                 ]
    #                 else "clean"
    #             ]
    #             * len(X_)
    #         )
    #     )
    #     prompts.extend(prompts_)
    #     X.append(X_)
    # X = np.concatenate(X)
    # y = np.concatenate(y)

    # run_sup_semisup_experiments(
    #     X,
    #     y,
    #     exp_name=f"toxicseparation_withjigsawclean_{backbone}",
    #     n_folds=n_folds,
    #     pcts=[0.8, 0.9, 0.95],
    # )
    # unsupervised_embedding(
    #     np.reshape(X, (len(X), -1)),
    #     y,
    #     exp_name=f"toxicseparation_withjigsawclean_{backbone}",
    # )
    # Clustering results
    # fig, axs = plt.subplots(1, 2)
    # scores = clustering(
    #     statistics, labels, model=KMeans(n_clusters=20, n_init="auto"), by="layer"
    # )
    # seaborn.heatmap(scores * 100, annot=True, fmt=".1f", ax=axs[0])
    # scores = clustering(
    #     statistics, labels, model=KMeans(n_clusters=20, n_init="auto"), by="feature"
    # )
    # seaborn.heatmap(scores * 100, annot=True, fmt=".1f", ax=axs[1])
    # plt.tight_layout()
    # plt.show()
    # scores = clustering(
    #     statistics, labels, model=KMeans(n_clusters=20, n_init="auto"), by=None
    # )
