import numpy as np

import pandas as pd
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding

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


def fit_sup(data, labels, method, by=None, n_folds=5, pct_unlabeled=0):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    names = encoder.classes_
    if by == None:
        chunks = [data.reshape((len(data), -1))]
        index = ["all"]
    elif by == "layer":
        chunks = [data[:, i] for i in range(data.shape[1])]
        index = [f"layer {1+i}" for i in range(data.shape[1])]
    elif by == "feature":
        chunks = [data[:, :, i] for i in range(data.shape[2])]
        index = [f"feature {1+i}" for i in range(data.shape[2])]
    scores = np.zeros((len(chunks), len(names)))
    scores = pd.DataFrame(scores, columns=names, index=index)
    scaler = StandardScaler()

    columns = list(names) + ["Total"]
    if pct_unlabeled:
        nrows = 3
        index = ["labeled train set", "unlabeled train set", "test set"]
    else:
        nrows = 2
        index = ["train set", "test set"]
    sizes = pd.DataFrame(
        np.zeros((nrows, len(names) + 1)), columns=columns, index=index
    )

    for c, chunk in tqdm(enumerate(chunks), total=len(chunks), disable=by is None):
        for fold in range(n_folds):
            X_train, X_test, y_train, y_test = train_test_split(
                chunk, y, test_size=0.3, random_state=fold, stratify=y
            )
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if pct_unlabeled:
                cp_y_train = y_train.copy()
                _, undo = train_test_split(
                    range(len(y_train)),
                    test_size=pct_unlabeled,
                    random_state=fold,
                    stratify=y_train,
                )
                cp_y_train[undo] = -1
                yhat = method.fit(X_train, cp_y_train).predict_proba(X_test)
            else:
                yhat = method.fit(X_train, y_train).predict_proba(X_test)
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
    # DT1 = DecisionTreeClassifier(
    #     class_weight="balanced", min_samples_leaf=3, max_depth=4
    # )
    # DT2 = DecisionTreeClassifier(
    #     class_weight="balanced", min_samples_leaf=3, max_depth=20
    # )
    LR = LogisticRegression(class_weight="balanced", n_jobs=-1, max_iter=400)

    """
    Semi supervised experiments that is here done only on the RF model
    """
    kwargs = dict(annot=True, fmt=".1f")
    kwargs["annot_kws"] = {"fontsize": 11, "color": "k"}
    fig, ax = plt.subplots(figsize=(9, 8))
    all_semisup_scores = []
    all_sup_scores = []
    for m in [RF, LR]:
        name = type(m).__name__

        if Path(f"./figures/table_{exp_name}_semisup_{name}.txt").is_file():
            Path(f"./figures/table_{exp_name}_semisup_{name}.txt").unlink()
        if Path(f"./figures/table_{exp_name}_semisup_acc_{name}.txt").is_file():
            Path(f"./figures/table_{exp_name}_semisup_acc_{name}.txt").unlink()
        for pct in tqdm(pcts, desc=f"Semisup {type(m).__name__}"):
            # set up the semisup learning method
            k_best = int(len(y) * 0.7 * pct * 0.1)
            m = SelfTrainingClassifier(m, criterion="k_best", k_best=k_best)
            # fit, get score and number of samples per dataset
            sc, sz = fit_sup(X, y, method=m, pct_unlabeled=pct, n_folds=n_folds)
            with open(f"./figures/table_{exp_name}_semisup_{name}.txt", "a+") as f:
                print(sz)
                f.write(f"{np.round(100*pct,3)}\%\n")
                f.write(str(sz.to_latex(float_format="{:.2f}".format)))
                f.write(r"\\")
            all_semisup_scores.append(sc)
            all_semisup_scores[-1].index = [f"{name} ({pct}%)"]
        scores = pd.concat(all_semisup_scores)
        with open(f"./figures/table_{exp_name}_semisup_acc_{name}.txt", "a+") as f:
            f.write(str(scores.to_latex(float_format="{:.2f}".format)))
        seaborn.heatmap(scores, ax=ax, cmap=cmap, **kwargs)
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
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        for ax, by in zip(axs, ["layer", "feature"]):
            print(f"\t\t-Group: {by}")
            scores, _ = fit_sup(X, y, method=m, by=by, n_folds=n_folds)
            seaborn.heatmap(scores, ax=ax, cmap=cmap, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        plt.subplots_adjust(0.08, 0.15, 1, 0.995)
        plt.savefig(f"./figures/{exp_name}_classification_{name}.png")
        plt.close()
        print(f"\t\t-Group: All")
        all_sup_scores.append(fit_sup(X, y, method=m)[0])
        all_sup_scores[-1].index = [name]
    print(pd.concat(all_semisup_scores).to_latex(float_format="{:.2f}".format))
    print(pd.concat(all_sup_scores).to_latex(float_format="{:.2f}".format))


def load_data(path: str, jigsaw_subset: str = "clean") -> Tuple[np.ndarray, List[str]]:
    """load the statistics and prompts saved using another script.

    Args:
        datasets (str): the path to .csv

    Returns:
        Tuple[np.ndarray, List[str]]: the loaded data in a numpy format
    """

    raw_data = pd.read_csv(path)
    if "jigsaw" in str(path):
        y = raw_data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].values
        if "clean" in jigsaw_subset:
            clean = ~y.any(1)
            valid = np.flatnonzero(clean)
            if jigsaw_subset == "clean_small":
                valid = valid[np.random.permutation(np.count_nonzero(clean))[:10000]]
        elif jigsaw_subset == "toxic":
            toxic = y.any(1)
            valid = np.flatnonzero(toxic)
        elif jigsaw_subset == "very_toxic":
            toxic = y[:, 1]
            valid = np.flatnonzero(toxic)
        raw_data = raw_data.loc[valid]
    prompts = raw_data["prompt"].values
    stats = raw_data["stats"].apply(ast.literal_eval)
    stats = np.stack(stats.values)
    return stats, prompts


def load_jigsaw(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """load the X, y and prompts for jigsaw

    Args:
        datasets (str): the path to .csv file

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: the loaded data in a numpy format
    """
    raw_data = pd.read_csv(path)
    y = raw_data[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values
    prompts = raw_data["prompt"].values
    X = np.stack(raw_data["stats"].apply(ast.literal_eval).values)
    return X, y, prompts


if __name__ == "__main__":
    rc("axes", linewidth=2)
    rc("font", weight="bold")
    rc("font", size=16)
    rc("axes", labelsize=18)

    # X, y, prompts = load_jigsaw("../Downloads/polytope_all/full_jigsaw/statistics.csv")
    # X = X.reshape((len(X), -1))

    # def scoring(y_test, yhat):
    #     return roc_auc_score(y_test, yhat[:, 1])

    # scorer = make_scorer(scoring, needs_proba=True)
    # df = pd.DataFrame(
    #     columns=[
    #         "toxic",
    #         "severe_toxic",
    #         "obscene",
    #         "threat",
    #         "insult",
    #         "identity_hate",
    #     ]
    # )
    # scaler = StandardScaler()
    # (
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     prompts_train,
    #     prompts_test,
    # ) = train_test_split(X, y, prompts, test_size=0.5, random_state=0)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # for module, params in zip(
    #     [
    #         RandomForestClassifier(n_jobs=-1),
    #         LogisticRegression(
    #             class_weight="balanced",
    #             max_iter=600,
    #             penalty="elasticnet",
    #             solver="saga",
    #             n_jobs=-1,
    #         ),
    #         KNeighborsClassifier(n_jobs=-1, weights="distance"),
    #     ],
    #     [
    #         {
    #             "class_weight": ("balanced", "balanced_subsample"),
    #             "min_samples_leaf": (20, 200, 1000),
    #             "max_depth": (4, None),
    #         },
    #         {"C": (0.001, 0.1, 1, 10, 1000), "l1_ratio": (0.1, 0.5, 0.9)},
    #         {"n_neighbors": (200, 2000), "leaf_size": (50)},
    #     ],
    # ):
    #     clf = GridSearchCV(module, params, verbose=1, cv=2, n_jobs=-1)
    #     clf = MultiOutputClassifier(clf)

    #     yhat = clf.fit(X_train, y_train).predict_proba(X_test)
    #     for i in np.flatnonzero((yhat[0].argmax(1) == 1) & (y_test[:, 0] == 0))[:100]:
    #         print("Predicted toxic but labelled not toxic:", prompts_test[i])
    #     # for i in np.flatnonzero((yhat[0].argmax(1) == 0) & (y_test[:, 0] == 1))[:30]:
    #     #     print("Predicted not toxic but labelled toxic:", prompts_test[i])

    #     aucs = []
    #     for p in range(len(yhat)):
    #         aucs.append(roc_auc_score(y_test[:, p], yhat[p][:, 1]))
    #     df.loc[type(module).__name__] = aucs
    #     print(df.to_latex(float_format="{:.2f}".format))
    #     print(df.mean(1))

    # datasets = {
    #     "FreeLaw": "../Downloads/polytope_all/FreeLaw/statistics.csv",
    #     "PubMed": "../Downloads/polytope_all/PubMed Abstracts/statistics.csv",
    #     "DM Math.": "../Downloads/polytope_all/DM Mathematics/statistics.csv",
    #     "USPTO": "../Downloads/polytope_all/USPTO Backgrounds/statistics.csv",
    #     "Github": "../Downloads/polytope_all/Github/statistics.csv",
    #     "dollyQA": "../Downloads/polytope_all/dollyQA/statistics.csv",
    #     "jigsaw_clean": "../Downloads/polytope_all/full_jigsaw/statistics.csv",
    # }
    # X = []
    # y = []
    # prompts = []
    # pbar = tqdm(datasets.items(), total=len(datasets))
    # for name, path in pbar:
    #     pbar.set_description(f"Loading {name} from {path}")
    #     X_, prompts_ = load_data(path, jigsaw_subset="clean_small")
    #     print(X_.shape)
    #     y.append(np.asarray([name] * len(X_)))
    #     prompts.extend(prompts_)
    #     X.append(X_)
    # X = np.concatenate(X)
    # y = np.concatenate(y)

    # run_sup_semisup_experiments(X, y, exp_name="nontoxicdatasets", n_folds=1, pcts=[0.5, 0.6,0.7,0.8,0.9,0.95,0.99])
    # unsupervised_embedding(np.reshape(X, (len(X), -1)), y, exp_name="nontoxicdatasets")

    datasets = {
        "toxic_pile": "../Downloads/polytope_all/toxic_pile/statistics.csv",
        "toxigen": "../Downloads/polytope_all/toxicity/statistics.csv",
        "FreeLaw": "../Downloads/polytope_all/FreeLaw/statistics.csv",
        "PubMed": "../Downloads/polytope_all/PubMed Abstracts/statistics.csv",
        "DM Math.": "../Downloads/polytope_all/DM Mathematics/statistics.csv",
        "USPTO": "../Downloads/polytope_all/USPTO Backgrounds/statistics.csv",
        "dollyQA": "../Downloads/polytope_all/dollyQA/statistics.csv",
        "Github": "../Downloads/polytope_all/Github/statistics.csv",
        "jigsaw_clean": "../Downloads/polytope_all/full_jigsaw/statistics.csv",
        "jigsaw_very_toxic": "../Downloads/polytope_all/full_jigsaw/statistics.csv",
    }
    X = []
    y = []
    prompts = []
    pbar = tqdm(datasets.items(), total=len(datasets))
    for name, path in pbar:
        pbar.set_description(f"Loading {name} from {path}")
        if "jigsaw_clean" == name:
            X_, prompts_ = load_data(path, jigsaw_subset="clean")
        elif "jigsaw_toxic" == name:
            X_, prompts_ = load_data(path, jigsaw_subset="toxic")
        elif "jigsaw_very_toxic" == name:
            X_, prompts_ = load_data(path, jigsaw_subset="very_toxic")
        else:
            X_, prompts_ = load_data(path)
        y.append(
            np.asarray(
                [
                    "toxic"
                    if name
                    in ["toxigen", "toxic_pile", "jigsaw_toxic", "jigsaw_very_toxic"]
                    else "clean"
                ]
                * len(X_)
            )
        )
        prompts.extend(prompts_)
        X.append(X_)
    X = np.concatenate(X)
    y = np.concatenate(y)

    run_sup_semisup_experiments(
        X, y, exp_name="toxicseparation", n_folds=1, pcts=[0.8, 0.9, 0.95]
    )
    unsupervised_embedding(np.reshape(X, (len(X), -1)), y, exp_name="toxicseparation")

    # Clustering results
    # fig, axs = plt.subplots(1, 2)
    # scores = clustering(
    #     statistics, labels, method=KMeans(n_clusters=20, n_init="auto"), by="layer"
    # )
    # seaborn.heatmap(scores * 100, annot=True, fmt=".1f", ax=axs[0])
    # scores = clustering(
    #     statistics, labels, method=KMeans(n_clusters=20, n_init="auto"), by="feature"
    # )
    # seaborn.heatmap(scores * 100, annot=True, fmt=".1f", ax=axs[1])
    # plt.tight_layout()
    # plt.show()
    # scores = clustering(
    #     statistics, labels, method=KMeans(n_clusters=20, n_init="auto"), by=None
    # )
