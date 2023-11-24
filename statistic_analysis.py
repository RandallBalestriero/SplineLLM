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
)
from sklearn.semi_supervised import SelfTrainingClassifier
import seaborn
from matplotlib import rc

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
    scaler = sklearn.preprocessing.StandardScaler()

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
            scores.iloc[c, :] += roc_auc_score(
                y_test, yhat, multi_class="ovr", average=None
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


def unsupervised_embedding(data, labels, n_components=2, plot=False):
    colors = plt.cm.tab10.colors

    print(data.shape, labels.shape)
    pbar = tqdm(range(3), total=3)
    X = data.reshape((len(data), -1))
    # X = sklearn.preprocessing.StandardScaler().fit_transform(X)

    pbar.set_description(f"Computing PCA")
    X_pca = PCA(n_components=n_components, random_state=0).fit_transform(X)
    pbar.update(1)
    pbar.set_description(f"Computing TSNE")
    X_tsne = TSNE(n_components=n_components, random_state=0).fit_transform(X)
    pbar.update(1)
    pbar.set_description(f"Computing FastICA")
    X_se = TSNE(
        n_components=n_components, random_state=0, perplexity=100
    ).fit_transform(X)
    pbar.close()

    if plot:
        colors = LabelEncoder().fit_transform(labels)
        fig, axs = plt.subplots(1, 3, figsize=(12, 12))
        for ax, x, name in zip(
            axs, [X_pca, X_tsne, X_se], ["PCA", "T-SNE", "Spectral Embedding"]
        ):
            ax.scatter(x[:, 0], x[:, 1], c=colors)
            ax.set_title(name)
        plt.show()
    # plt.savefig("toxic_pile_pca.png")


def load_data(datasets: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """load the statistics and prompts saved using another script.

    Args:
        datasets (Dict[str, str]): the mapping from dataset name (or subset) to statistic file

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: the loaded data in a numpy format
    """

    data = []
    labels = []
    prompts = []
    pbar = tqdm(datasets.items(), total=len(datasets))
    for name, path in pbar:
        pbar.set_description(f"Loading {name} from {path}")
        raw_data = pd.read_csv(path)
        labels.append(np.asarray([name] * len(raw_data)))
        for index, row in tqdm(raw_data.iterrows(), total=len(raw_data), leave=False):
            stats = np.array(ast.literal_eval(row["stats"]))
            data.append(stats)
            prompts.append(row["prompt"])
    data = np.stack(data)
    labels = np.concatenate(labels)
    return data, labels, prompts


if __name__ == "__main__":
    rc("axes", linewidth=2)
    rc("font", weight="bold")
    rc("font", size=16)
    rc("axes", labelsize=18)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    datasets = {
        "toxic_pile": "../Downloads/polytope/toxic_pile/statistics.csv",
        "hotel": "../Downloads/polytope/hotel/statistics.csv",
        "FreeLaw": "../Downloads/polytope/FreeLaw/statistics.csv",
        "PubMed": "../Downloads/polytope/PubMed Abstracts/statistics.csv",
        "DM Math.": "../Downloads/polytope/DM Mathematics/statistics.csv",
        "USPTO": "../Downloads/polytope/USPTO Backgrounds/statistics.csv",
        "Github": "../Downloads/polytope/Github/statistics.csv",
    }
    n_folds = 5

    features, labels, prompts = load_data(datasets)

    # unsupervised_embedding(statistics, labels, plot=True)

    """
    We define some models that will be used for the supervised and semi-supervised experiments
    """
    RF = RandomForestClassifier(class_weight="balanced", n_jobs=-1, min_samples_leaf=3)
    KNN = KNeighborsClassifier(
        n_jobs=-1, n_neighbors=32, weights="distance", leaf_size=5
    )
    GBDT = GradientBoostingClassifier(n_estimators=30)
    DT1 = DecisionTreeClassifier(
        class_weight="balanced", min_samples_leaf=3, max_depth=4
    )
    DT2 = DecisionTreeClassifier(
        class_weight="balanced", min_samples_leaf=3, max_depth=20
    )
    LR = LogisticRegression(class_weight="balanced", n_jobs=-1, max_iter=400)

    """
    Semi supervised experiments that is here done only on the RF model
    """
    kwargs = dict(annot=True, fmt=".1f")
    kwargs["annot_kws"] = {"fontsize": 11, "color": "k"}
    all_scores = []
    fig, ax = plt.subplots(figsize=(9, 8))
    pcts = (1 - np.linspace(0, 1, 15)[::-1] ** 4) * 0.395 + 0.6
    for pct in tqdm(pcts, desc=f"Semisup {type(RF).__name__}"):
        # set up the semisup learning method
        k_best = int(len(labels) * 0.7 * pct * 0.1)
        m = SelfTrainingClassifier(RF, criterion="k_best", k_best=k_best)
        # fit, get score and number of samples per dataset
        sc, sz = fit_sup(features, labels, method=m, pct_unlabeled=pct, n_folds=n_folds)
        print(sz)
        all_scores.append(sc)
    scores = pd.concat(all_scores)
    scores.index = np.round(100 - pcts * 100, 2)
    seaborn.heatmap(scores, ax=ax, cmap=cmap, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel("% of training set being labeled", weight="bold")
    plt.subplots_adjust(0.14, 0.15, 1, 0.99)
    plt.savefig("./figures/semisup.png")
    plt.close()

    """
    Supervised experiments that is done on all the defined models
    """
    kwargs = dict(annot=True, fmt=".1f")
    kwargs["annot_kws"] = {"fontsize": 11, "color": "k"}
    all_scores = []
    for m in [RF, KNN, DT1, DT2, LR]:
        name = type(m).__name__
        if "DecisionTreeClassifier" == name:
            name += f"(depth={m.max_depth})"
        name = name.replace("Classifier", "")
        print(f"\t-Fitting {name}")
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        for ax, by in zip(axs, ["layer", "feature"]):
            print(f"\t\t-Group: {by}")
            scores, _ = fit_sup(features, labels, method=m, by=by, n_folds=n_folds)
            seaborn.heatmap(scores, ax=ax, cmap=cmap, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
        plt.subplots_adjust(0.08, 0.15, 1, 0.995)
        plt.savefig(f"./figures/classification_{name}.png")
        plt.close()
        print(f"\t\t-Group: All")
        all_scores.append(fit_sup(features, labels, method=m)[0])
        all_scores[-1].index = [name]
    print(pd.concat(all_scores).to_latex(float_format="{:.2f}".format))

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
