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
from typing import Dict, Tuple, List, Literal, Optional
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


def load_data(datasets: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """load the statistics and prompts saved using another script.

    Args:
        datasets (Dict[str, str]): the mapping from dataset name (or subset) to statistic file

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: the loaded data in a numpy format
    """
    from numpy import nan

    all_stats = []
    all_tokens = []
    pbar = tqdm(datasets, total=len(datasets))
    for path in pbar:
        pbar.set_description(f"Loading {path}")
        raw_data = pd.read_csv(path)
        all_stats.append([])
        all_tokens.append([])
        sample_id = 0
        stats = []
        for index, row in tqdm(raw_data.iterrows(), total=len(raw_data), leave=False):
            if row["sample_id"] != sample_id:
                all_stats[-1].append(np.stack(stats))
                stats = []
                all_tokens[-1].append(tokens)
            sample_id = np.array(row["sample_id"])
            tokens = np.array(eval(row["tokens"]))
            stats.append(np.array(eval(row["stats"])))
    return all_stats, all_tokens


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.text import Annotation
from matplotlib.transforms import Transform
from matplotlib.backends.backend_agg import get_hinting_flag


def text_with_autofit(
    ax: plt.Axes,
    txt: str,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    transform: Optional[Transform] = None,
    ha: Literal["left", "center", "right"] = "center",
    va: Literal["bottom", "center", "top"] = "center",
    show_rect: bool = False,
    **kwargs,
) -> Annotation:
    if transform is None:
        transform = ax.transData

    #  Different alignments give different bottom left and top right anchors.
    x, y = xy
    xa0, xa1 = {
        "center": (x - width / 2, x + width / 2),
        "left": (x, x + width),
        "right": (x - width, x),
    }[ha]
    ya0, ya1 = {
        "center": (y - height / 2, y + height / 2),
        "bottom": (y, y + height),
        "top": (y - height, y),
    }[va]
    a0 = xa0, ya0
    a1 = xa1, ya1

    x0, _ = transform.transform(a0)
    x1, _ = transform.transform(a1)
    # rectangle region size to constrain the text in pixel
    rect_width = x1 - x0

    fig: plt.Figure = ax.get_figure()

    props = FontProperties()
    font = get_font(findfont(props))
    font.set_size(props.get_size_in_points(), fig.dpi)
    angle = 0
    font.set_text(txt, angle, flags=get_hinting_flag())
    w, _ = font.get_width_height()
    subpixels = 64
    adjusted_size = props.get_size_in_points() * rect_width / w * subpixels
    props.set_size(adjusted_size)

    text: Annotation = ax.annotate(
        txt, xy, ha=ha, va=va, xycoords=transform, fontproperties=props, **kwargs
    )

    if show_rect:
        rect = mpatches.Rectangle(a0, width, height, fill=False, ls="--")
        ax.add_patch(rect)

    return text


if __name__ == "__main__":
    rc("axes", linewidth=2)
    rc("font", weight="bold")
    rc("font", size=16)
    rc("axes", labelsize=18)
    datasets = [
        "../Downloads/polytope_incremental/toxic_uspto/incemental_stats.csv",
        # "../Downloads/polytope_incremental/toxic_pubmeb/incemental_stats.csv",
        # "../Downloads/polytope_incremental/toxic_maths/incemental_stats.csv",
        # "../Downloads/polytope_incremental/toxic_github/incemental_stats.csv",
        # "../Downloads/polytope_incremental/toxic_freelaw/incemental_stats.csv",
    ]
    stats, tokens = load_data(datasets)
    sizes = [len(u) for u in tokens[0][0]]
    sizes = np.cumsum(sizes)
    print(stats[0][0].shape)
    stats = stats[0][0].reshape((len(stats[0][0]), -1))
    print(stats)
    y = np.linspace(0, 1, stats.shape[1])
    stats[1:] = np.diff(stats, axis=0)
    stats[0] = 0
    print([str(i) for i in tokens[0][0]])
    # stats[0] = 0
    ax = plt.pcolormesh(
        sizes, y, np.log(np.abs(stats.T) + 1e-4), antialiased=True, shading="gouraud"
    )
    plt.xticks(sizes[:-1] / 2 + sizes[1:] / 2, tokens[0][0][1:], fontsize=8)
    print("".join([str(i) for i in tokens[0][0]]))
    # text_with_autofit(
    #     ax, "Hello, World! How are you?", (0.5, 0.5), 0.4, 0.4, show_rect=True
    # )
    plt.show()
