import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.set_style("whitegrid")
sbn.set_context(font_scale=2)
plt.rcParams["font.size"] = 22


category = "asian"  # "asian"  #  muslim #trans
dataset_file_path = f"{category}.csv"

df = pd.read_csv(dataset_file_path)


last_id = []

for row in df["ID"]:
    last_id.append(row.split(" ")[-1][:-1])
int_list = [int(x) for x in last_id]
df2 = df
df2["Intrinsic Dimension"] = int_list
print(df2["Prompt"])
print(df2["Response"])

df2["Label"] = df2["Label"].replace(0, "Non-Toxic Generation")
df2["Label"] = df2["Label"].replace(1, "Toxic Generation")
max_context_length = df2["Context Length"]
max_context_length = max_context_length.max()


fig = plt.figure(figsize=(12, 9))
print(df2)

ax = sbn.lineplot(
    data=df2,
    x="Context Length",
    y="Intrinsic Dimension",
    hue="Label",
    marker="o",
    markersize=12,
    errorbar=None,
    linewidth=3,
    palette=["tab:blue", "tab:red"],
)


def plot_text(ax, idx, text_x, text_y, color, max_width=700):
    txt = ax.text(
        text_x,
        text_y,
        r"${\bf Prompt}: $"
        f"{df2['Prompt'].iloc[idx]}"
        "\n"
        r"${\bf Answer:}$"
        f"{df2['Response'].iloc[idx]}",
        wrap=True,
        bbox=dict(facecolor=color, edgecolor="gray", boxstyle="round,pad=1"),
        verticalalignment="center",
        horizontalalignment="left",
        fontsize=14,
    )
    txt._get_wrap_line_width = lambda: max_width
    bbox = txt.get_window_extent().transformed(ax.transData.inverted())
    plt.plot(
        [df2["Context Length"].iloc[idx], bbox.x0],
        [df2["Intrinsic Dimension"].iloc[idx], bbox.y0],
        color="gray",
        linewidth=3,
        linestyle="--",
    )


plot_text(
    ax,
    3,
    df2["Context Length"].iloc[0] - 20,
    df2["Intrinsic Dimension"].min() - 20,
    "tab:green",
    500,
)

plot_text(
    ax,
    -4,
    df2["Context Length"].iloc[-3] - 25,
    df2["Intrinsic Dimension"].max() - 40,
    "tab:red",
    560,
)

plt.subplots_adjust(0.1, 0.1, 0.96, 0.99)
plt.savefig(f"{category}.png")
plt.close()
pd.set_option("display.max_colwidth", None)
df2 = df2.drop("Prepended", axis=1).drop("ID", axis=1)

print(str(df2.to_latex(index=False)))
