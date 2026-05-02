import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json


METHODS = [
    "slay_auto",
    #    "slay_l2_auto",
    "si",
    "si_auto",
]
METHOD_LABELS = {
    "slay_auto": "SLAy",
    # "slay_l2_auto": "SLAy L2",
    "si": "SI",
    "si_auto": "SI auto",
}


if __name__ == "__main__":
    n_boot = 1000
    bootstrap_seed = 0
    results_dir = "../../results/artificial_splits/"
    rows = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file)) as fp:
                split_results = json.load(fp)

            filename = file[:-5]  # strip .json
            avg_percent_split = split_results["percent_split"]["avg_percent_split"]

            for method in METHODS:
                if method not in split_results:
                    continue
                for recall, pct in zip(
                    split_results[method]["raw_recall"],
                    split_results[method]["raw_percent_merged"],
                ):
                    rows.append(
                        {
                            "filename": filename,
                            "method": METHOD_LABELS[method],
                            "recall": recall,
                            "merged_split_ratio": pct / avg_percent_split,
                        }
                    )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(24, 12))
    sns.barplot(
        data=df,
        x="filename",
        y="recall",
        hue="method",
        errorbar=("ci", 95),
        n_boot=n_boot,
        seed=bootstrap_seed,
        ax=ax,
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Recall")
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(
        "../../results/artificial_splits/as_recall_comparison.svg",
        transparent=True,
        dpi=300,
    )

    fig, ax = plt.subplots(figsize=(24, 12))
    sns.barplot(
        data=df,
        x="filename",
        y="merged_split_ratio",
        hue="method",
        errorbar=("ci", 95),
        n_boot=n_boot,
        seed=bootstrap_seed,
        ax=ax,
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="ideal ratio = 1")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percent Merged / Percent Split")
    plt.tight_layout()
    plt.savefig(
        "../../results/artificial_splits/as_merged_split_ratio_comparison.svg",
        transparent=True,
        dpi=300,
    )

    plt.show()
