import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    results_dir = "../../results/artificial_splits/"
    rows = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file)) as fp:
                split_results = json.load(fp)

            dataset = file[:-5]  # strip .json
            avg_percent_split = split_results["percent_split"]["avg_percent_split"]

            for method in METHODS:
                if method not in split_results:
                    continue
                data = split_results[method]
                avg_recall = float(np.mean(data["raw_recall"]))
                avg_percent_merged = float(np.mean(data["raw_percent_merged"]))
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "avg_recall": avg_recall,
                        "merged_split_ratio": avg_percent_merged / avg_percent_split,
                    }
                )

    df = pd.DataFrame(rows)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_color = {m: colors[i] for i, m in enumerate(METHODS)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in METHODS:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["avg_recall"],
            sub["merged_split_ratio"],
            color=method_color[method],
            marker="o",
            s=50,
            label=METHOD_LABELS[method],
            zorder=3,
            alpha=0.5,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal ratio = 1")
    ax.set_xlabel("Average Recall")
    ax.set_ylabel("Avg Percent Merged / Avg Percent Split")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_title("Recall vs. Merge Ratio")

    plt.tight_layout()
    out_path = "../../results/artificial_splits/as_scatterplot.svg"
    plt.savefig(out_path, transparent=True, dpi=300)
    print(f"Saved to {out_path}")
    plt.show()
