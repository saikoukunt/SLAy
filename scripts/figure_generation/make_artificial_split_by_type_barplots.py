import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json


def extract_rows_from_recall_dicts(raw_recall_by_type, method, filename):
    rows = []
    for recall_dict in raw_recall_by_type:
        try:
            for split_type, recall in recall_dict.items():
                rows.append(
                    {
                        "filename": filename,
                        "method": method,
                        "split_type": split_type,
                        "recall": recall,
                    }
                )
        except AttributeError:
            print(filename, method)
    return rows


if __name__ == "__main__":
    results_dir = "../../results/artificial_splits/"
    all_rows = []

    methods = [
        ("slay_auto", "SLAy"),
        ("si", "SI"),
        ("si_auto", "SI auto"),
    ]

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as fp:
                    split_results = json.load(fp)

                filename = file[:-5]

                for key, method_label in methods:
                    if key not in split_results:
                        continue
                    all_rows.extend(
                        extract_rows_from_recall_dicts(
                            split_results[key]["raw_recall_by_type"],
                            method_label,
                            filename,
                        )
                    )

    df = pd.DataFrame(all_rows)

    for split_type in df["split_type"].unique():
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.barplot(
            data=df[df["split_type"] == split_type],
            x="filename",
            y="recall",
            hue="method",
            ax=ax,
        )
        ax.set_xlabel("Dataset")
        ax.set_ylabel(f"Recall on {split_type} splits")
        plt.ylim([0, 1])
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            f"../../results/artificial_splits/as_recall_on_{split_type}.svg",
            transparent=True,
            dpi=300,
        )
        plt.close(fig)
