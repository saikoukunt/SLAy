import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from spikeinterface.core import load_sorting_analyzer
from tqdm import trange

from slay import compute_slay_merges


def custom_json_dumps(obj, indent=2, _level=0):
    pad = " " * (indent * _level)
    inner_pad = " " * (indent * (_level + 1))
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{inner_pad}{json.dumps(k)}: {custom_json_dumps(v, indent, _level + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return json.dumps(obj)


def plot_pairwise_similarities(similarities):
    num_runs = len(similarities)

    plt.figure(figsize=(16, 16))
    for i in range(num_runs):
        for j in range(i, num_runs):
            plt.subplot(num_runs, num_runs, i * num_runs + j + 1)
            plt.scatter(similarities[i], similarities[j], s=0.3, alpha=0.2)
            plt.xlabel(f"Run {i} similarity")
            plt.ylabel(f"Run {j} similarity")

            plt.title(
                f"Correlation: {np.corrcoef(similarities[i], similarities[j])[0, 1]:.4f}"
            )

    plt.tight_layout()


if __name__ == "__main__":
    ks_folder = sys.argv[1]
    sorting_name = sys.argv[2]
    num_ae_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    num_split_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    results_path = "../../results/multiple_seed/"
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")

    sorting_analyzer = load_sorting_analyzer(path_to_analyzer)

    merge_lists = {}

    np.zeros((num_ae_seeds, num_split_seeds), dtype=object)
    similarities = []

    for i in trange(num_ae_seeds):
        for j in trange(num_split_seeds):
            torch.manual_seed(i)
            torch.cuda.manual_seed(i)

            model_path = model_path = os.path.join(
                ks_folder, "automerge", f"autoencoder_seed{i}.pt"
            )
            merges, split_analyzer, slay_metrics = compute_slay_merges(
                sorting_analyzer,
                model_path=model_path,
                autoencoder_seed=i,
                artificial_split_seed=j,
                retrain_autoencoder=(j == 0),
            )

            merge_lists[f"ae-{i}, as-{j}"] = {
                "merges": merges,
                "params": slay_metrics["merge_parameters"],
            }
        similarities.append(slay_metrics["similarity"].flatten())

    output_file = os.path.join(results_path, f"{sorting_name}.json")
    with open(output_file, "w") as f:
        f.write(custom_json_dumps(merge_lists))

    plot_pairwise_similarities(similarities)
    figure_path = os.path.join(results_path, f"{sorting_name}.svg")
    plt.savefig(figure_path, dpi=300)
