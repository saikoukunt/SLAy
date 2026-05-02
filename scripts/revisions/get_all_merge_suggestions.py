import json
import os
import sys
import numpy as np
from spikeinterface.core import load_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups

from slay import compute_slay_merges
from slay.autoselect_params import evaluate_merge_predictions


def _load_auto_params(sorting_name, parameters_path):
    with open(os.path.join(parameters_path, f"{sorting_name}.json")) as f:
        params = json.load(f)
    slay_auto_params = params["slay_auto"]["parameters"]
    slay_l2_auto_params = params["slay_l2_auto"]["parameters"]
    si_auto_params = params["si_auto"]["parameters"]

    return slay_auto_params, slay_l2_auto_params, si_auto_params


def _get_one_method_merges(preset, sorting_analyzer, model_path=None, **kwargs):
    if preset == "slay":
        merges, split_analyzer, _ = compute_slay_merges(
            sorting_analyzer, model_path=model_path, **kwargs
        )
    elif preset == "si":
        merges = compute_merge_unit_groups(sorting_analyzer, **kwargs)
    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    return merges


def get_all_merge_suggestions(
    sorting_analyzer, sorting_name, model_path, results_path, parameters_path
):
    slay_auto_params, slay_l2_auto_params, si_auto_params = _load_auto_params(
        sorting_name, parameters_path
    )
    method_configs = [
        (
            "slay_auto",
            "slay",
            {
                "merge_parameters": {"k1": 0.25, "k2": 1, "merge_threshold": 0.45},
                "retrain_autoencoder": False,
            },
        ),
        (
            "slay_auto_l2",
            "slay",
            {"merge_parameters": "auto", "similarity_type": "l2"},
        ),
        ("si", "si", {}),
        ("si_auto", "si", {"steps_params": si_auto_params}),
    ]

    output = {}
    for name, preset, kwargs in method_configs:
        merges = _get_one_method_merges(
            preset, sorting_analyzer, model_path=model_path, **kwargs
        )
        output[name] = merges

    output_file = os.path.join(
        results_path,
        f"{sorting_name}.json",
    )

    with open(output_file, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            default=lambda x: (
                int(x) if isinstance(x, np.integer) else TypeError(repr(x))
            ),
        )

    print(f"Saved merges to: {output_file}")


if __name__ == "__main__":
    ks_folder = sys.argv[1]
    sorting_name = sys.argv[2]

    parameters_path = "../../results/artificial_splits"
    results_path = "../../results/merge_suggestions/"
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")
    model_path = os.path.join(ks_folder, "automerge", "vanilla_autoencoder.pt")

    sorting_analyzer = load_sorting_analyzer(path_to_analyzer)

    get_all_merge_suggestions(
        sorting_analyzer, sorting_name, model_path, results_path, parameters_path
    )
