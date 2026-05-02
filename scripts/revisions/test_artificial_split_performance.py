import json
import os
import sys
import numpy as np
from spikeinterface.core import load_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups

import slay
from slay import (
    autoselect_merge_parameters,
    make_artificial_splits,
)
from slay.algorithm import compute_slay_metrics
from slay.autoselect_params import (
    evaluate_merge_predictions,
    compute_parameter_performances,
    get_best_parameters,
    get_pareto_frontier,
)


sys.path.append("..")
from evaluate_si_merging import sweep_si_merge_parameters

sys.path.append(".")
from compare_to_human_merges import load_human_merges


def test_artificial_split_performance(
    sorting_analyzer,
    sorting_name,
    model_path,
    results_path,
    num_repetitions=5,
    splitting_probability=0.4,
):
    (
        slay_auto_params,
        parameter_combinations,
        split_analyzer,
        split_ids,
        split_types,
    ) = autoselect_merge_parameters(
        sorting_analyzer,
        splitting_probability,
        similarity_type="autoencoder",
        model_path=model_path,
        random_seed=0,
    )

    (
        slay_l2_auto_params,
        parameter_combinations,
        split_analyzer,
        split_ids,
        split_types,
    ) = _autoselect_slay_existing_analyzer(
        split_analyzer, split_ids, split_types, similarity_type="l2"
    )

    si_parameter_combinations, si_percents_merged, si_recalls = (
        sweep_si_merge_parameters(
            split_analyzer, split_ids, split_types, "similarity_correlograms"
        )
    )
    si_autoselected_parameters = slay.autoselect_params.get_best_parameters(
        si_parameter_combinations, si_percents_merged, si_recalls
    )

    method_configs = [
        (
            "slay_auto",
            "slay",
            slay_auto_params,
            {},
            True,
        ),
        (
            "slay_l2_auto",
            "slay",
            slay_l2_auto_params,
            {"similarity_type": "l2"},
            True,
        ),
        ("si", "si", None, {}, False),
        ("si_auto", "si", si_autoselected_parameters, {}, True),
    ]

    percent_split = np.zeros(num_repetitions)
    splits = []
    for i in range(num_repetitions):
        split_analyzer, split_ids, split_types = make_artificial_splits(
            sorting_analyzer, splitting_probability, random_seed=i + 1
        )
        splits.append((split_analyzer, split_ids, split_types))
        percent_split[i] = _calculate_total_splits(
            split_ids, sorting_name, len(split_analyzer.unit_ids)
        )

    os.makedirs(results_path, exist_ok=True)
    output = {
        "percent_split": {
            "raw_percent_split": percent_split.tolist(),
            "avg_percent_split": percent_split.mean(),
        },
    }

    for name, preset, parameters, kwargs, save_parameters in method_configs:
        percent_merged = np.zeros(num_repetitions)
        recall = np.zeros(num_repetitions)
        recall_by_type = np.zeros(num_repetitions, dtype="object")

        for i, (split_analyzer, split_ids, split_types) in enumerate(splits):
            percent_merged[i], recall[i], recall_by_type[i] = _evaluate_merge_algorithm(
                preset,
                split_analyzer,
                split_ids,
                split_types,
                parameters=parameters,
                model_path=model_path,
                **kwargs,
            )

        entry = {}
        if save_parameters:
            entry["parameters"] = parameters
        entry.update(
            {
                "raw_percent_merged": percent_merged.tolist(),
                "raw_recall": recall.tolist(),
                "raw_recall_by_type": recall_by_type.tolist(),
            }
        )
        output[name] = entry
    print(output)

    output_file = os.path.join(
        results_path,
        f"{sorting_name}.json",
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved artificial split performance metrics to: {output_file}")


def _evaluate_merge_algorithm(
    preset,
    split_analyzer,
    split_ids,
    split_types,
    parameters=None,
    model_path=None,
    **kwargs,
):
    if preset == "slay":
        merges, split_analyzer, _ = slay.compute_slay_merges(
            split_analyzer, merge_parameters=parameters, model_path=model_path, **kwargs
        )
    elif preset == "si":
        kwargs = {"steps_params": parameters} if parameters is not None else {}
        merges = compute_merge_unit_groups(split_analyzer, **kwargs)
    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    percent_merged, recall, recall_by_type = evaluate_merge_predictions(
        merges,
        list(split_ids.values()),
        list(split_types.values()),
        len(split_analyzer.unit_ids),
    )
    return percent_merged, recall, recall_by_type


def _autoselect_slay_existing_analyzer(
    split_analyzer,
    split_ids,
    split_types,
    similarity_type,
    similarity_threshold: float = 0.4,
    correlogram_params={
        "window_ms": 100,
        "bin_ms": 0.5,
        "method": "auto",
    },
    maximum_contamination: float = 0.15,
    parameter_combinations=None,
):
    similarity, ccg_metric, refractory_penalty = compute_slay_metrics(
        split_analyzer,
        None,
        None,
        None,
        similarity_threshold,
        False,
        model_path,
        correlogram_params,
        maximum_contamination,
        similarity_type,
    )

    parameter_combinations, percents_merged, recalls = compute_parameter_performances(
        parameter_combinations,
        split_analyzer,
        list(split_ids.values()),
        list(split_types.values()),
        similarity,
        ccg_metric,
        refractory_penalty,
    )
    pareto_indices = get_pareto_frontier(percents_merged, recalls)

    autoselected_parameters = get_best_parameters(
        [parameter_combinations[i] for i in pareto_indices],
        percents_merged[pareto_indices],
        recalls[pareto_indices],
    )
    if autoselected_parameters is None:
        autoselected_parameters = parameter_combinations[
            pareto_indices[0]
        ]  # parameter combinations are ordered from most to least conservative. pick most conservative parameters if we can't distinguish by artificial split performance

    return (
        autoselected_parameters,
        parameter_combinations,
        split_analyzer,
        split_ids,
        split_types,
    )


def _calculate_total_splits(split_ids, sorting_name, num_units):
    human_merges_path = f"../../results/curation/{sorting_name}"

    human_1_merges, human_2_merges = load_human_merges(human_merges_path)
    human_merges = human_1_merges | human_2_merges

    counted_human_units = set()

    num_split = len(split_ids) * 2
    for merge in human_merges:
        for unit in merge:
            if unit not in list(split_ids.keys()) and unit not in counted_human_units:
                num_split += 1
            counted_human_units.add(unit)

    return num_split / num_units


if __name__ == "__main__":
    ks_folder = sys.argv[1]
    sorting_name = sys.argv[2]

    results_path = "../../results/artificial_splits/"
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")
    model_path = os.path.join(ks_folder, "automerge", "vanilla_autoencoder.pt")

    sorting_analyzer = load_sorting_analyzer(path_to_analyzer)

    test_artificial_split_performance(
        sorting_analyzer, sorting_name, model_path, results_path
    )
