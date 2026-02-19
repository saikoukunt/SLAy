from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from spikeinterface.core import SortingAnalyzer
from torch import nn

from .algorithm import find_merges
from .artificial_splits import make_artificial_splits
from .autoencoder import (
    CN_AE,
    SpikeDataset,
    compute_autoencoder_similarity,
    extract_spike_snippets,
)
from .metrics import (
    compute_ccg_metric,
    compute_final_metric,
    compute_refractory_penalty,
)


def autoselect_merge_parameters(
    sorting_analyzer: SortingAnalyzer,
    similarity_type: str = "autoencoder",
    autoencoder_params: dict[str, Any] = {
        "num_chan": 8,
    },
    autoencoder: nn.Module = None,
    model_path: str = None,
    similarity_threshold: float = 0.4,
    correlogram_params: dict[str, Any] = {
        "window_ms": 100,
        "bin_ms": 0.5,
        "method": "auto",
    },
    maximum_contamination: float = 0.15,
    parameter_combinations: NDArray[np.floating] = None,
):
    # compute metric for an analyzer with artificial splits
    split_analyzer, split_pairs, widowed_ids = make_artificial_splits(
        sorting_analyzer, splitting_probability=0.3
    )

    if similarity_type == "autoencoder":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = CN_AE().to(device)
        autoencoder.load_state_dict(torch.load(model_path, map_location=device))
        spike_snippets, unit_ids = extract_spike_snippets(
            split_analyzer, autoencoder_params
        )
        similarity = compute_autoencoder_similarity(
            split_analyzer, SpikeDataset(spike_snippets, unit_ids), autoencoder
        )
    else:
        similarity = split_analyzer.get_extension("template_similarity").get_data()

    pair_mask = similarity >= similarity_threshold

    correlogram_extension = split_analyzer.get_extension("correlograms")
    if (
        correlogram_extension is None
        or correlogram_extension.params["window_ms"] != correlogram_params["window_ms"]
        or correlogram_extension.params["bin_ms"] != correlogram_params["bin_ms"]
    ):
        correlogram_extension = split_analyzer.compute(
            "correlograms", **correlogram_params, save=False
        )
    correlograms, _ = correlogram_extension.get_data()

    ccg_metric = compute_ccg_metric(
        split_analyzer, correlograms, correlogram_extension.params["bin_ms"], pair_mask
    )
    refractory_penalty = compute_refractory_penalty(
        split_analyzer,
        correlograms,
        correlogram_extension.params["bin_ms"],
        maximum_contamination,
        pair_mask,
    )

    best_f1 = 0
    best_scores = 0
    best_merges = None
    best_parameters = None
    if parameter_combinations is None:
        parameter_combinations = generate_parameter_combinations()

    for parameters in parameter_combinations:
        final_metric = compute_final_metric(
            similarity, ccg_metric, refractory_penalty, parameters
        )
        merges = find_merges(
            split_analyzer, final_metric, parameters["merge_threshold"]
        )
        (
            num_tp,
            num_fp,
            num_fn,
            precision,
            recall,
            f1,
            tp_merges,
            fp_merges,
            fn_merges,
        ) = evaluate_merge_predictions(merges, split_pairs, widowed_ids)

        if f1 > best_f1:
            best_f1 = f1
            best_scores = {
                "num_tp": num_tp,
                "num_fp": num_fp,
                "num_fn": num_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            best_merges = {"tp": tp_merges, "fp": fp_merges, "fn": fn_merges}
            best_parameters = parameters

    return (
        best_parameters,
        best_scores,
        best_merges,
        split_analyzer,
        split_pairs,
        widowed_ids,
    )


def evaluate_merge_predictions(predicted_merges, true_pairs, widowed_ids):

    merge_partners = {}
    for merge in predicted_merges:
        for unit_id in merge:
            merge_partners[unit_id] = [
                partner_id for partner_id in merge if partner_id != unit_id
            ]

    num_fp = 0
    num_tp = 0
    num_fn = 0
    tp_merges = []
    fp_merges = []
    fn_merges = []

    for pair in true_pairs:
        if pair[0] not in merge_partners or pair[1] not in merge_partners[pair[0]]:
            num_fn += 1
            fn_merges.append([pair[0], pair[1]])
            if pair[0] in merge_partners:
                num_fp += 1
                fp_merges.append([pair[0]] + merge_partners[pair[0]])
            if pair[1] in merge_partners:
                num_fp += 1
                fp_merges.append([pair[1]] + merge_partners[pair[1]])
        else:
            num_tp += 1
            tp_merges.append([pair[0]] + merge_partners[pair[0]])

    counted_widows = set()
    for unit_id in widowed_ids:
        if unit_id in merge_partners.keys() and unit_id not in counted_widows:
            if all(partner in widowed_ids for partner in merge_partners[unit_id]):
                num_fp += 1
            else:
                num_fp += 0.5
            fp_merges.append(
                [unit_id] + [partner for partner in merge_partners[unit_id]]
            )

            counted_widows.add(unit_id)
            for partner_id in merge_partners[unit_id]:
                counted_widows.add(partner_id)

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return (
        num_tp,
        num_fp,
        num_fn,
        precision,
        recall,
        f1,
        tp_merges,
        fp_merges,
        fn_merges,
    )


def generate_parameter_combinations(
    k1_values=np.arange(0.0, 0.55, 0.05),
    k2_values=np.arange(0, 1.5, 0.25),
    merge_threshold_values=np.arange(0.4, 0.8, 0.05),
):
    parameter_combinations = []
    for k1 in k1_values:
        for k2 in k2_values:
            for merge_threshold in merge_threshold_values:
                parameter_combinations.append(
                    {"k1": k1, "k2": k2, "merge_threshold": merge_threshold}
                )

    return parameter_combinations
