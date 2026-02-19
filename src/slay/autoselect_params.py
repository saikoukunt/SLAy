from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from spikeinterface.core import SortingAnalyzer
from torch import nn

from .algorithm import find_merges, compute_slay_metrics
from .artificial_splits import make_artificial_splits
from .autoencoder import (
    CN_AE,
    SpikeDataset,
    compute_autoencoder_similarity,
    extract_spike_snippets,
    train_ae,
)
from .metrics import (
    compute_ccg_metric,
    compute_final_metric,
    compute_refractory_penalty,
)

from kneed import KneeLocator


def autoselect_merge_parameters(
    sorting_analyzer: SortingAnalyzer,
    splitting_probability,
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
    split_analyzer, split_pairs = make_artificial_splits(
        sorting_analyzer, splitting_probability=0.3
    )

    similarity, ccg_metric, refractory_penalty = compute_slay_metrics(
        split_analyzer,
        autoencoder_params,
        CN_AE,
        train_ae,
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
        list(split_pairs.values()),
        similarity,
        ccg_metric,
        refractory_penalty,
    )
    pareto_percents, pareto_recalls = get_pareto_frontier(percents_merged, recalls)

    knee_locator = KneeLocator(pareto_percents, pareto_recalls)
    best_percent_merged = knee_locator.knee
    best_recall = knee_locator.knee_y
    autoselected_parameters = get_closest_parameters(
        parameter_combinations, best_percent_merged, best_recall
    )

    return autoselected_parameters, parameter_combinations


def compute_parameter_performances(
    parameter_combinations,
    split_analyzer,
    split_pairs,
    similarity,
    ccg_metric,
    refractory_penalty,
):
    if parameter_combinations is None:
        parameter_combinations = generate_parameter_combinations()

    for i, parameters in enumerate(parameter_combinations):
        final_metric = compute_final_metric(
            similarity, ccg_metric, refractory_penalty, parameters
        )
        merges = find_merges(
            split_analyzer, final_metric, parameters["merge_threshold"]
        )
        percent_merged, recall, _, _ = evaluate_merge_predictions(
            merges, split_pairs, len(split_analyzer.unit_ids)
        )

        parameter_combinations[i]["percent_merged"] = percent_merged
        parameter_combinations[i]["recall"] = recall

    percents_merged = np.array(
        [combo["percent_merged"] for combo in parameter_combinations]
    )
    recalls = np.array([combo["recall"] for combo in parameter_combinations])

    return parameter_combinations, percents_merged, recalls


def evaluate_merge_predictions(predicted_merges, true_pairs, num_units):
    merge_partners = {}
    for merge in predicted_merges:
        for unit_id in merge:
            merge_partners[unit_id] = [
                partner_id for partner_id in merge if partner_id != unit_id
            ]
    num_tp = 0
    num_fn = 0
    tp_merges = []
    fn_merges = []

    for pair in true_pairs:
        if pair[0] not in merge_partners or pair[1] not in merge_partners[pair[0]]:
            num_fn += 1
            fn_merges.append([pair[0], pair[1]])
        else:
            num_tp += 1
            tp_merges.append([pair[0]] + merge_partners[pair[0]])

    percent_merged = sum([len(merge) for merge in predicted_merges]) / num_units
    recall = num_tp / len(true_pairs)

    return percent_merged, recall, tp_merges, fn_merges


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


def get_pareto_frontier(percents_merged, recalls):
    num_combinations = recalls.shape[0]
    is_pareto = np.ones(num_combinations, dtype=bool)

    for i in range(num_combinations):
        for j in range(num_combinations):
            if recalls[j] > recalls[i] and percents_merged[j] < percents_merged[i]:
                is_pareto[i] = False
                break
    pareto_indices = np.argwhere(is_pareto).flatten()

    return percents_merged[pareto_indices], recalls[pareto_indices]


def get_closest_parameters(parameter_combinations, percent_merged, recall):
    knee_parameters = None
    min_distance = float("inf")

    for parameters in parameter_combinations:
        # Calculate Euclidean distance to knee point
        distance = np.sqrt(
            (parameters["percent_merged"] - percent_merged) ** 2
            + (parameters["recall"] - recall) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            knee_parameters = parameters

    return knee_parameters
