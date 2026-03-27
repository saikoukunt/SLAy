import os
from collections import deque
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from spikeinterface.core import SortingAnalyzer
from spikeinterface.postprocessing import compute_template_similarity

from .autoencoder import (
    AE,
    compute_autoencoder_similarity,
    extract_spike_snippets,
    train_autoencoder,
)
from .metrics import (
    compute_ccg_metric,
    compute_final_metric,
    compute_refractory_penalty,
)


def compute_slay_merges(
    sorting_analyzer: SortingAnalyzer,
    merge_parameters: Any = "auto",
    splitting_probability: float = 0.3,
    max_distance: int = 60,
    autoencoder_params: dict[str, Any] = {
        "num_chan": 8,
    },
    autoencoder_architecture: type = AE,
    autoencoder_train_fn: Callable = train_autoencoder,
    similarity_threshold: float = 0.4,
    retrain_autoencoder: bool = False,
    model_path: str | None = None,
    correlogram_params: dict[str, Any] = {
        "window_ms": 100,
        "bin_ms": 0.5,
        "method": "auto",
    },
    maximum_contamination: float = 0.15,
    similarity_type: str = "autoencoder",
    **job_kwargs: dict[str, Any],
) -> tuple[list[list[int]], SortingAnalyzer, dict[str, NDArray[np.floating]]]:
    """
    Compute unit merges using the SLAy algorithm.

    This function identifies units that should be merged based on spike waveform similarity,
    cross-correlogram metrics, and refractory period violations. It supports two similarity
    computation methods: autoencoder-based and L2-based template similarity.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing spike sorting results.
    merge_parameters : Any, default: "auto"
        Dictionary containing weights "k1" and "k2" for combining metrics, or "auto"
        to automatically select optimal parameters.
    merge_threshold : float, default: 0.5
        Threshold for the final merge metric above which units are merged.
    max_distance : int, default: 10
        Maximum distance (in channels) between peak channels for a merge to be valid.
    autoencoder_params : dict[str, Any], default: {}
        Parameters for spike snippet extraction and autoencoder training.
        Only used when similarity_type="autoencoder".
    autoencoder_architecture : default: CN_AE
        The autoencoder architecture class to use for similarity computation.
        Only used when similarity_type="autoencoder".
    similarity_threshold : float, default: 0.4
        Minimum similarity threshold for considering unit pairs as merge candidates.
    retrain_autoencoder : bool, default: False
        If True, trains a new autoencoder even if model_path exists.
        Only used when similarity_type="autoencoder".
    model_path : str or None, default: None
        Path to a saved autoencoder model. If provided and exists, loads the model
        instead of training a new one (unless retrain_autoencoder=True).
        Only used when similarity_type="autoencoder".
    correlogram_params : dict[str, Any], default: {"window_ms": 100, "bin_ms": 0.5, "method": "auto"}
        Parameters for computing cross-correlograms.
    maximum_contamination : float, default: 0.15
        Maximum acceptable contamination threshold for refractory period violations.
    similarity_type : str, default: "autoencoder"
        Method for computing similarity: "autoencoder" or "l2".
    job_kwargs : dict[str, Any], default: {}
        Additional keyword arguments for parallel job execution.

    Returns
    -------
    merges : list[list[int]]
        List of merge groups, where each group is a list of unit IDs to merge together.
    sorting_analyzer : SortingAnalyzer
        The input sorting analyzer (potentially modified with new extensions).
    slay_metrics : dict[str, NDArray[np.floating]]
        Dictionary containing computed metrics:
        - "similarity": Pairwise similarity matrix
        - "ccg_metric": Cross-correlogram significance metric matrix
        - "refractory_penalty": Refractory period violation penalty matrix
        - "final_metric": Combined final merge metric matrix

    Raises
    ------
    NotImplementedError
        If an unknown similarity_type is provided.
    """
    # handle unit filtering

    similarity, ccg_metric, refractory_penalty = compute_slay_metrics(
        sorting_analyzer,
        autoencoder_params,
        autoencoder_architecture,
        autoencoder_train_fn,
        similarity_threshold,
        retrain_autoencoder,
        model_path,
        correlogram_params,
        maximum_contamination,
        similarity_type,
        **job_kwargs,
    )

    if merge_parameters == "auto":
        from .autoselect_params import autoselect_merge_parameters

        merge_parameters = autoselect_merge_parameters(
            sorting_analyzer,
            splitting_probability,
            similarity_type,
            similarity,
            ccg_metric,
            refractory_penalty,
        )

    final_metric = compute_final_metric(
        similarity, ccg_metric, refractory_penalty, merge_parameters
    )

    merges = find_merges(
        sorting_analyzer,
        final_metric,
        merge_parameters["merge_threshold"],
        max_distance=max_distance,
    )

    slay_metrics = {
        "similarity": similarity,
        "ccg_metric": ccg_metric,
        "refractory_penalty": refractory_penalty,
        "final_metric": final_metric,
    }

    return merges, sorting_analyzer, slay_metrics


def compute_slay_metrics(
    sorting_analyzer,
    autoencoder_params,
    autoencoder_architecture,
    autoencoder_train_fn,
    similarity_threshold,
    retrain_autoencoder,
    model_path,
    correlogram_params,
    maximum_contamination,
    similarity_type,
    **job_kwargs,
):
    match similarity_type:
        case "autoencoder":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if (
                not retrain_autoencoder
                and model_path is not None
                and os.path.exists(model_path)
            ):
                autoencoder = autoencoder_architecture().to(device)
                autoencoder.load_state_dict(torch.load(model_path, map_location=device))
                spike_dataset = None
            else:
                autoencoder = autoencoder_architecture().to(device)
                spike_snippets, unit_ids = extract_spike_snippets(
                    sorting_analyzer, autoencoder_params
                )
                autoencoder, spike_dataset = autoencoder_train_fn(
                    spike_snippets, unit_ids, autoencoder
                )
                if model_path is not None:
                    torch.save(autoencoder.state_dict(), model_path)

            autoencoder.eval()
            similarity = compute_autoencoder_similarity(
                sorting_analyzer, autoencoder, autoencoder_params, spike_dataset
            )

        case "l2":
            if not sorting_analyzer.has_extension("random_spikes"):
                sorting_analyzer.compute(["random_spikes", "templates"], **job_kwargs)
            elif not sorting_analyzer.has_extension("templates"):
                sorting_analyzer.compute("templates", **job_kwargs)

            similarity_extension = sorting_analyzer.get_extension("template_similarity")
            if (similarity_extension is None) or (
                similarity_extension.params["method"] != "l2"
            ):
                similarity = compute_template_similarity(
                    sorting_analyzer, method="l2", save=False
                )
            else:
                similarity = similarity_extension.get_data()

        case _:
            raise NotImplementedError(f"Unknown similarity_type: {similarity_type}")

    pair_mask = similarity >= similarity_threshold

    correlogram_extension = sorting_analyzer.get_extension("correlograms")
    if (
        correlogram_extension is None
        or correlogram_extension.params["window_ms"] != correlogram_params["window_ms"]
        or correlogram_extension.params["bin_ms"] != correlogram_params["bin_ms"]
    ):
        correlogram_extension = sorting_analyzer.compute(
            "correlograms", **correlogram_params, save=False
        )

    correlograms, _ = correlogram_extension.get_data()

    ccg_metric = compute_ccg_metric(
        sorting_analyzer,
        correlograms,
        correlogram_extension.params["bin_ms"],
        pair_mask,
    )
    refractory_penalty = compute_refractory_penalty(
        sorting_analyzer,
        correlograms,
        correlogram_extension.params["bin_ms"],
        maximum_contamination,
        pair_mask,
    )

    return similarity, ccg_metric, refractory_penalty


def find_merges(
    sorting_analyzer: SortingAnalyzer,
    final_metric: NDArray[np.floating],
    merge_threshold: float,
    max_distance: int = 100,
) -> list[list[int]]:
    """
    Find cluster merges based on final metric values.

    This function identifies multi-way merges between candidate cluster pairs
    by ranking pairs by their final metric and ensuring transitivity of merges.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing unit information.
    final_metric : NDArray[np.floating]
        Pairwise final metric matrix (NxN) where higher values indicate
        greater merge confidence.
    merge_threshold : float
        Minimum final metric value for considering a merge.
    max_distance : int, default: 10
        Maximum distance between peak channels for a merge to be valid.

    Returns
    -------
    merges : list[list[int]]
        List of merge groups, where each group is a list of unit IDs to merge together.
    """
    unit_ids = sorting_analyzer.unit_ids

    # find channel with peak amplitude for each cluster
    templates_ext = sorting_analyzer.get_extension("templates")
    templates = templates_ext.get_templates()
    channel_amplitudes = np.max(templates, axis=1) - np.min(templates, axis=1)
    peak_chan_indices = np.argmax(channel_amplitudes, axis=1)

    # get physical channel locations
    channel_locations = sorting_analyzer.get_channel_locations()
    peak_locations = channel_locations[peak_chan_indices]

    # make rank-ordered list of cluster pairs
    candidate_pairs = np.array(
        np.unravel_index(np.argsort(final_metric.flatten()), shape=final_metric.shape)
    ).T[::-1][::2]

    # threshold list and remove pairs that are too far apart
    merged_pairs = deque()
    pair_ind = 0

    while (
        pair_ind < len(candidate_pairs)
        and final_metric[candidate_pairs[pair_ind, 0], candidate_pairs[pair_ind, 1]]
        > merge_threshold
    ):
        unit1_idx = candidate_pairs[pair_ind, 0]
        unit2_idx = candidate_pairs[pair_ind, 1]

        # compute Euclidean distance between peak channel locations
        peak_distance = np.linalg.norm(
            peak_locations[unit1_idx] - peak_locations[unit2_idx]
        )

        if peak_distance < max_distance:
            merged_pairs.append((unit1_idx, unit2_idx))
        pair_ind += 1

    unit_to_group = {}
    merge_groups = {}
    next_group_id = 0

    # merge units greedily, checking for multi-way merges
    for unit1_idx, unit2_idx in merged_pairs:
        unit1_idx, unit2_idx = int(unit1_idx), int(unit2_idx)

        if unit1_idx not in unit_to_group and unit2_idx not in unit_to_group:
            # do a pairwise merge if neither unit is already merged
            merge_groups[next_group_id] = [unit1_idx, unit2_idx]
            unit_to_group[unit1_idx] = next_group_id
            unit_to_group[unit2_idx] = next_group_id
            next_group_id += 1
        else:
            # get merged group and check all pairwise merges if either unit is already merged
            group1 = (
                merge_groups[unit_to_group[unit1_idx]]
                if unit1_idx in unit_to_group
                else [unit1_idx]
            )
            group2 = (
                merge_groups[unit_to_group[unit2_idx]]
                if unit2_idx in unit_to_group
                else [unit2_idx]
            )
            if group1 == group2:
                continue
            combined = group1 + group2

            if all(
                final_metric[combined[i], combined[j]] >= merge_threshold
                for i in range(len(combined))
                for j in range(i + 1, len(combined))
            ):
                merge_groups[next_group_id] = combined
                for idx in combined:
                    unit_to_group[idx] = next_group_id
                next_group_id += 1

    # remove intermediate ids
    final_merge_ids = list(set(unit_to_group.values()))
    for merge_id in list(set(merge_groups.keys())):
        if merge_id not in final_merge_ids:
            del merge_groups[merge_id]

    # convert unit indices to unit IDs
    return [[int(unit_ids[idx]) for idx in group] for group in merge_groups.values()]
