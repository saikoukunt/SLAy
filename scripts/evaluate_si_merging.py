import numpy as np
from spikeinterface.core import SortingAnalyzer
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.curation.auto_merge import (
    check_improve_contaminations_score,
    compute_cross_contaminations,
)
from spikeinterface.curation.curation_tools import resolve_merging_graph
from spikeinterface.qualitymetrics import compute_refrac_period_violations

import slay
from slay.autoselect_params import evaluate_merge_predictions

from tqdm import tqdm


def sweep_si_merge_parameters(
    split_analyzer: SortingAnalyzer,
    split_ids,
    split_types,
    merge_preset: str,
):
    parameter_combinations = _generate_si_parameter_combinations(merge_preset)
    _, outputs = compute_merge_unit_groups(
        split_analyzer,
        merge_preset,
        steps_params=parameter_combinations[-1],
        extra_outputs=True,
    )

    pair_mask = np.triu(np.arange(len(split_analyzer.unit_ids)), 1) > 0

    spikes_mask = outputs["num_spikes"]
    pair_mask[spikes_mask, :] = False
    pair_mask[:, spikes_mask] = False

    contaminations, _ = compute_refrac_period_violations(
        split_analyzer, refractory_period_ms=1.0, censored_period_ms=0.3
    )
    contaminations = np.array(list(contaminations.values()))
    contamination_mask = outputs["remove_contaminated"]
    pair_mask[contamination_mask, :] = False
    pair_mask[:, contamination_mask] = False

    pair_mask = pair_mask & (outputs["unit_distances"] <= 150)

    n = len(split_analyzer.unit_ids)
    cics_cache = {
        "result_mask": np.zeros((n, n), dtype=bool),
        "computed_mask": np.zeros((n, n), dtype=bool),
    }
    for i, parameters in enumerate(
        tqdm(parameter_combinations, desc=f"{merge_preset} parameter sweep")
    ):
        merges = _rethreshold_si_metrics(
            split_analyzer,
            outputs,
            pair_mask,
            contaminations,
            parameters,
            merge_preset,
            cics_cache,
        )
        percent_merged, recall, _ = evaluate_merge_predictions(
            merges,
            list(split_ids.values()),
            list(split_types.values()),
            len(split_analyzer.unit_ids),
        )

        parameter_combinations[i]["percent_merged"] = percent_merged
        parameter_combinations[i]["recall"] = recall

    percents_merged = np.array(
        [combo["percent_merged"] for combo in parameter_combinations]
    )
    recalls = np.array([combo["recall"] for combo in parameter_combinations])
    pareto_indices = slay.autoselect_params.get_pareto_frontier(
        percents_merged, recalls
    )

    return (
        [parameter_combinations[i] for i in pareto_indices],
        percents_merged[pareto_indices],
        recalls[pareto_indices],
    )


def _rethreshold_si_metrics(
    split_analyzer, outputs, pair_mask, contaminations, parameters, preset, cics_cache
):
    pair_mask = pair_mask & (
        outputs["templates_diff"]
        < parameters["template_similarity"]["template_diff_thresh"]
    )
    match preset:
        case "similarity_correlograms":
            pair_mask = pair_mask & (
                outputs["correlogram_diff"]
                < parameters["correlogram"]["corr_diff_thresh"]
            )
        case "x_contaminations":
            _, p_values = compute_cross_contaminations(
                split_analyzer,
                pair_mask,
                parameters["cross_contamination"]["cc_thresh"],
                (1.0, 0.3),
                contaminations,
            )
            pair_mask = pair_mask & (p_values > 0.2)
        case _:
            raise NotImplementedError(f"Unknown preset '{preset}'")

    uncached_mask = pair_mask & ~cics_cache["computed_mask"]
    if uncached_mask.any():
        result_mask, _ = check_improve_contaminations_score(
            split_analyzer, uncached_mask, contaminations, 1.5, 1.0, 0.3
        )
        cics_cache["result_mask"][uncached_mask] = result_mask[uncached_mask]
        cics_cache["computed_mask"] |= uncached_mask
    pair_mask = pair_mask & cics_cache["result_mask"]
    ind1, ind2 = np.nonzero(pair_mask)
    merge_pairs = list(
        zip(split_analyzer.unit_ids[ind1], split_analyzer.unit_ids[ind2])
    )
    merge_unit_groups = resolve_merging_graph(split_analyzer.sorting, merge_pairs)

    return merge_unit_groups


def _generate_si_parameter_combinations(
    preset: str = "similarity_correlograms",
):
    parameter_combinations = []
    template_diff_thresh_values = np.arange(0, 1.0, 0.1)

    match preset:
        case "similarity_correlograms":
            corr_diff_thresh_values = np.arange(0, 1.0, 0.1)
            for template_thresh in template_diff_thresh_values:
                for corr_thresh in corr_diff_thresh_values:
                    parameter_combinations.append(
                        {
                            "template_similarity": {
                                "template_diff_thresh": template_thresh
                            },
                            "correlogram": {"corr_diff_thresh": corr_thresh},
                        }
                    )
        case "x_contaminations":
            cc_thresh_values = np.arange(0, 1, 0.1)
            for template_thresh in template_diff_thresh_values:
                for cc_thresh in cc_thresh_values:
                    parameter_combinations.append(
                        {
                            "template_similarity": {
                                "template_diff_thresh": template_thresh
                            },
                            "cross_contamination": {"cc_thresh": cc_thresh},
                        }
                    )
        case _:
            raise NotImplementedError(f"Unknown preset '{preset}'")

    return parameter_combinations
