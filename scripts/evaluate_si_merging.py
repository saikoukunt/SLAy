import matplotlib.pyplot as plt
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
from slay.autoselect_params import (
    autoselect_merge_parameters,
    evaluate_merge_predictions,
)
from tqdm import tqdm


def compare_merging_algorithms_on_artificial_splits(
    sorting_analyzer: SortingAnalyzer, model_path
):
    (
        autoselected_parameters,
        parameter_combinations,
        split_analyzer,
        split_ids,
        split_types,
    ) = autoselect_merge_parameters(
        sorting_analyzer,
        0.3,
        similarity_type="autoencoder",
        model_path=model_path,
        random_seed=0,
    )
    recalls = np.array([combo["recall"] for combo in parameter_combinations])
    percents_merged = np.array(
        [combo["percent_merged"] for combo in parameter_combinations]
    )
    pareto_indices = slay.autoselect_params.get_pareto_frontier(
        percents_merged, recalls
    )
    merges, split_analyzer, slay_metrics = slay.compute_slay_merges(
        split_analyzer,
        merge_parameters={"k1": 0.25, "k2": 1, "merge_threshold": 0.5},
        similarity_type="autoencoder",
        model_path=model_path,
    )
    slay_default_percent_merged, slay_default_recall, tp_merges, fn_merges = (
        evaluate_merge_predictions(
            merges,
            list(split_ids.values()),
            list(split_types.values()),
            len(split_analyzer.unit_ids),
        )
    )

    sim_corr_percents_merged, sim_corr_recalls = sweep_si_merge_parameters(
        split_analyzer, split_ids, split_types, "similarity_correlograms"
    )
    merges = compute_merge_unit_groups(split_analyzer)
    sim_corr_default_percent_merged, sim_corr_default_recall, _, _ = (
        evaluate_merge_predictions(
            merges,
            list(split_ids.values()),
            list(split_types.values()),
            len(split_analyzer.unit_ids),
        )
    )

    xcont_percents_merged, xcont_recalls = sweep_si_merge_parameters(
        split_analyzer, split_ids, split_types, "x_contaminations"
    )
    merges = compute_merge_unit_groups(split_analyzer, "x_contaminations")
    xcont_default_percent_merged, xcont_default_recall, tp_merges, fn_merges = (
        evaluate_merge_predictions(
            merges,
            list(split_ids.values()),
            list(split_types.values()),
            len(split_analyzer.unit_ids),
        )
    )

    create_pareto_comparison_plot(
        autoselected_parameters,
        percents_merged,
        recalls,
        pareto_indices,
        slay_default_percent_merged,
        slay_default_recall,
        sim_corr_percents_merged,
        sim_corr_recalls,
        sim_corr_default_percent_merged,
        sim_corr_default_recall,
        xcont_percents_merged,
        xcont_recalls,
        xcont_default_percent_merged,
        xcont_default_recall,
    )


def create_pareto_comparison_plot(
    autoselected_parameters,
    percents_merged,
    recalls,
    pareto_indices,
    slay_default_percent_merged,
    slay_default_recall,
    sim_corr_percents_merged,
    sim_corr_recalls,
    sim_corr_default_percent_merged,
    sim_corr_default_recall,
    xcont_percents_merged,
    xcont_recalls,
    xcont_default_percent_merged,
    xcont_default_recall,
):
    plt.scatter(
        percents_merged[pareto_indices],
        recalls[pareto_indices],
        label="SLAy Pareto frontier",
    )
    plt.scatter(
        autoselected_parameters["percent_merged"],
        autoselected_parameters["recall"],
        c="r",
        label="Best SLAy parameters (knee)",
    )
    plt.scatter(
        slay_default_percent_merged,
        slay_default_recall,
        c="magenta",
        label="Default SLAy parameters",
    )

    plt.scatter(
        sim_corr_percents_merged,
        sim_corr_recalls,
        c="lime",
        label="sim-corr Pareto frontier",
    )
    plt.scatter(
        sim_corr_default_percent_merged,
        sim_corr_default_recall,
        c="g",
        label="sim-corr default",
    )

    plt.scatter(
        xcont_percents_merged,
        xcont_recalls,
        c="cyan",
        label="xcont Pareto frontier",
    )
    plt.scatter(
        xcont_default_percent_merged,
        xcont_default_recall,
        c="red",
        label="xcont default",
    )

    plt.xlabel("% of total clusters merged")
    plt.ylabel("Recall on artificial splits")
    plt.legend()

    plt.show()
    plt.savefig(
        "../figures/revisions/si_merge_comparison/bijan_ks4.svg",
        transparent=True,
    )


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

    for i, parameters in enumerate(
        tqdm(parameter_combinations, desc=f"{merge_preset} parameter sweep")
    ):
        merges = _rethreshold_si_metrics(
            split_analyzer, outputs, pair_mask, contaminations, parameters, merge_preset
        )
        percent_merged, recall, _, _ = evaluate_merge_predictions(
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

    return percents_merged[pareto_indices], recalls[pareto_indices]


def _rethreshold_si_metrics(
    split_analyzer, outputs, pair_mask, contaminations, parameters, preset
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
            CC, p_values = compute_cross_contaminations(
                split_analyzer,
                pair_mask,
                parameters["cross_contamination"]["cc_thresh"],
                (1.0, 0.3),
                contaminations,
            )
            pair_mask = pair_mask & (p_values > 0.2)
        case _:
            raise NotImplementedError(f"Unknown preset '{preset}'")

    pair_mask, _ = check_improve_contaminations_score(
        split_analyzer, pair_mask, contaminations, 1.5, 1.0, 0.3
    )
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
