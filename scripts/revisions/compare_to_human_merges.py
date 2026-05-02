import sys
import os
import json
import re
from itertools import combinations


def _expand_to_pairs(group):
    return [tuple(sorted(pair)) for pair in combinations(group, 2)]


def _sort_merge_list(merges):
    return set(pair for group in merges for pair in _expand_to_pairs(group))


def _load_si_gui_merges(path_to_json):
    merges = []
    with open(path_to_json) as f:
        raw_merges = json.load(f)["merges"]

    for merge in raw_merges:
        merges.append(merge["unit_ids"])

    return _sort_merge_list(merges)


def load_human_merges(results_path):
    human_1_merges = _load_si_gui_merges(os.path.join(results_path, "human_1.json"))
    try:
        human_2_merges = _load_si_gui_merges(os.path.join(results_path, "human_2.json"))
    except FileNotFoundError:
        print("Only one curator found, returning it twice")

    return human_1_merges, human_2_merges


def load_auto_merges(auto_merges_path, algorithm):
    with open(auto_merges_path) as f:
        auto_merges = _sort_merge_list(json.load(f)[algorithm])

    return auto_merges


def calculate_human_algorithm_agreement(auto_merges, human_1_merges, human_2_merges):
    merged_all = auto_merges & human_1_merges & human_2_merges

    merged_auto_human_1_only = (auto_merges & human_1_merges) - human_2_merges
    merged_auto_human_2_only = (auto_merges & human_2_merges) - human_1_merges
    merged_humans_only = (human_1_merges & human_2_merges) - auto_merges

    merged_auto_only = auto_merges - human_1_merges - human_2_merges
    merged_human_1_only = human_1_merges - auto_merges - human_2_merges
    merged_human_2_only = human_2_merges - auto_merges - human_1_merges

    merge_venn = {
        "Auto and humans": list(merged_all),
        "Auto and human 1 only": list(merged_auto_human_1_only),
        "Auto and human 2 only": list(merged_auto_human_2_only),
        "Humans only\t": list(merged_humans_only),
        "Auto only\t": list(merged_auto_only),
        "Human 1 only\t": list(merged_human_1_only),
        "Human 2 only\t": list(merged_human_2_only),
    }

    counts_venn = {
        "Auto and humans": len(merged_all),
        "Auto and human 1 only": len(merged_auto_human_1_only),
        "Auto and human 2 only": len(merged_auto_human_2_only),
        "Humans only\t": len(merged_humans_only),
        "Auto only\t": len(merged_auto_only),
        "Human 1 only\t": len(merged_human_1_only),
        "Human 2 only\t": len(merged_human_2_only),
    }

    return merge_venn, counts_venn


def compare_one_algo_to_humans(
    human_1_merges, human_2_merges, auto_merges_path, algorithm
):
    auto_merges = load_auto_merges(auto_merges_path, algorithm)
    merge_venn, counts_venn = calculate_human_algorithm_agreement(
        auto_merges, human_1_merges, human_2_merges
    )

    return merge_venn, counts_venn


def compare_all_algos_to_humans(algorithms, results_path, auto_merges_path):
    results = {}
    for algorithm in algorithms:
        results[algorithm] = {}

    human_merges_1, human_merges_2 = load_human_merges(results_path)

    for algorithm in algorithms:
        results[algorithm]["merges"], results[algorithm]["counts"] = (
            compare_one_algo_to_humans(
                human_merges_1, human_merges_2, auto_merges_path, algorithm
            )
        )

    return results


if __name__ == "__main__":
    sorting_name = sys.argv[1]

    results_path = os.path.join("../../results/curation/", sorting_name)
    auto_merges_path = os.path.join(
        "../../results/merge_suggestions/", f"{sorting_name}.json"
    )

    algorithms = ["slay_auto", "slay_auto_l2", "si", "si_auto"]

    results = compare_all_algos_to_humans(algorithms, results_path, auto_merges_path)
    output_path = os.path.join(results_path, "auto_human_comparisons.json")

    json_str = json.dumps(results, indent=2)
    json_str = re.sub(
        r"\[\s+([\d,\s]+?)\s+\]",
        lambda m: (
            "[" + ", ".join(s.strip() for s in m.group(1).split(",") if s.strip()) + "]"
        ),
        json_str,
    )

    with open(output_path, "w") as f:
        f.write(json_str)
