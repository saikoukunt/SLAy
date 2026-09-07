import argparse
import json
from collections import defaultdict
from pathlib import Path


def _normalize_inner_list(inner_list):
    return tuple(sorted(inner_list))


def analyze_runs(run_lists):
    run_sets = []
    appearances = defaultdict(set)

    for run_index, run in enumerate(run_lists):
        normalized_run = {_normalize_inner_list(inner_list) for inner_list in run}
        run_sets.append(normalized_run)

        for normalized_inner_list in normalized_run:
            appearances[normalized_inner_list].add(run_index)

    all_runs = set(range(len(run_lists)))
    union_inner_lists = sorted(appearances)
    common_inner_lists = [
        inner_list
        for inner_list in union_inner_lists
        if appearances[inner_list] == all_runs
    ]
    non_common_inner_lists = [
        {
            "inner_list": list(inner_list),
            "present_in_runs": sorted(appearances[inner_list]),
            "missing_from_runs": sorted(all_runs - appearances[inner_list]),
        }
        for inner_list in union_inner_lists
        if appearances[inner_list] != all_runs
    ]

    missing_by_run = {
        str(run_index): [
            list(inner_list) for inner_list in sorted(set(union_inner_lists) - run_set)
        ]
        for run_index, run_set in enumerate(run_sets)
    }

    return {
        "num_runs": len(run_lists),
        "common_inner_lists": [list(inner_list) for inner_list in common_inner_lists],
        "non_common_inner_lists": non_common_inner_lists,
        "missing_by_run": missing_by_run,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the outer lists in a JSON file and report which normalized inner "
            "lists do not appear in every run."
        )
    )
    parser.add_argument("json_path", type=Path, help="Path to the JSON file to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the analysis report as JSON",
    )
    args = parser.parse_args()

    with args.json_path.open() as f:
        run_lists = json.load(f)

    report = analyze_runs(run_lists)

    print(json.dumps(report, indent=2))

    if args.output is not None:
        with args.output.open("w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
