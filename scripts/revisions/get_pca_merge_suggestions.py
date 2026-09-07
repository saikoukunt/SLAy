import json
import os
import sys
import numpy as np
from spikeinterface.core import load_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups

from slay import compute_slay_merges
from slay.autoselect_params import evaluate_merge_predictions

if __name__ == "__main__":
    ks_folder = sys.argv[1]
    sorting_name = sys.argv[2]

    results_path = "../../results/merge_suggestions/pca"
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")
    sorting_analyzer = load_sorting_analyzer(path_to_analyzer)

    merges, analyzer, _ = compute_slay_merges(
        sorting_analyzer, similarity_type="pca", merge_parameters="auto"
    )

    output_file = os.path.join(
        results_path,
        f"{sorting_name}.json",
    )

    with open(output_file, "w") as f:
        json.dump(
            merges,
            f,
            indent=2,
            default=lambda x: (
                int(x) if isinstance(x, np.integer) else TypeError(repr(x))
            ),
        )
