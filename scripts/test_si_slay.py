import os
import sys

import spikeinterface.full as si
from slay import compute_slay_merges


def main(path_to_analyzer):
    sorting_analyzer = si.load_sorting_analyzer(path_to_analyzer)
    merges, sorting_analyzer, slay_metrics = compute_slay_merges(sorting_analyzer)
    print(merges)


if __name__ == "__main__":
    ks_folder = sys.argv[1]
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")
    main(path_to_analyzer)
