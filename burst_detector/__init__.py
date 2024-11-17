import logging
import sys

import burst_detector.custom_metrics as custom_metrics
import burst_detector.run as run
from burst_detector.algo import run_merge
from burst_detector.autoencoder import (
    CN_AE,
    SpikeDataset,
    generate_train_data,
    train_ae,
)
from burst_detector.bursts import base_algo, find_bursts
from burst_detector.cluster_metrics import calc_wf_norms, wf_means_similarity
from burst_detector.plot import plot_corr, plot_merges, plot_wfs
from burst_detector.stages import (
    calc_ae_sim,
    calc_mean_sim,
    calc_ref_p,
    calc_xcorr_metric,
    merge_clusters,
)
from burst_detector.utils import (
    calc_mean_and_std_wf,
    extract_spikes,
    find_best_channels,
    find_times_multi,
    get_dists,
    parse_cmd_line_args,
    parse_kilosort_params,
    spikes_per_cluster,
)
from burst_detector.xcorr import (
    auto_correlogram,
    bin_spike_trains,
    x_correlogram,
    xcorr_sig,
)

logger = logging.getLogger("burst-detector")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)

logger.addHandler(console)
logger.propagate = False
