"""
General utilities for ephys data-wrangling.

Assumes that ephys data is stored in the phy output format.
"""

import argparse
import json
import os
from typing import Any

import cupy as cp
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from scipy import stats
from scipy.signal import butter, sosfiltfilt

import slay


def parse_cmd_line_args() -> dict[str, Any]:
    """
    Parse command line arguments for burst-detector, including plot_units.py, run.py, and custom_metrics.py.
    Returns:
        dict[str, Any]: A dictionary containing the parsed arguments.
    """
    # code implementation...
    parser = argparse.ArgumentParser(description="Parse arguments for burst-detector")
    # Define optional arguments
    parser.add_argument("--input_json", type=str, help="Path to JSON file of arguments")
    parser.add_argument("--KS_folder", type=str, help="Path to Kilosort folder")

    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    args = vars(args)
    if args["input_json"]:
        if os.path.exists(args["input_json"]):
            if args["input_json"].endswith(".json"):
                with open(args["input_json"], "r") as f:
                    json_args = json.load(f)
                    args.update(json_args)
            else:
                parser.error(
                    "Input file must be a JSON file with .json file extension."
                )
        else:
            parser.error(f"File {args['input_json']} does not exist")
    del args["input_json"]
    if not args["KS_folder"]:
        parser.error(
            "Please provide a Kilosort folder via --input_json or --KS_folder."
        )
    # Include unknown arguments
    for key, value in zip(unknown_args[::2], unknown_args[1::2]):
        # remove '--' from key
        key = key[2:]
        args[key] = value
    return args


def parse_kilosort_params(args: dict) -> dict[str, Any]:
    # Process KS params.py file
    ksparam_path = os.path.join(args["KS_folder"], "params.py")
    print(ksparam_path)
    ksparams = {}
    with open(ksparam_path, "r") as f:
        for line in f:
            elem = line.split(sep="=")
            ksparams[elem[0].strip()] = eval(elem[1].strip())
    ksparams['dat_path'] = ksparams['dat_path'][0]
    # print(ksparams['dat_path'])
    ksparams["data_filepath"] = os.path.join(
        args["KS_folder"], ksparams.pop("dat_path")
    )
    ksparams["n_chan"] = ksparams.pop("n_channels_dat")
    args.update(ksparams)
    return args


def load_ks_files(params):
    """
    Load and process Kilosort output files and ephys recording data.

    Args:
        params (dict[str, Any]): Dictionary containing the following keys:
            - "KS_folder" (str): Path to the folder containing Kilosort output files.
            - "data_filepath" (str): Path to the ephys recording data file.
            - "dtype" (str): Data type of the ephys recording.
            - "n_chan" (int): Number of channels in the ephys recording.
            - "pre_samples" (int): Number of samples before the spike peak.
            - "post_samples" (int): Number of samples after the spike peak.

    Returns:
        clusters (NDArray[np.int_]): Array of cluster IDs for each spike.
        cl_labels (pd.DataFrame): DataFrame containing cluster labels.
        channel_pos (NDArray[np.float_]): Array of channel positions.
        data (NDArray[np.int_]): Reshaped ephys recording data.
        n_clust (int): Number of clusters.
        times_multi (list[NDArray[np.float64]]): List of arrays containing spike times for each cluster.
        counts (NDArray[np.int_]): Array containing the number of spikes for each cluster.
    """

    os.makedirs(os.path.join(params["KS_folder"], "automerge"), exist_ok=True)

    # Load sorting and recording info.
    tqdm.write("Loading files...")
    times: NDArray[np.float_] = np.load(
        os.path.join(params["KS_folder"], "spike_times.npy")
    ).flatten()
    clusters: NDArray[np.int_] = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    cl_labels: pd.DataFrame = pd.read_csv(
        os.path.join(params["KS_folder"], "cluster_group.tsv"),
        sep="\t",
        index_col="cluster_id",
    )

    channel_pos: NDArray[np.float_] = np.load(
        os.path.join(params["KS_folder"], "channel_positions.npy")
    )

    if "label" not in cl_labels.columns:
        try:
            cl_labels["label"] = cl_labels["KSLabel"]
        except KeyError:
            cl_labels["label"] = cl_labels["group"]

    # Compute useful cluster info.
    # Load the ephys recording.
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data = np.reshape(rawData, (int(rawData.size / params["n_chan"]), params["n_chan"]))

    n_clust = clusters.max() + 1
    times_multi = slay.find_times_multi(
        times / 30000,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )
    counts = np.array([len(times_multi[i]) for i in range(n_clust)])
    return clusters, cl_labels, channel_pos, data, n_clust, times_multi, counts


def spikes_per_cluster(sp_clust: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Counts the number of spikes in each cluster.

    Args
        sp_clust (NDArray): Spike cluster assignments.

    Returns
        counts_array (NDArray): Number of spikes per cluster, indexed by
            cluster id. Shape is (max_cluster_id + 1,).

    """
    ids, counts = np.unique(sp_clust, return_counts=True)
    counts_array = np.zeros(ids.max() + 1, dtype=int)
    counts_array[ids] = counts

    return counts_array


def find_times_multi(
    sp_times: NDArray[np.float64],
    sp_clust: NDArray[np.int_],
    clust_ids: list[int],
    data: NDArray[np.int_],
    pre_samples: int,
    post_samples: int,
) -> dict[NDArray[np.float64]]:
    """
    Finds all the spike times for each of the specified clusters.

    Args:
        sp_times (NDArray): Spike times (in any unit of time).
        sp_clust (NDArray): Spike cluster assignments.
        clust_ids (NDArray): Clusters for which spike times should be returned.
        data (NDArray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        pre_samples (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        post_samples (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.

    Returns:
        cl_times (dict): found cluster spike times in samples.
    """
    times_multi = {}

    for i in clust_ids:
        cl_spike_times = sp_times[sp_clust == i]
        cl_spike_times = cl_spike_times[
            (cl_spike_times >= pre_samples)
            & (cl_spike_times < data.shape[0] - post_samples)
        ]
        times_multi[i] = cl_spike_times

    return times_multi


def extract_spikes(
    data: NDArray[np.int_],
    times_multi: list[NDArray[np.float64]],
    clust_id: int,
    pre_samples: int,
    post_samples: int,
    max_spikes: int,
) -> NDArray[np.int_]:
    """
    Extracts spike waveforms for the specified cluster.

    If the cluster contains more than `max_spikes` spikes, `max_spikes` random
    spikes are extracted instead.

    Args:
        data (NDArray): Ephys data with shape (# of timepoints, # of channels).
            Should be passed in as an np.memmap for large datasets.
        times_multi (list): Spike times indexed by cluster id.
        clust_id (list): The cluster to extract spikes from
        pre_samples (int): The number of samples to extract before the peak of the
            spike. Defaults to 20.
        post_samples (int): The number of samples to extract after the peak of the
            spike. Defaults to 62.
        max_spikes (int): The maximum number of spikes to extract. If -1, all
            spikes are extracted. Defaults to -1.

    Returns:
        spikes (NDArray): Array of extracted spike waveforms with shape
            (# of spikes, # of channels, # of timepoints).
    """
    times = times_multi[clust_id].astype("int64")
    # spikes cut off by the ends of the recording is handled in times_multi
    # times = times[(times >= pre_samples) & (times < data.shape[0] - post_samples)]

    # Randomly pick spikes if the cluster has too many
    if (max_spikes != -1) and (times.shape[0] > max_spikes):
        np.random.shuffle(times)
        times = times[:max_spikes]

    # Create an array to store the spikes
    # Extract spike data around each spike time and avoid for loops for speed
    start_times = times - pre_samples
    n_spikes = len(start_times)
    n_channels = data.shape[1]
    n_samples = post_samples + pre_samples

    # Create an array to store the spikes
    spikes = np.empty((n_spikes, n_channels, n_samples), dtype=data.dtype)

    # Use broadcasting to create index arrays for slicing
    row_indices = np.arange(n_samples).reshape(-1, 1) + start_times

    # Extract the spikes using advanced indexing
    spikes = data[row_indices, :].transpose(
        1, 2, 0
    )  # Shape (n_spikes, n_channels, n_samples)

    return spikes


def calc_mean_wf(
    params: dict[str, Any],
    n_clusters: int,
    cluster_ids: list[int],
    times_multi: dict[NDArray[np.int_]],
    data: NDArray[np.int_],
) -> NDArray:
    """
    Calculate mean waveform and std waveform for each cluster. Need to have loaded some metrics. If return_spikes is True, also returns the spike waveforms.
    Use GPU acceleration with cupy. If the mean waveform is the incorrect shape, it will be recalculated.

    Args:
        params (dict): Parameters for the recording.
        n_clusters (int): Number of clusters in the recording. Equal to the maximum cluster id + 1.
        cluster_ids (list): List of cluster ids to calculate waveforms for.
        times_multi (dict): Dictionary of spike times indexed by cluster id.
        data (NDArray): Ephys data with shape (n_timepoints, n_channels).

    Returns:
        NDArray: Mean waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        NDArray: Std waveforms for each cluster (uV). Shape (n_clusters, n_channels, pre_samples + post_samples) dtype float32
        dict[int, NDArray]: Spike waveforms for each cluster (bits). NDArray shape (n_spikes, n_channels, pre_samples + post_samples) dtype int16
    """
    mean_wf_path = os.path.join(params["KS_folder"], "mean_waveforms.npy")

    if os.path.exists(mean_wf_path):
        mean_wf = np.load(mean_wf_path)
        # recalculate mean_wf if it is not the right shape
        if mean_wf.shape[0] == n_clusters:
            if np.any(np.isnan(mean_wf)):
                mean_wf = np.nan_to_num(mean_wf, nan=0)
                # save the fixed mean waveform
                np.save(mean_wf_path, mean_wf)

            return mean_wf

    mean_wf = cp.zeros(
        (
            n_clusters,
            params["n_chan"],
            params["pre_samples"] + params["post_samples"],
        )
    )
    for i in tqdm(cluster_ids, desc="Calculating mean waveforms"):
        spikes = extract_spikes(
            data,
            times_multi,
            i,
            params["pre_samples"],
            params["post_samples"],
            params["max_spikes"],
        )
        if len(spikes) > 0:  # edge case
            spikes_cp = cp.array(spikes, dtype=cp.float32)
            mean_wf[i, :, :] = cp.mean(spikes_cp, axis=0)

    # Convert back to numpy arrays for compatibility
    mean_wf = cp.asnumpy(mean_wf)

    return mean_wf


def calc_sliding_RP_viol(
    times_multi: dict[NDArray[np.float64]],
    clust_ids: NDArray[np.int_],
    bin_size=1,
    acceptThresh: float = 0.15,
    window_size: float = 2,
    overlap_tol: int = 5,
):
    RP_viol = {}

    for i in tqdm(clust_ids, desc="Calculating RP viol confs"):
        times = times_multi[i] / 30000  # convert times to seconds

        if times.shape[0] <= 1:
            RP_viol[i] = 0
        else:
            acg = slay.auto_correlogram(
                times, window_size, bin_size / 1000, overlap_tol / 30000
            )
            RP_viol[i] = _sliding_RP_viol(
                acg,
                bin_size,
                acceptThresh,
            )

    return RP_viol


def _sliding_RP_viol(
    correlogram,
    bin_size: float = 1,
    acceptThresh: float = 0.15,
) -> float:
    """
    Calculate the sliding refractory period violation confidence for a cluster.
    Args:
        correlogram (NDArray): The auto-correlogram of the cluster.
        bin_size (float, optional): The size of each bin in ms. Defaults to 0.25.
        acceptThresh (float, optional): The threshold for accepting refractory period violations. Defaults to 0.1.
    Returns:
        float: The refractory period violation confidence for the cluster.
    """
    # create various refractory periods sizes to test (between 0 and 20x bin size)
    b = np.arange(0, 21 * bin_size, bin_size) / 1000
    bTestIdx = np.array([1, 2, 4, 6, 8, 12, 16, 20], dtype="int8")
    bTest = [b[i] for i in bTestIdx]

    # calculate and avg halves of acg to ensure symmetry
    # keep only second half of acg, refractory period violations are compared from the center of acg
    half_len = int(correlogram.shape[0] / 2)
    correlogram = (correlogram[half_len:] + correlogram[:half_len][::-1]) / 2

    acg_cumsum = np.cumsum(correlogram)
    sum_res = acg_cumsum[bTestIdx - 1]  # -1 bc 0th bin corresponds to 0-bin_size ms

    # low-pass filter acg and use max as baseline event rate
    order = 4  # Hz
    cutoff_freq = 250  # Hz
    fs = 1 / bin_size * 1000
    nyqist = fs / 2
    cutoff = cutoff_freq / nyqist
    sos = butter(order, cutoff, btype="low", output="sos")
    smoothed_acg = sosfiltfilt(sos, correlogram)

    bin_rate_max = np.max(smoothed_acg)
    max_conts_max = np.array(bTest) / bin_size * 1000 * (bin_rate_max * acceptThresh)
    # compute confidence of less than acceptThresh contamination at each refractory period
    confs = 1 - stats.poisson.cdf(sum_res, max_conts_max)
    rp_viol = 1 - confs.max()

    return rp_viol


### @internal
def get_closest_channels(
    channel_positions: NDArray[np.float64], ref_chan: int, num_close: int | None = None
) -> NDArray[np.int_]:
    """
    Gets the channels closest to a specified channel on the probe.

    Args:
        channel_positions (NDArray): The XY coordinates of each channel on the
            probe (in arbitrary units)
        ref_chan (int): The index of the channel to calculate distances relative to.
        num_close (int, optional): The number of closest channels to return,
            including `ref_chan`.

    Returns:
        close_chans (NDArray): The indices of the closest channels, sorted from
            closest to furthest. Includes the ref_chan.

    """
    x, y = channel_positions[:, 0], channel_positions[:, 1]
    x0, y0 = channel_positions[ref_chan]

    dists: NDArray[np.float64] = (x - x0) ** 2 + (y - y0) ** 2
    close_chans = np.argsort(dists)
    if num_close:
        close_chans = close_chans[:num_close]
    return close_chans


def find_best_channels(
    template: NDArray[np.float64],
    channel_pos: NDArray[np.float64],
    n_chan: int,
    num_close: int,
) -> tuple[NDArray[np.int_], int]:
    """
    For a given waveform, finds the channels with the largest amplitude.

    Args:
        template (NDArray): The waveform to find the best channels for.
        channel_pos (NDArray): The XY coordinates of each channel on the probe
            (in arbitrary units).
        n_chan (int): The number of channels on the probe.
        num_close (int): The number of closest channels to return, including the
            peak channel (channel with largest amplitude).

    Returns:
        close_chans (NDArray): The indices of the closest channels, sorted
            from closest to furthest. Includes the peak channel.
        peak_chan (int): The index of the peak channel.

    """
    amplitude: NDArray[np.float64] = template.max(axis=1) - template.min(axis=1)
    peak_channel: int = min(int(np.argmax(amplitude)), n_chan - 3)
    close_chans: NDArray[np.int_] = get_closest_channels(
        channel_pos, peak_channel, num_close
    )

    return close_chans, peak_channel


def get_dists(
    channel_positions: NDArray[np.float64],
    ref_chan: int,
    target_chans: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Calculates the distance from a specified channel on the probe to a set of
    target channels.

    Args:
        channel_positions (NDArray): XY coordinates of each channel on the
            probe (in arbitrary units)
        ref_chan (int): The index of the channel to calculate distances relative to.
        target_chans (NDArray): The set of target channels.

    Returns:
        dists (NDArray): Distances to each of the target channels, in the same
            order as `target_chans`.
    """
    x: NDArray[np.float64] = channel_positions[:, 0]
    y: NDArray[np.float64] = channel_positions[:, 1]
    x0: float
    y0: float

    x0, y0 = channel_positions[ref_chan]
    dists: NDArray[np.float64] = (x - x0) ** 2 + (y - y0) ** 2
    dists = dists[target_chans]
    return dists


def calc_fr_unif(
    spike_times: list[np.int_],
    old2new: dict[int, int],
    new2old: dict[int, list[int]],
    times_multi: list[NDArray[np.int_]],
) -> tuple[NDArray, NDArray]:
    """
    Calculate firing rate uniformity metrics for spike trains.
    Args:
        spike_times (list): List of spike times for each cluster.
        old2new (dict): Dictionary mapping old cluster IDs to new cluster IDs.
        new2old (dict): Dictionary mapping new cluster IDs to old cluster IDs.
        times_multi (dict): Dictionary mapping cluster IDs to spike times.
    Returns:
        single_ds (NDArray): Array of firing rate uniformity metrics for individual clusters.
        merged_ds (NDArray): Array of firing rate uniformity metrics for merged clusters.
    """
    merged_ds = np.zeros(len(new2old.keys()))
    for i in range(len(new2old.keys())):
        spike_times = []
        for clust in new2old[list(new2old.keys())[i]]:
            spike_times.append(times_multi[clust])

        spike_times = np.concatenate(spike_times)
        c1, _ = slay.bin_spike_trains(spike_times, spike_times, 20)
        n = c1.shape[0]
        merged_ds[i] = 1 - wasserstein_distance(
            u_values=np.arange(n) / n,
            v_values=np.arange(n) / n,
            u_weights=c1,
            v_weights=np.ones(n),
        )

    single_ds = np.zeros(len(old2new.keys()))
    for i in range(len(old2new.keys())):
        clust = int(list(old2new.keys())[i])
        spike_times = times_multi[clust]
        c1, _ = slay.bin_spike_trains(spike_times, spike_times, 20)
        n = c1.shape[0]
        single_ds[i] = 1 - wasserstein_distance(
            u_values=np.arange(n) / n,
            v_values=np.arange(n) / n,
            u_weights=c1,
            v_weights=np.ones(n),
        )

    return single_ds, merged_ds


def temp_mismatch(
    clust_id: int,
    templates: list,
    channel_pos: NDArray[np.float64],
    n_chan: int,
    num_close: int,
    mean_wf: NDArray[np.float64],
) -> float:
    """
    Calculate the temporal mismatch between the proximity ranks and amplitude ranks of a given cluster.
    Args:
        clust_id (int): The ID of the cluster.
        templates (list): List of templates.
        channel_pos (NDArray): Array of channel positions.
        n_chan (int): Number of channels.
        num_close (int): Number of closest channels to consider.
        mean_wf (NDArray): Array of mean waveforms.
    Returns:
        mismatch (float): The magnitude and direction of the temporal mismatch.
    """
    ch_ids, peak_channel = find_best_channels(
        templates[clust_id], channel_pos, n_chan, num_close
    )

    # calculate and rank distances (proximity)
    dists = get_dists(channel_pos, peak_channel, ch_ids)
    prox_order = np.argsort(dists)
    prox_ranks = np.argsort(prox_order)

    # calculate and rank amplitudes
    means = mean_wf[clust_id, ch_ids, :]
    amp = means.max(axis=1) - means.min(axis=1)
    amp_order = np.argsort(amp)
    amp_ranks = np.argsort(amp_order)

    # calculate magnitude and direction of mismatch
    mismatch = np.abs(
        (prox_ranks[prox_order] - amp_ranks[prox_order])[
            int(ch_ids.shape[0] / 2) :
        ].sum()
        / 26
    )  # 26 is maximum possible raw mismatch
    return mismatch
