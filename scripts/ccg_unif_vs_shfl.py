import functools
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from scipy.signal import butter, find_peaks_cwt, sosfiltfilt
from scipy.stats import wasserstein_distance
from tqdm import tqdm

import slay
from slay.schemas import RunParams


def main():
    ind = 0
    xcorr_times = np.zeros(8)
    shfl_xcorr_times = np.zeros(8)
    pass_ms = np.zeros(8, dtype=object)
    xcorr_sig = np.zeros(8, dtype=object)
    shfl_xcorr_sig = np.zeros(8, dtype=object)

    data_dir = "D:/SLAY_data/"
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if ("ks" in dir_name) and ("orig" not in dir_name):
                ks_dir = os.path.join(root, dir_name)
                args = {"KS_folder": ks_dir}
                args = slay.parse_kilosort_params(args)
                schema = RunParams()
                params = schema.load(args)
                clusters, cl_labels, channel_pos, data, n_clust, times_multi, counts = (
                    slay.load_ks_files(params)
                )
                pass_ms[ind] = run_ae_sim(
                    data, params, counts, cl_labels, times_multi, n_clust, channel_pos
                )
                xcorr_sig[ind], xcorr_times[ind] = time_unif_ccg_sig(
                    times_multi, n_clust, pass_ms[ind], params
                )
                print(f"Time taken for unif ccg sig: {xcorr_times[ind]}")
                shfl_xcorr_sig[ind], shfl_xcorr_times[ind] = time_shfl_ccg_sig(
                    times_multi, n_clust, pass_ms[ind], params
                )
                print(f"Time taken for shfl ccg sig: {shfl_xcorr_times[ind]}")

                ind += 1

    print("All done!")
    np.save(
        "../results/figures/ccg_unif_vs_shfl/pass_ms.npy", pass_ms, allow_pickle=True
    )
    np.save(
        "../results/figures/ccg_unif_vs_shfl/xcorr_sig.npy",
        xcorr_sig,
        allow_pickle=True,
    )
    np.save(
        "../results/figures/ccg_unif_vs_shfl/shfl_xcorr_sig.npy",
        shfl_xcorr_sig,
        allow_pickle=True,
    )
    np.save("../results/figures/ccg_unif_vs_shfl/xcorr_times.npy", xcorr_times)
    np.save(
        "../results/figures/ccg_unif_vs_shfl/shfl_xcorr_times.npy", shfl_xcorr_times
    )


def shuffle_spike_trains(c1_counts, c2_counts, shuffle_bin_width):
    """
    Randomizes spike times in two spike trains. Spike trains are split into bins and the
    spike times in each bin are randomized while preserving the number of spikes per bin.

    Parameters
    ----------
    c1_counts, c2_counts: array-like
        An array containing binned spike counts, like the output of bin_spike_trains.
    shuffle_bin_width: float
        Width in seconds of bins to shuffle within.

    Returns
    -------
    c1_shfl, c2_shfl: array-like
        An array containing shuffled spike times.
    """

    c1_shfl = np.zeros(int(c1_counts.sum()))
    c2_shfl = np.zeros(int(c2_counts.sum()))

    c1_ind = 0
    c2_ind = 0

    for i in range(c1_counts.shape[0]):
        if c1_counts[i] > 0:
            # randomly sample ISIs from exponential distribution with rate equal to the spike rate in the bin
            spks = np.cumsum(
                np.random.exponential(shuffle_bin_width / c1_counts[i], c1_counts[i])
            )
            c1_shfl[c1_ind : c1_ind + c1_counts[i]] = spks + i * shuffle_bin_width
            c1_ind += c1_counts[i]

    for i in range(c2_counts.shape[0]):
        if c2_counts[i] > 0:
            spks = np.cumsum(
                np.random.exponential(shuffle_bin_width / c2_counts[i], c2_counts[i])
            )
            c2_shfl[c2_ind : c2_ind + c2_counts[i]] = spks + i * shuffle_bin_width
            c2_ind += c2_counts[i]

    return np.sort(c1_shfl), np.sort(c2_shfl)


def calc_xgrams(
    c1_times,
    c2_times,
    n_iter=200,
    shuffle_bin_width=0.1,
    window_size=0.1,
    xcorr_bin_width=0.0005,
    overlap_tol=5 / 30000,
):
    """
    Calculates the raw and baseline cross-correlations between two clusters.

    Parameters
    ----------
    c1_times, c2_times: array-like
        An array containing spike times in seconds.
    n_iter: int, optional
        The number of shuffle iterations for the baseline cross-correlation calculation.
        Default value is 200.
    shuffle_bin_width: float, optional
        The width of bins in seconds for spike train shuffling. Default value is .1.
    window_size: float, optional
        The width in seconds of the cross correlogram window. Default value is .25.
    xcorr_bin_width: float, optional
        The width in seconds of the bins for cross correlogram calculation. Default value
        is .001.
    overlap_tol: float, optional
        Overlap tolerance in seconds. Spikes within the tolerance of the
        reference spike time will not be counted for cross correlogram calculation.

    Returns
    -------
    xgram: array-like
        The actual cross-correlogram for the cluster pair.
    shfl_xgram: array_like
        The baseline cross-correlogram for the cluster pair, calculated by averaging
        cross-correlograms for shuffled versions of the spike trains.
    xgram_overlap, shfl_overlap: float
        The number of overlapping spikes for the respective correlograms.
    """

    import slay

    # bin spike trains for shuffling
    c1_counts, c2_counts = slay.bin_spike_trains(c1_times, c2_times, shuffle_bin_width)

    # init
    n_bins = math.ceil(window_size / xcorr_bin_width)
    shfl_xgram = np.zeros(n_bins)

    # do shuffled iterations
    for i in range(n_iter):
        c1_shfl, c2_shfl = shuffle_spike_trains(c1_counts, c2_counts, shuffle_bin_width)

        shfl = slay.x_correlogram(
            c1_shfl, c2_shfl, window_size, xcorr_bin_width, overlap_tol=overlap_tol
        )
        shfl_xgram += shfl / n_iter

    xgram = slay.x_correlogram(
        c1_times, c2_times, window_size, xcorr_bin_width, overlap_tol=overlap_tol
    )

    return xgram, shfl_xgram


def xcorr_func(
    c1: int,
    c2: int,
    times_multi,
    params,
):
    """
    Multithreading function definition for calculating a cross-correlogram for a
    candidate cluster pair.

    Args:
        c1 (int): The ID of the first cluster.
        c2 (int): The ID of the second cluster.
        times_multi (list): Spike times in samples indexed by cluster id.
        params (dict): General burst-detector parameters.

    Returns:
        ccg (NDArray): The computed cross-correlogram.

    """

    # extract spike times
    c1_times = times_multi[c1] / params["sample_rate"]
    c2_times = times_multi[c2] / params["sample_rate"]
    # compute xgrams
    return calc_xgrams(c1_times, c2_times)


def calc_xcorr_metric(
    times_multi,
    n_clust,
    pass_ms,
    params,
):
    """
    Calculates the cross-correlogram significance metric between each candidate pair of
    clusters.

    Args:
        times_multi (list): Spike times in samples indexed by cluster id.
        n_clust (int): The number of clusters, taken to be the largest cluster id + 1.
        pass_ms (NDArray): True if a cluster pair passes the mean similarity
            threshold, false otherwise.
        params (dict): General burst-detector params.

    Returns:
        xcorr_sig (NDArray): The calculated cross-correlation significance metric
            for each cluster pair.
        xgrams (NDArray): Calculated cross-correlograms for each cluster pair.
        null_xgrams (NDArray): Null distribution (uniforms) cross-correlograms
            for each cluster pair.
    """

    # define cross correlogram job
    xcorr_job = functools.partial(xcorr_func, times_multi=times_multi, params=params)

    # run cross correlogram jobsW
    args = []
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if pass_ms[c1, c2]:
                args.append((c1, c2))
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.starmap(xcorr_job, args)

    # convert cross correlogram output to np arrays
    xgrams = np.empty_like(pass_ms, dtype="object")
    shfl_xgrams = np.empty_like(pass_ms, dtype="object")

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]
        xgrams[c1, c2] = res[i][0]
        shfl_xgrams[c1, c2] = res[i][1]

    # compute metric
    xcorr_sig = np.zeros_like(pass_ms, dtype="float64")

    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if pass_ms[c1, c2]:
                xcorr_sig[c1, c2] = calc_xcorr_sig(
                    xgrams[c1, c2],
                    shfl_xgrams[c1, c2],
                    xcorr_bin_width=params["xcorr_bin_width"],
                    min_xcorr_rate=params["min_xcorr_rate"],
                )
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            xcorr_sig[c2, c1] = xcorr_sig[c1, c2]

    return xcorr_sig, xgrams


def calc_xcorr_sig(
    xgram,
    shfl_xgram,
    xcorr_bin_width: float,
    min_xcorr_rate: float,
) -> float:
    """
    Calculates a cross-correlation significance metric for a cluster pair.

    Uses the wasserstein distance between an observed cross-correlogram and a null
    distribution as an estimate of how significant the dependence between
    two neurons is. Low spike count cross-correlograms have large wasserstein
    distances from null by chance, so we first try to expand the window size. If
    that fails to yield enough spikes, we apply a penalty to the metric.

    Args:
        xgram (NDArray): The raw cross-correlogram for the cluster pair.
        xcorr_bin_width (float): The width in seconds of the bin size of the
            input ccgs.
        max_window (float): The largest allowed window size during window
            expansion.
        min_xcorr_rate (float): The minimum ccg firing rate in Hz.

    Returns:
        sig (float): The calculated cross-correlation significance metric.
    """
    # calculate low-pass filtered second derivative of ccg
    fs = 1 / xcorr_bin_width
    cutoff_freq = 100
    nyqist = fs / 2
    cutoff = cutoff_freq / nyqist
    peak_width = 0.002 / xcorr_bin_width

    xgram_2d = np.diff(xgram, 2)
    sos = butter(4, cutoff, output="sos")
    xgram_2d = sosfiltfilt(sos, xgram_2d)

    # find negative peaks of second derivative of ccg, these are the edges of dips in ccg
    peaks = find_peaks_cwt(-xgram_2d, peak_width, noise_perc=90) + 1
    peaks = np.abs(peaks - xgram.shape[0] / 2)
    peaks = peaks[peaks > 0.5 * peak_width]
    min_peaks = np.sort(peaks)

    # start with peaks closest to 0 and move to the next set of peaks if the event count is too low
    window_width = min_peaks * 1.5
    starts = np.maximum(xgram.shape[0] / 2 - window_width, 0)
    ends = np.minimum(xgram.shape[0] / 2 + window_width, xgram.shape[0] - 1)
    ind = 0
    xgram_window = xgram[int(starts[0]) : int(ends[0] + 1)]
    shfl_xgram_window = shfl_xgram[int(starts[0]) : int(ends[0] + 1)]
    xgram_sum = xgram_window.sum()
    window_size = xgram_window.shape[0] * xcorr_bin_width
    while (xgram_sum < (min_xcorr_rate * window_size * 10)) and (ind < starts.shape[0]):
        xgram_window = xgram[int(starts[ind]) : int(ends[ind] + 1)]
        shfl_xgram_window = shfl_xgram[int(starts[ind]) : int(ends[ind] + 1)]
        xgram_sum = xgram_window.sum()
        window_size = xgram_window.shape[0] * xcorr_bin_width
        ind += 1

    # use the whole ccg if peak finding fails
    if ind == starts.shape[0]:
        xgram_window = xgram
        shfl_xgram_window = shfl_xgram

    sig = (
        wasserstein_distance(
            np.arange(xgram_window.shape[0]) / xgram_window.shape[0],
            np.arange(xgram_window.shape[0]) / xgram_window.shape[0],
            xgram_window,
            shfl_xgram_window,
        )
        * 4
    )

    # apply a penalty if we had to use the whole ccg and event count is too low
    if (ind == starts.shape[0]) and xgram_window.sum() < (min_xcorr_rate * window_size):
        sig *= (xgram_window.sum() / (min_xcorr_rate * window_size)) ** 2

    return sig


def run_ae_sim(data, params, counts, cl_labels, times_multi, n_clust, channel_pos):
    good_ids = np.argwhere(
        (counts > params["min_spikes"]) & (cl_labels["label"].isin(params["good_lbls"]))
    ).flatten()
    mean_wf = slay.calc_mean_wf(
        params,
        n_clust,
        good_ids,
        times_multi,
        data,
    )
    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf, 2), 1)

    t0 = time.time()

    tqdm.write("Done, calculating cluster similarity...")
    sim = np.ndarray(0)

    # Autoencoder-based similarity calculation.
    ci = {
        "times_multi": times_multi,
        "counts": counts,
        "good_ids": good_ids,
        "mean_wf": mean_wf,
    }
    ext_params = {
        "pre_samples": params["ae_pre"],
        "post_samples": params["ae_post"],
        "num_chan": params["ae_chan"],
        "for_shft": params["ae_shft"],
    }
    spk_snips, cl_ids = slay.generate_train_data(
        data, ci, channel_pos, ext_params, params
    )
    # Train the autoencoder if needed.
    model_path = (
        params["model_path"]
        if params["model_path"]
        else os.path.join(params["KS_folder"], "automerge", "ae.pt")
    )
    if not os.path.exists(model_path):
        tqdm.write("Training autoencoder...")
        net, spk_data = slay.train_ae(
            spk_snips,
            cl_ids,
            counts,
            num_epochs=params["ae_epochs"],
            max_snips=params["max_spikes"],
        )
        torch.save(
            net.state_dict(),
            model_path,
        )
        tqdm.write(f"Autoencoder saved in {model_path}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = slay.CN_AE().to(device)
        net.load_state_dict(torch.load(model_path, weights_only=True))
        net.eval()
        spk_data = slay.SpikeDataset(spk_snips, cl_ids)

    # Calculate similarity using distances in the autoencoder latent space.
    sim, _, _, _ = slay.calc_ae_sim(mean_wf, net, peak_chans, spk_data, good_ids)
    pass_ms = sim > params["sim_thresh"]

    pass_ms = sim > params["sim_thresh"]
    tqdm.write(f"Found {pass_ms.sum() / 2} candidate cluster pairs")
    t1 = time.time()
    mean_sim_time = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))

    return pass_ms


def time_unif_ccg_sig(times_multi, n_clust, pass_ms, params):
    start = time.time()

    tqdm.write("Calculating cross-correlation metric...")
    xcorr_sig, _ = slay.calc_xcorr_metric(times_multi, n_clust, pass_ms, params)

    return xcorr_sig, time.time() - start


def time_shfl_ccg_sig(times_multi, n_clust, pass_ms, params):
    start = time.time()
    xcorr_sig, _ = calc_xcorr_metric(times_multi, n_clust, pass_ms, params)

    return xcorr_sig, time.time() - start


if __name__ == "__main__":
    main()
