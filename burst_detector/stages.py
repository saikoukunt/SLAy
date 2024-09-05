import functools
import logging
import multiprocessing as mp
from collections import deque
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

import burst_detector as bd

logger = logging.getLogger("burst-detector")


def calc_mean_sim(
    clusters: NDArray[np.int_],
    counts: NDArray[np.int_],
    n_clust: int,
    labels: pd.DataFrame,
    mean_wf: NDArray[np.float_],
    params: dict[str, Any],
) -> tuple[
    NDArray[np.float_],
    NDArray[np.int_],
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.bool_],
]:
    """
    Calculates inner product similarity using mean waveforms.

    Args:
        clusters (NDArray): Spike cluster assignments.
        counts (NDArray): Number of spikes per cluster.
        n_clust (int): The number of clusters, taken to be the largest cluster id + 1.
        labels (pd.Dataframe): Cluster quality labels.
        mean_wf (NDArray): Cluster mean waveforms with shape (# of clusters,
            # channels, # timepoints).
        params (dict): General SpECtr params.

    Returns:
        mean_sim (NDArray): The (maximum) pairwise similarity for each pair of
            (normalized) waveforms.
        offset (NDArray): If jitter is enabled, the shift that produces the
            max inner product for each pair of waveforms.
        wf_norms (NDArray): Calculated waveform norms.
        mean_wf (NDArray): Cluster mean waveforms with shape (# of clusters,
            # channels, # timepoints).
        pass_ms (NDArray): True if a cluster pair passes the mean similarity
            threshold, false otherwise.
    """
    cl_good = np.zeros(n_clust, dtype=bool)
    unique = np.unique(clusters)
    for i in range(n_clust):
        if (
            (i in unique)
            and (counts[i] > params["sp_num_thresh"])
            and (labels.loc[labels["cluster_id"] == i, "group"].item() == "good")
        ):
            cl_good[i] = True

    # calculate mean similarity
    mean_sim, wf_norms, offset = bd.wf_means_similarity(
        mean_wf, cl_good, use_jitter=params["jitter"], max_jitter=params["jitter_amt"]
    )

    # check which cluster pairs pass threshold
    pass_ms = np.zeros_like(mean_sim, dtype="bool")
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if mean_sim[c1, c2] >= params["sim_thresh"] and (
                (counts[c1] >= params["sp_num_thresh"])
                and (counts[c2] >= params["sp_num_thresh"])
            ):
                if (
                    labels.loc[labels["cluster_id"] == c1, "group"].item() == "good"
                ) and (
                    labels.loc[labels["cluster_id"] == c2, "group"].item() == "good"
                ):
                    pass_ms[c1, c2] = True
                    pass_ms[c2, c1] = True

    return mean_sim, offset, wf_norms, mean_wf, pass_ms


def calc_ae_sim(
    mean_wf: NDArray[np.float_],
    model: nn.Module,
    peak_chans: NDArray[np.int_],
    spk_data: bd.SpikeDataset,
    cl_good: NDArray[np.bool_],
    do_shft: bool,
    zDim: int = 15,
    sf: int = 1,
) -> tuple[
    NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]
]:
    """
    Calculates autoencoder-based cluster similarity.

    Args:
        mean_wf (NDArray): Cluster mean waveforms with shape (# of clusters, #
            channels, # timepoints).
        model (nn.Module): Pre-trained autoencoder in eval mode and moved to GPU if
            available.
        peak_chans (NDArray): Peak channel for each cluster.
        spk_data (SpikeDataset): Dataset containing snippets used for
            cluster comparison.
        cl_good (NDArray): Cluster quality labels.
        do_shft (bool): True if model and spk_data are for an autoencoder explicitly
            trained on time-shifted snippets.
        zDim (int, optional): Latent dimensionality of CN_AE. Defaults to 15.
        sf (float, optional): Scaling factor for peak channels appended to latent vector. This
            does not affect the similarity calculation, only the returned `spk_lat_peak`
            array. Defaults to 1.

    Returns:
        ae_sim (NDArray): Pairwise autoencoder-based similarity. ae_sim[i,j] = 1
            indicates maximal similarity.
        spk_lat_peak (NDArray): hstack(latent vector, cluster peak channel) for
            each spike snippet in spk_data.
        lat_mean (NDArray): Centroid hstack(latent vector, peak channel) for each
            cluster in spk_data.
        spk_lab (NDArray): Cluster ids for each snippet in spk_data.
    """

    # init dataloader, latent arrays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(spk_data, batch_size=128)
    spk_lat = np.zeros((len(spk_data), zDim))
    spk_lab = np.zeros(len(spk_data), dtype="int32")
    loss_fn = nn.MSELoss()

    # calculate latent representations of spikes
    loss = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dl, desc="Calculating latent representations")):
            spks, lab = data[0].to(device), data[1].to(device)
            if do_shft:
                spks = spks[:, :, :, 5:-5]
            targ = spks.clone()

            rec = model(spks)
            loss += loss_fn(targ, rec).item()

            out = model.encoder(
                spks if not do_shft else rec
            )  # does this need to be (net.encoder(net(spks)) for time-shift?
            start_idx = idx * 128
            end_idx = start_idx + 128
            spk_lat[start_idx:end_idx] = out.cpu().detach().numpy()
            spk_lab[start_idx:end_idx] = lab.cpu().detach().numpy()

    logger.info(f"Average Loss: {loss/len(dl):.4f}")

    # construct dataframes with peak channel
    ae_df = pd.DataFrame({"cluster_id": spk_lab})
    for i in range(ae_df.shape[0]):
        ae_df.loc[i, "peak"] = peak_chans[
            int(ae_df.loc[i, "cluster_id"])  # type: ignore
        ]
    spk_lat_peak = np.hstack((spk_lat, sf * np.expand_dims(np.array(ae_df["peak"]), 1)))

    lat_df = pd.DataFrame(spk_lat_peak)
    lat_df["cluster_id"] = ae_df["cluster_id"]
    lat_df = lat_df.query("cluster_id != -1")  # exclude noise

    # calculate cluster centroids
    lat_mean = np.zeros((mean_wf.shape[0], zDim + 1))
    for cluster_id, group_df in lat_df.groupby("cluster_id"):
        lat_mean[int(cluster_id), :] = group_df.iloc[:, group_df.columns != "cluster_id"].mean(axis=0)  # type: ignore

    # calculate nearest neighbors, pairwise distances for cluster centroids
    neigh = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(lat_mean[:, :zDim])
    dists, _ = neigh.kneighbors(lat_mean[:, :zDim], return_distance=True)
    ae_dist = dist.squareform(dist.pdist(lat_mean[:, :zDim], "euclidean"))

    # similarity threshold for further analysis -- mean + std of distance to 1st NN
    ref_dist = dists[dists[:, 1] != 0, 1].mean() + dists[dists[:, 1] != 0, 1].std()

    # calculate similarity -- ref_dist is scaled to 0.6 similarity
    ae_sim = np.exp(-0.5 * ae_dist / ref_dist)

    # ignore self-similarity, low-spike and noise clusters
    np.fill_diagonal(ae_sim, 0)
    ae_sim[cl_good == False, :] = 0
    ae_sim[:, cl_good == False] = 0

    # penalize pairs with different peak channels
    amps = np.max(mean_wf, 2) - np.min(mean_wf, 2)
    for i in tqdm(range(ae_dist.shape[0]), "Calculating similarity metric"):
        for j in range(i, ae_dist.shape[0]):
            if (cl_good[i]) and (cl_good[j]):
                p1 = peak_chans[i]
                p2 = peak_chans[j]

                # penalize by geometric mean of cross-decay
                ae_sim[i, j] *= np.sqrt(
                    amps[i, p2] / amps[i, p1] * amps[j, p1] / amps[j, p2]
                )
                ae_sim[j, i] = ae_sim[i, j]

    return ae_sim, spk_lat_peak, lat_mean, spk_lab


def calc_xcorr_metric(
    times_multi: list[NDArray[np.float_]],
    n_clust: int,
    pass_ms: NDArray[np.bool_],
    params: dict[str, Any],
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
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
    xcorr_job: Callable = functools.partial(
        xcorr_func, times_multi=times_multi, params=params
    )

    # run cross correlogram jobs
    pool = mp.Pool(mp.cpu_count())  # type: ignore
    args = []
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if pass_ms[c1, c2]:
                args.append((c1, c2))
    res = pool.starmap(xcorr_job, args)

    # convert cross correlogram output to np arrays
    xgrams = np.empty_like(pass_ms, dtype="object")
    x_olaps = np.zeros_like(pass_ms, dtype="int16")
    null_xgrams = np.empty_like(pass_ms, dtype="object")

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]
        xgrams[c1, c2] = res[i][0]
        x_olaps[c1, c2] = res[i][1]

    # compute metric
    xcorr_sig: NDArray[np.float_] = np.zeros_like(pass_ms, dtype="float64")

    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if pass_ms[c1, c2]:
                xcorr_sig[c1, c2] = bd.xcorr_sig(
                    xgrams[c1, c2],
                    null_xgram=np.ones_like(xgrams[c1, c2]),
                    window_size=params["window_size"],
                    xcorr_bin_width=params["xcorr_bin_width"],
                    max_window=params["max_window"],
                    min_xcorr_rate=params["min_xcorr_rate"],
                )
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            xcorr_sig[c2, c1] = xcorr_sig[c1, c2]

    return xcorr_sig, xgrams, null_xgrams


def calc_ref_p(
    times_multi: list[NDArray[np.float_]],
    n_clust: int,
    pass_ms: NDArray[np.bool_],
    xcorr_sig: NDArray[np.float_],
    params: dict[str, Any],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Calculates the cross-correlogram significance metric between each candidate pair of
    clusters.

    Args:
        times_multi (list): Spike times in samples indexed by cluster id..
        n_clust (int): The number of clusters, taken to be the largest cluster id + 1.
        pass_ms (NDArray): True if a cluster pair passes the mean similarity
            threshold, false otherwise.
        xcorr_sig (NDArray): The calculated cross-correlation significance metric
            for each cluster pair.
        params (dict): General SpECtr params.

    Returns:
        ref_pen (NDArray): The calculated refractory period penalty for each pair of
            clusters.
        ref_per (NDArray): The inferred refractory period that was used to calculate
            the refractory period penalty.
    """
    # define refractory penalty job
    ref_p_job: Callable = functools.partial(
        ref_p_func, times_multi=times_multi, params=params
    )

    # run refractory penalty job
    pool = mp.Pool(mp.cpu_count())  # type: ignore
    args = []
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if (pass_ms[c1, c2]) and xcorr_sig[c1, c2] > 0:
                args.append((c1, c2))

    res = pool.starmap(ref_p_job, args)

    # convert output to numpy arrays
    ref_pen = np.zeros_like(pass_ms, dtype="float64")
    ref_per = np.zeros_like(pass_ms, dtype="float64")

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]

        ref_pen[c1, c2] = res[i][0]
        ref_pen[c2, c1] = res[i][0]

        ref_per[c1, c2] = res[i][1]
        ref_per[c2, c1] = res[i][1]

    return ref_pen, ref_per


def merge_clusters(
    clusters: NDArray[np.int_],
    mean_wf: NDArray[np.float_],
    final_metric: NDArray[np.float_],
    params: dict[str, Any],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Computes (multi-way) merges between candidate cluster pairs.

    Args:
        clusters (NDArray): Spike cluster assignments.
        mean_wf (NDArray): Cluster mean waveforms with shape (# of clusters,
            # channels, # timepoints).
        final_metric (NDArray): Final metric values for each cluster pair.
        params (dict): General burst-detector params.

    Returns:
        old2new (dict): Map from pre-merge cluster ID to post-merge cluster ID.
            Cluster IDs that were unchanged do not appear.
        new2old (dict): Map from post-merge cluster ID to pre-merge cluster IDs.
            Intermediate/unused/unchanged cluster IDs do not appear.

    """
    # find channel with peak amplitude for each cluster
    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf, 2), 1)

    # make rank-order list of merging
    ro_list = np.array(
        np.unravel_index(np.argsort(final_metric.flatten()), shape=final_metric.shape)
    ).T[::-1][::2]

    # threshold list and remove pairs that are too far apart
    pairs = deque()
    ind = 0

    while final_metric[ro_list[ind, 0], ro_list[ind, 1]] > params["final_thresh"]:
        c1_chan = peak_chans[ro_list[ind, 0]]
        c2_chan = peak_chans[ro_list[ind, 1]]
        if np.abs(c1_chan - c2_chan) < params["max_dist"]:
            pairs.append((ro_list[ind, 0], ro_list[ind, 1]))
        ind += 1

    # init dicts and lists
    new2old = {}
    old2new = {}
    new_ind = int(clusters.max() + 1)

    # merge logic
    for pair in pairs:
        c1 = int(pair[0])
        c2 = int(pair[1])

        # both clusters are original
        if c1 not in old2new and c2 not in old2new:
            old2new[c1] = int(new_ind)
            old2new[c2] = int(new_ind)
            new2old[new_ind] = [c1, c2]
            new_ind += 1

        # one or more clusters has been merged already
        else:
            # get un-nested cluster list
            cl1 = new2old[old2new[c1]] if c1 in old2new else [c1]
            cl2 = new2old[old2new[c2]] if c2 in old2new else [c2]
            if cl1 == cl2:
                continue
            cl_list = cl1 + cl2

            # iterate through cluster pairs
            merge = True
            for i in range(len(cl_list)):
                for j in range(i + 1, len(cl_list)):
                    i1 = cl_list[i]
                    i2 = cl_list[j]

                    if final_metric[i1, i2] < params["final_thresh"]:
                        merge = False

            # do merge
            if merge:
                for i in range(len(cl_list)):
                    old2new[cl_list[i]] = new_ind
                new2old[new_ind] = cl_list
                new_ind += 1

    # remove intermediate clusters
    for key in list(set(new2old.keys())):
        if key not in list(set(old2new.values())):
            del new2old[key]

    return old2new, new2old


# Multithreading function definitions
def xcorr_func(
    c1: int,
    c2: int,
    times_multi: list[NDArray[np.float_]],
    params: dict[str, Any],
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
    import burst_detector as bd

    # extract spike times
    c1_times = times_multi[c1] / params["sample_rate"]
    c2_times = times_multi[c2] / params["sample_rate"]

    # compute xgrams
    return bd.x_correlogram(
        c1_times,
        c2_times,
        params["max_window"],
        params["xcorr_bin_width"],
        params["overlap_tol"],
    )


def ref_p_func(
    c1: int,
    c2: int,
    times_multi: list[NDArray[np.float_]],
    params: dict[str, Any],
):
    """
    Multithreading function definition for calculating the refractory period penalty for
    a candidate cluster pair.

    Args:
        c1 (int): The ID of the first cluster.
        c2 (int): The ID of the second cluster.
        times_multi (list): Spike times in samples indexed by cluster id.
        params (dict): General burst-detector parameters.

    Returns:
        ref_pen (float): The computed refractory period penalty.
        ref_per (float): The inferred refractory period.

    """

    import numpy as np
    from scipy.stats import poisson

    import burst_detector as bd

    # Extract spike times.
    c1_times = times_multi[c1] / params["sample_rate"]
    c2_times = times_multi[c2] / params["sample_rate"]

    # Calculate cross-correlogram.
    ccg = bd.x_correlogram(
        c1_times,
        c2_times,
        params["ref_pen_bin_width"] / 1000,
        2,
        params["overlap_tol"],
    )[0]

    # Average the halves of the cross-correlogram.
    half_len = int(ccg.shape[0] / 2)
    ccg[half_len:] = (ccg[half_len:] + ccg[:half_len][::-1]) / 2
    ccg = ccg[half_len:]

    # Only test a few refractory period sizes.
    bTestIdx = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])
    bTest = bTestIdx * params["ref_pen_bin_width"] / 1000
    bTestIdx = bTestIdx[bTest <= 0.01]  # 10 ms is the hardcoded max refractory period
    bTest = bTest[bTest <= 0.01]

    ccg_cumsum = np.cumsum(ccg)
    sum_res = ccg_cumsum[bTestIdx - 1]  # -1 bc first bin is 0-bin_size

    # Calculate max violations per refractory period size.
    num_bins_2s = ccg.shape[0]
    num_bins_1s = int(num_bins_2s / 2)
    bin_rate = np.mean(ccg[num_bins_1s:num_bins_2s])  # taking 1-2s as a baseline
    max_contam = (
        np.array(bTest)
        / params["ref_pen_bin_width"]
        * 1000
        * bin_rate
        * params["max_viol"]
    )
    # Compute confidence of less than thresh contamination at each refractory period.
    confs = np.zeros(sum_res.shape[0])
    for j, cnt in enumerate(sum_res):
        if max_contam[j] > 3:
            confs[j] = 1 - poisson.cdf(cnt, max_contam[j])
        else:
            confs[j] = max(1 - cnt / max_contam[j], 1 - poisson.cdf(cnt, max_contam[j]))

    return 1 - confs.max(), bTest[confs.argmax()]
