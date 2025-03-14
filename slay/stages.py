import functools
import json
import multiprocessing as mp
import os
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

import slay


def calc_ae_sim(
    mean_wf: NDArray[np.float64],
    model: nn.Module,
    peak_chans: NDArray[np.int_],
    spk_data: slay.SpikeDataset,
    good_ids: NDArray[np.int_],
    zDim: int = 15,
    sf: int = 1,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]
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
        good_ids (NDArray): IDs for clusters that passed the quality metrics and min_spikes threshold.
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
            spks, lab, idx = data[0].to(device), data[1].to(device), data[2]

            rec, _ = model(spks)
            loss += loss_fn(spks, rec).item()

            out = model.encode(spks)
            spk_lat[idx] = out.cpu().detach().numpy()
            spk_lab[idx] = lab.cpu().detach().numpy()

    tqdm.write(f"\nAverage Loss: {loss/len(dl):.4f}")

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
    bad_ids = np.setdiff1d(np.arange(mean_wf.shape[0]), good_ids)
    np.fill_diagonal(ae_sim, 0)
    ae_sim[bad_ids, :] = 0
    ae_sim[:, bad_ids] = 0

    # penalize pairs with different peak channels
    amps = np.max(mean_wf, 2) - np.min(mean_wf, 2)
    for i in tqdm(good_ids, "Calculating similarity metric"):
        for j in good_ids:
            p1 = peak_chans[i]
            p2 = peak_chans[j]

            # penalize by sigmoided version of geometric mean of cross-decay
            decay_pen_raw = np.sqrt(
                amps[i, p2] / amps[i, p1] * amps[j, p1] / amps[j, p2]
            )
            decay_pen = 1 / (1 + np.exp(-10 * (decay_pen_raw - 0.5)))

            ae_sim[i, j] *= decay_pen
            ae_sim[j, i] = ae_sim[i, j]

    return ae_sim, spk_lat_peak, lat_mean, spk_lab


def calc_xcorr_metric(
    times_multi: list[NDArray[np.float64]],
    n_clust: int,
    pass_ms: NDArray[np.bool_],
    params: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]
        xgrams[c1, c2] = res[i]

    # compute metric
    xcorr_sig: NDArray[np.float64] = np.zeros_like(pass_ms, dtype="float64")

    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            if pass_ms[c1, c2]:
                xcorr_sig[c1, c2] = slay.xcorr_sig(
                    xgrams[c1, c2],
                    xcorr_bin_width=params["xcorr_bin_width"],
                    min_xcorr_rate=params["min_xcorr_rate"],
                )
    for c1 in range(n_clust):
        for c2 in range(c1 + 1, n_clust):
            xcorr_sig[c2, c1] = xcorr_sig[c1, c2]

    return xcorr_sig, xgrams


def calc_ref_p(
    times_multi: list[NDArray[np.float64]],
    n_clust: int,
    pass_ms: NDArray[np.bool_],
    xcorr_sig: NDArray[np.float64],
    params: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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

    for i in range(len(res)):
        c1 = args[i][0]
        c2 = args[i][1]

        ref_pen[c1, c2] = res[i]
        ref_pen[c2, c1] = res[i]

    return ref_pen


def merge_clusters(
    clusters: NDArray[np.int_],
    mean_wf: NDArray[np.float64],
    final_metric: NDArray[np.float64],
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
    times_multi: list[NDArray[np.float64]],
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
    import npx_utils as npx

    # extract spike times
    c1_times = times_multi[c1] / params["sample_rate"]
    c2_times = times_multi[c2] / params["sample_rate"]
    # compute xgrams
    return npx.x_correlogram(
        c1_times,
        c2_times,
        params["window_size"],
        params["xcorr_bin_width"],
        params["overlap_tol"],
    )


def ref_p_func(
    c1: int,
    c2: int,
    times_multi: list[NDArray[np.float64]],
    params: dict[str, Any],
) -> tuple[float, float]:
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
    import npx_utils as npx

    # Extract spike times.
    c1_times = times_multi[c1] / params["sample_rate"]
    c2_times = times_multi[c2] / params["sample_rate"]

    # Calculate cross-correlogram.
    ccg = npx.x_correlogram(
        c1_times,
        c2_times,
        window_size=2,
        bin_width=params["ref_pen_bin_width"] / 1000,
        overlap_tol=params["overlap_tol"],
    )

    return npx.metrics._sliding_RP_viol(
        ccg, params["ref_pen_bin_width"], params["max_viol"]
    )


def accept_all_merges(vals, params) -> None:
    tqdm.write("Auto Accepting Merges")
    # merge suggested clusters
    new2old = os.path.join(params["KS_folder"], "automerge", "new2old.json")
    with open(new2old, "r") as f:
        merges = json.load(f)
        merges = {int(k): v for k, v in sorted(merges.items())}

    data, cl_labels, mean_wf, n_spikes, spike_times, spike_clusters, times_multi = vals

    for new_id, old_ids in merges.items():
        mean_wf, cl_labels, spike_times, spike_clusters = accept_merge(
            cl_labels,
            mean_wf,
            n_spikes,
            spike_times,
            spike_clusters,
            times_multi,
            params,
            old_ids,
            new_id,
        )

    # save data
    np.save(
        os.path.join(params["KS_folder"], "mean_waveforms.npy"),
        mean_wf,
    )
    cl_labels.to_csv(os.path.join(params["KS_folder"], "cluster_group.tsv"), sep="\t")
    np.save(os.path.join(params["KS_folder"], "spike_clusters.npy"), spike_clusters)
    np.save(os.path.join(params["KS_folder"], "spike_times.npy"), spike_times)


def accept_merge(
    cl_labels,
    mean_wf,
    n_spikes,
    spike_times,
    spike_clusters,
    times_multi,
    params,
    old_ids: list[int],
    new_id,
) -> int:
    # TODO change old_row data to 0's?
    """
    Merge clusters together into a new cluster.

    Args:
        cluster_ids (list[int]): List of cluster_ids to merge.
        new_id (int): New cluster_id.

    Returns:
        int: New cluster_id.
    """
    # if new_id exists fix duplicate spikes as needed
    if new_id in cl_labels.index.values:
        print(f"New cluster_id {new_id} already exists, skipping merge")
        return mean_wf, cl_labels, spike_times, spike_clusters

    new_spike_times, new_spike_clusters = remove_duplicate_spikes(
        times_multi, spike_times, spike_clusters, old_ids
    )

    # calculate new mean_wf and std_wf as weighted average, will be slightly affected by duplicated spike_times, but so minimal
    weighted_mean_wf = np.sum(
        mean_wf[old_ids, :, :]
        * (n_spikes[old_ids][:, np.newaxis, np.newaxis] / np.sum(n_spikes[old_ids])),
        axis=0,
    )

    new_mean_wf = np.zeros(
        (
            new_id + 1,
            params["n_chan"],
            params["pre_samples"] + params["post_samples"],
        )
    )

    old_rows_i = min(new_id, mean_wf.shape[0])
    new_mean_wf[:old_rows_i, :, :] = mean_wf[:old_rows_i, :, :]  # If get partial save
    new_mean_wf[new_id, :, :] = weighted_mean_wf

    # add new row
    cl_labels.loc[new_id, "label"] = cl_labels.loc[old_ids, "label"].mode().values[0]
    cl_labels.loc[new_id, "label_reason"] = f"merged from cluster_ids {old_ids}"
    new_spike_clusters[np.isin(new_spike_clusters, old_ids)] = new_id

    # calculate new metrics
    cl_labels.loc[old_ids, "label"] = "merged"
    cl_labels.loc[old_ids, "label_reason"] = f"merged into cluster_id {new_id}"

    return new_mean_wf, cl_labels, new_spike_times, new_spike_clusters


def remove_duplicate_spikes(
    times_multi, spike_times, spike_clusters, old_ids, n_samples=5
):
    combined_spike_times = times_multi[old_ids[0]]
    for old_id in old_ids[1:]:
        # add spike times that are NOT within 5 samples of each other
        old_times = times_multi[old_id]
        combined_spike_times = np.concatenate([combined_spike_times, old_times])
        combined_spike_times = np.sort(combined_spike_times)

        duplicate_mask = np.diff(combined_spike_times) <= n_samples
        first_duplicates = combined_spike_times[:-1][duplicate_mask]
        second_duplicates = combined_spike_times[1:][duplicate_mask]
        all_duplicates = np.concatenate([first_duplicates, second_duplicates])
        duplicate_times = np.intersect1d(
            old_times, all_duplicates
        )  # get times from old_id that are duplicates

        # delete duplicate times from combined_spike_times
        duplicate_idxs = np.where(np.isin(combined_spike_times, duplicate_times))[0]
        combined_spike_times = np.delete(combined_spike_times, duplicate_idxs)

        # find index where spike_time is duplicate and spike_cluster is old_ids
        cluster_idxs = np.where(spike_clusters == old_id)[0]
        duplicate_idxs = np.where(np.isin(spike_times, duplicate_times))[0]
        duplicate_idxs = np.intersect1d(cluster_idxs, duplicate_idxs)

        assert len(duplicate_idxs) == len(duplicate_times)
        # for each duplicate spike, remove associated spike_time and spike_cluster
        spike_times = np.delete(spike_times, duplicate_idxs)
        spike_clusters = np.delete(spike_clusters, duplicate_idxs)

    return spike_times, spike_clusters
