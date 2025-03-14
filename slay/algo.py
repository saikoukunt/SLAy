import json
import os
import time
from typing import Any

import npx_utils as npx
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm

import slay


def run_merge(params: dict[str, Any]) -> tuple[str, str, str, str, str, int, int]:
    if not torch.cuda.is_available():
        tqdm.write("CUDA not available, running on CPU.")

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
    times_multi = npx.find_times_multi(
        times,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )
    counts = np.zeros(n_clust, dtype=int)
    for k, v in times_multi.items():
        counts[k] = len(v)

    # update cl_labels to be list with all cluster_ids
    cl_labels = cl_labels.reindex(np.arange(n_clust))
    good_ids = np.argwhere(
        (counts > params["min_spikes"]) & (cl_labels["label"].isin(params["good_lbls"]))
    ).flatten()

    mean_wf = npx.calc_mean_wf(
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

    # Calculate a significance metric for cross-correlograms.
    tqdm.write("Calculating cross-correlation metric...")
    xcorr_sig, _ = slay.calc_xcorr_metric(times_multi, n_clust, pass_ms, params)

    t4 = time.time()
    xcorr_time = time.strftime("%H:%M:%S", time.gmtime(t4 - t1))
    # Calculate a refractor period penalty.
    tqdm.write("Calculating refractory period penalty...")
    ref_pen = slay.calc_ref_p(times_multi, n_clust, pass_ms, xcorr_sig, params)
    t5 = time.time()
    ref_pen_time = time.strftime("%H:%M:%S", time.gmtime(t5 - t4))

    # Calculate the final metric.
    tqdm.write("Calculating final metric...")
    final_metric = np.zeros_like(sim)
    for c1 in range(n_clust):
        for c2 in range(c1, n_clust):
            met = (
                sim[c1, c2]
                + params["xcorr_coeff"] * xcorr_sig[c1, c2]
                - params["ref_pen_coeff"] * ref_pen[c1, c2]
            )

            final_metric[c1, c2] = max(met, 0)
            final_metric[c2, c1] = max(met, 0)

    # Create a dataframe with the metric values for each candidate pair
    candidate_idxs = np.argwhere(pass_ms)
    c1_ids, c2_ids = candidate_idxs[:, 0], candidate_idxs[:, 1]
    candidate_sim = sim[c1_ids, c2_ids]
    candidate_xcorr_sig = xcorr_sig[c1_ids, c2_ids]
    candidate_ref_pen = ref_pen[c1_ids, c2_ids]
    candidate_final_metric = final_metric[c1_ids, c2_ids]

    candidate_metrics = pd.DataFrame(
        {
            "Cluster 1": c1_ids,
            "Cluster 2": c2_ids,
            "Similarity": candidate_sim,
            "Cross-correlation Significance": candidate_xcorr_sig,
            "Refractory Period Penalty": candidate_ref_pen,
            "Final Metric": candidate_final_metric,
        }
    )
    candidate_metrics = candidate_metrics.sort_values(
        by="Final Metric", ascending=False
    )

    # Save the dataframe as metrics.tsv in the automerge folder
    candidate_metrics.to_csv(
        os.path.join(params["KS_folder"], "automerge", "metrics.tsv"),
        sep="\t",
        index=False,
    )

    # Calculate merges.
    tqdm.write("Calculating merge suggestions...")
    old2new, new2old = slay.merge_clusters(clusters, mean_wf, final_metric, params)

    t6 = time.time()
    merge_time: str = time.strftime("%H:%M:%S", time.gmtime(t6 - t5))
    tqdm.write("Writing to output...")
    with open(
        os.path.join(params["KS_folder"], "automerge", "old2new.json"), "w"
    ) as file:
        file.write(json.dumps(old2new, separators=(",\n", ":")))

    with open(
        os.path.join(params["KS_folder"], "automerge", "new2old.json"), "w"
    ) as file:
        file.write(json.dumps(new2old, separators=(",\n", ":")))

    t7 = time.time()
    total_time: str = time.strftime("%H:%M:%S", time.gmtime(t7 - t0))

    vals = [
        data,
        cl_labels,
        mean_wf,
        counts,
        times,
        clusters,
        times_multi,
    ]  # values needed if auto_accept_merges is True

    return (
        vals,
        mean_sim_time,
        xcorr_time,
        ref_pen_time,
        merge_time,
        total_time,
        len(list(new2old.keys())),
        int(clusters.max()),
    )
