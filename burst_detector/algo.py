import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm

import burst_detector as bd


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
    counts = bd.spikes_per_cluster(clusters)
    times_multi = bd.find_times_multi(
        times,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )

    # shouldnt occur but just in case remove duplicates
    # cl_labels = cl_labels.loc[~cl_labels.index.duplicated(keep="first")]

    # update cl_labels to be list with all cluster_ids
    cl_labels = cl_labels.reindex(np.arange(n_clust))
    good_ids = np.argwhere(
        (counts > params["min_spikes"]) & (cl_labels["label"].isin(params["good_lbls"]))
    ).flatten()

    mean_wf, std_wf, spikes = bd.calc_mean_and_std_wf(
        params,
        n_clust,
        good_ids,
        times_multi,
        data,
        return_std=params["plot_merges"],
        return_spikes=True,
    )

    peak_chans = np.argmax(np.max(mean_wf, 2) - np.min(mean_wf, 2), 1)

    t0 = time.time()

    tqdm.write("Done, calculating cluster similarity...")
    sim = np.ndarray(0)

    # Autoencoder-based similarity calculation.
    if params["sim_type"] == "ae":
        spk_fld: str = os.path.join(params["KS_folder"], "automerge", "spikes")
        ci = {
            "times_multi": times_multi,
            "counts": counts,
            "good_ids": good_ids,
            "labels": cl_labels,
            "mean_wf": mean_wf,
        }
        ext_params = {
            "spk_fld": spk_fld,
            "pre_samples": params["ae_pre"],
            "post_samples": params["ae_post"],
            "num_chan": params["ae_chan"],
            "for_shft": params["ae_shft"],
        }
        spk_snips, cl_ids = bd.generate_train_data(
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
            net, spk_data = bd.train_ae(
                spk_snips,
                cl_ids,
                do_shft=params["ae_shft"],
                num_epochs=params["ae_epochs"],
            )
            torch.save(
                net.state_dict(),
                model_path,
            )
            tqdm.write(f"Autoencoder saved in {model_path}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = bd.CN_AE().to(device)
            net.load_state_dict(torch.load(model_path))
            net.eval()
            spk_data = bd.SpikeDataset(spk_snips, cl_ids)

        # Calculate similarity using distances in the autoencoder latent space.
        sim, _, _, _ = bd.calc_ae_sim(
            mean_wf, net, peak_chans, spk_data, good_ids, do_shft=params["ae_shft"]
        )
        pass_ms = sim > params["sim_thresh"]
    elif params["sim_type"] == "mean":
        # Calculate similarity using inner products between waveforms.
        sim, _, _, mean_wf, pass_ms = bd.calc_mean_sim(
            clusters, counts, n_clust, cl_labels, mean_wf, params
        )
        sim[pass_ms == False] = 0
    pass_ms = sim > params["sim_thresh"]
    tqdm.write(f"Found {pass_ms.sum() / 2} candidate cluster pairs")
    t1 = time.time()
    mean_sim_time = time.strftime("%H:%M:%S", time.gmtime(t1 - t0))

    # Calculate a significance metric for cross-correlograms.
    tqdm.write("Calculating cross-correlation metric...")
    xcorr_sig, _, _ = bd.calc_xcorr_metric(times_multi, n_clust, pass_ms, params)

    t4 = time.time()
    xcorr_time = time.strftime("%H:%M:%S", time.gmtime(t4 - t1))
    # Calculate a refractor period penalty.
    tqdm.write("Calculating refractory period penalty...")
    ref_pen, _ = bd.calc_ref_p(times_multi, n_clust, pass_ms, xcorr_sig, params)
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

    # Calculate/perform merges.
    tqdm.write("Merging...")
    old2new, new2old = bd.merge_clusters(clusters, mean_wf, final_metric, params)

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

    merges = list(new2old.values())

    if params["plot_merges"]:
        os.makedirs(
            os.path.join(params["KS_folder"], "automerge", "merges"), exist_ok=True
        )
        bd.plot_merges(merges, times_multi, mean_wf, std_wf, spikes, params)

    t7 = time.time()
    total_time: str = time.strftime("%H:%M:%S", time.gmtime(t7 - t0))

    vals = [cl_labels, mean_wf, counts, clusters]

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
