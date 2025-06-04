import json
import os
from heapq import merge

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from figure_2 import calc_presence_ratio, calc_spatial_mismatch, load_channel_positions

import slay
from slay.schemas import RunParams

matplotlib.use("Qt5Agg")


def generate_figure_4():
    data_dir = "D:/SLAY_data"
    presence_ratio_dists = {}
    spatial_mismatch_dists = {}
    presence_ratio_merge_dists = {}
    spatial_mismatch_merge_dists = {}

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if (dir_name == "phy_output") and ("orig" not in dir_name):
                ks_dir = os.path.join(root, dir_name)
                args = {"KS_folder": ks_dir}
                args = slay.parse_kilosort_params(args)
                schema = RunParams()
                params = schema.load(args)
                clusters, cl_labels, _, data, n_clust, times_multi, counts = (
                    slay.load_ks_files(params)
                )
                print((cl_labels == "good").sum())
                n_chan = data.shape[-1]

                presence_ratio_dists[ks_dir] = calc_dist_presence_ratio(
                    cl_labels, n_clust, times_multi, counts, params
                )
                spatial_mismatch_dists[ks_dir] = calc_dist_spatial_mismatch(
                    ks_dir, clusters, n_clust, counts, cl_labels, params, n_chan
                )

                with open(os.path.join(ks_dir, "automerge", "new2old.json")) as f:
                    merges = json.load(f)

                if len(merges) == 0:
                    continue

                merge_clusters = np.concatenate(list(merges.values()))

                presence_ratio_merge_dists[ks_dir] = presence_ratio_dists[ks_dir][
                    merge_clusters
                ]
                spatial_mismatch_merge_dists[ks_dir] = spatial_mismatch_dists[ks_dir][
                    merge_clusters
                ]

                presence_ratio_dists[ks_dir][merge_clusters] = np.nan
                spatial_mismatch_dists[ks_dir][merge_clusters] = np.nan

                presence_ratio_dists[ks_dir] = presence_ratio_dists[ks_dir][
                    ~np.isnan(presence_ratio_dists[ks_dir])
                ]
                spatial_mismatch_dists[ks_dir] = spatial_mismatch_dists[ks_dir][
                    ~np.isnan(spatial_mismatch_dists[ks_dir])
                ]

    sns.violinplot(data=list(presence_ratio_dists.values()), cut=0)
    plt.xlabel("Dataset")
    plt.ylabel("Presence Ratio")
    plt.savefig(
        "../results/figures/fig4_presence_ratio_dists.svg",
        transparent=True,
        dpi=300,
    )

    plt.figure()
    sns.violinplot(data=list(presence_ratio_merge_dists.values()), cut=0)
    plt.xlabel("Dataset")
    plt.ylabel("Presence Ratio")
    plt.savefig(
        "../results/figures/fig4_merge_presence_ratio_dists.svg",
        transparent=True,
        dpi=300,
    )

    plt.figure()
    sns.violinplot(data=list(spatial_mismatch_dists.values()), cut=0)
    plt.xlabel("Dataset")
    # plt.yscale("log")
    plt.ylim(1, 1000)
    plt.savefig(
        "../results/figures/fig4_spatial_mismatch_dists.svg",
        transparent=True,
        dpi=300,
    )

    plt.figure()
    sns.violinplot(data=list(spatial_mismatch_merge_dists.values()), cut=0)
    plt.xlabel("Dataset")
    # plt.yscale("log")
    plt.ylim(1, 1000)
    plt.savefig(
        "../results/figures/fig4_merge_spatial_mismatch_dists.svg",
        transparent=True,
        dpi=300,
    )


def calc_dist_presence_ratio(cl_labels, n_clust, times_multi, counts, params):
    presence_ratios = np.ones(n_clust) * np.nan
    for i in range(n_clust):
        if (counts[i] > 0) and (cl_labels.loc[i, "label"] in params["good_lbls"]):
            times = times_multi[i]
            presence_ratios[i] = calc_presence_ratio(
                times / params["sample_rate"], thresh=2
            )

    return presence_ratios


def calc_dist_spatial_mismatch(
    ks_dir, clusters, n_clust, counts, cl_labels, params, n_chan
):
    channel_pos, channel_map, inverted_map, null_inds = load_channel_positions(
        ks_dir, n_chan
    )
    templates = np.load(os.path.join(ks_dir, "templates.npy"))[:, :, inverted_map]
    templates[:, :, null_inds] = np.zeros(
        (templates.shape[0], templates.shape[1], len(null_inds))
    )
    spike_templates = np.load(os.path.join(ks_dir, "spike_templates.npy"))
    mean_wf = np.load(os.path.join(ks_dir, "mean_waveforms.npy"))[:, channel_map, :]
    mean_wf = mean_wf[:, inverted_map, :]
    mean_wf[:, null_inds, :] = np.zeros(
        (mean_wf.shape[0], len(null_inds), mean_wf.shape[2])
    )

    spatial_mismatches = np.ones(n_clust) * np.nan
    for i in range(n_clust):
        if (
            (counts[i] > 0)
            and (cl_labels.loc[i, "label"] in params["good_lbls"])
            and (mean_wf[i].max() > 0)
        ):
            template_ind = scipy.stats.mode(
                spike_templates[np.where(clusters == i)[0]].flatten()
            )[0]
            spatial_mismatches[i] = calc_spatial_mismatch(
                templates[template_ind].T, mean_wf[i], channel_pos
            )

    return spatial_mismatches


if __name__ == "__main__":
    generate_figure_4()
