# import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

import slay
from slay.schemas import RunParams

matplotlib.use("Qt5Agg")


def generate_figure_2():
    data_dir = "D:/SLAY_data"
    presence_ratio_dists = {}
    spatial_mismatch_dists = {}

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name[-4:-1] == "_ks":
                ks_dir = os.path.join(root, dir_name)
                args = {"KS_folder": ks_dir}
                args = slay.parse_kilosort_params(args)
                schema = RunParams()
                params = schema.load(args)
                clusters, cl_labels, _, _, n_clust, times_multi, counts = (
                    slay.load_ks_files(params)
                )

                presence_ratio_dists[ks_dir] = calc_dist_presence_ratio(
                    cl_labels, n_clust, times_multi, counts, params
                )
                spatial_mismatch_dists[ks_dir] = calc_dist_spatial_mismatch(
                    ks_dir, clusters, n_clust, counts, cl_labels, params
                )

    sns.violinplot(data=list(presence_ratio_dists.values()), cut=0)
    plt.xlabel("Dataset")
    plt.ylabel("Presence Ratio")
    plt.savefig(
        "../results/figures/fig2_presence_ratio_dists.svg", transparent=True, dpi=300
    )

    sns.violinplot(data=list(spatial_mismatch_dists.values()), cut=0)
    plt.xlabel("Dataset")
    plt.ylabel("Spatial Mismatch")
    plt.yscale("log")
    plt.ylim(1, 1000)
    plt.hlines(20, 0, 1, colors="r", linestyles="--")
    plt.savefig(
        "../results/figures/fig2_spatial_mismatch_dists.svg", transparent=True, dpi=300
    )


## HELPER FUNCTIONS
def calc_dist_presence_ratio(cl_labels, n_clust, times_multi, counts, params):
    presence_ratios = np.ones(n_clust) * np.nan
    for i in range(n_clust):
        if (counts[i] > 0) and (cl_labels.loc[i, "label"] in params["good_lbls"]):
            times = times_multi[i]
            presence_ratios[i] = calc_presence_ratio(times / params["sample_rate"])
    presence_ratios = presence_ratios[~np.isnan(presence_ratios)]

    return presence_ratios


def calc_presence_ratio(times, thresh=5):
    c1, _ = slay.bin_spike_trains(times, times, 20)

    return (c1 > thresh).sum() / c1.shape[0]


def calc_dist_spatial_mismatch(ks_dir, clusters, n_clust, counts, cl_labels, params):
    channel_pos, channel_shanks = load_channel_positions(ks_dir)
    templates = np.load(os.path.join(ks_dir, "templates.npy"))
    spike_templates = np.load(os.path.join(ks_dir, "spike_templates.npy"))
    mean_wf = np.load(os.path.join(ks_dir, "mean_waveforms.npy"))[
        :, : channel_pos.shape[0], :
    ]

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
                templates[template_ind].T, mean_wf[i], channel_pos, channel_shanks
            )

    spatial_mismatches = spatial_mismatches[~np.isnan(spatial_mismatches)]

    return spatial_mismatches


def calc_spatial_mismatch(template, mean_wf, channel_pos, channel_shanks):
    template_centroid = calc_centroid(template, channel_pos, channel_shanks)
    mean_wf_centroid = calc_centroid(mean_wf, channel_pos, channel_shanks)
    distance = scipy.spatial.distance.euclidean(template_centroid, mean_wf_centroid)

    return distance


def calc_centroid(template, channel_positions, channel_shanks):
    channel_ids, best_channel = find_best_channels(
        template, channel_positions, channel_shanks
    )
    positions = channel_positions[channel_ids]
    ptp_amps = template[channel_ids, :].max(axis=1) - template[channel_ids, :].min(
        axis=1
    )

    centroid = np.average(positions, axis=0, weights=ptp_amps)

    return centroid


def get_closest_channels(channel_positions, channel_index, n=None):
    """Get the channels closest to a given channel on the probe."""
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    return out


def find_best_channels(template, channel_pos, channel_shanks):
    amplitude_threshold = 0.2

    amplitude = template.max(axis=1) - template.min(axis=1)
    best_channel = np.argmax(amplitude)
    max_amp = amplitude[best_channel]

    peak_channels = np.argsort(amplitude)[::-1]
    peak_channels = peak_channels[
        amplitude[peak_channels] > amplitude_threshold * max_amp
    ]

    # close_channels = get_closest_channels(channel_pos, best_channel, 12)

    # shank = channel_shanks[best_channel]
    # channels_on_shank = np.nonzero(channel_shanks == shank)[0]
    # close_channels = np.intersect1d(close_channels, channels_on_shank)
    # channel_ids = np.intersect1d(close_channels, peak_channels)

    return peak_channels, best_channel


def load_channel_positions(full_path):
    channel_pos = np.load(os.path.join(full_path, "channel_positions.npy"))
    try:
        channel_shanks = np.load(os.path.join(full_path, "channel_shanks.npy"))
    except FileNotFoundError:
        channel_shanks = np.ones(channel_pos.shape[0], dtype=int)
    channel_map = np.load(os.path.join(full_path, "channel_map.npy")).flatten()
    inverted_map = np.ones(channel_map.max() + 1, dtype=int) * -1
    for i, ch in enumerate(channel_map):
        inverted_map[ch] = i
    inverted_map = inverted_map[~(inverted_map == -1)]
    channel_pos = channel_pos[inverted_map]
    channel_shanks = channel_shanks[inverted_map]

    return channel_pos, channel_shanks


if __name__ == "__main__":
    generate_figure_2()
