import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

import slay
from slay.schemas import RunParams

matplotlib.use("Qt5Agg")


def label_spikes(q, lvl_thresh):
    labels = np.zeros(q.shape[0] + 1, dtype="bool")
    labels2 = np.zeros(q.shape[0] + 1, dtype="bool")

    # edges
    labels[0] = q[0] >= lvl_thresh
    labels[-1] = q[-1] >= lvl_thresh

    # middle
    for i in range(1, q.shape[0] - 1):
        if (q[i - 1] < lvl_thresh) and (q[i] < lvl_thresh):
            labels[i] = False
        else:
            labels[i] = True

    # ignore bursts with < 3 spikes
    burst = False
    for i in range(labels.shape[0]):
        # check if in a burst
        if not labels[i]:
            burst = False
        else:
            if labels.shape[0] - i <= 2:
                burst = burst
            else:
                burst = burst or (labels[i + 1] and labels[i + 2])

        labels2[i] = burst

    return labels2


def generate_burst_fig():
    data_dir = "D:/SLAY_data"

    new_burst_pcts = {}
    ch_ints = {}

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if ("_ks" in dir_name) and ("orig" not in dir_name):
                ks_dir = os.path.join(root, dir_name)
                print(ks_dir)
                args = {"KS_folder": ks_dir}
                args = slay.parse_kilosort_params(args)
                schema = RunParams()
                params = schema.load(args)
                clusters, cl_labels, _, data, n_clust, times_multi, counts = (
                    slay.load_ks_files(params)
                )
                print((cl_labels == "good").sum())

    #             f = open(os.path.join(ks_dir, "automerge", "old2new.json"))
    #             old2new = json.load(f)

    #             f = open(os.path.join(ks_dir, "automerge", "new2old.json"))
    #             new2old = json.load(f)

    #             new_burst_pcts[ks_dir], ch_ints[ks_dir] = calc_burst_metrics(
    #                 n_clust, times_multi, old2new, new2old
    #             )

    # sns.violinplot(data=list(new_burst_pcts.values()), cut=0)
    # plt.xlabel("Dataset")
    # plt.ylabel(r"% change in number of bursts")
    # plt.savefig(
    #     "../results/figures/figburst_new_burst_pcts.svg", transparent=True, dpi=300
    # )

    # plt.figure()
    # sns.violinplot(data=list(ch_ints.values()), cut=0)
    # plt.xlabel("Dataset")
    # plt.ylabel(r"% change in average burst intensity")
    # plt.savefig("../results/figures/figburst_ch_ints.svg", transparent=True, dpi=300)

    # json.dump(
    #     new_burst_pcts,
    #     open("../results/figures/new_burst_pcts.json", "w"),
    #     indent=4,
    # )

    # json.dump(
    #     ch_ints,
    #     open("../results/figures/ch_ints.json", "w"),
    #     indent=4,
    # )


def calc_burst_metrics(n_clust, times_multi, old2new, new2old):
    pre_q = np.zeros(len(old2new.keys()), dtype="object")
    pre_a = np.zeros(len(old2new.keys()), dtype="object")
    for i in range(len(old2new.keys())):
        if i % 2:
            print(i)
        key = int(list(old2new.keys())[i])
        _, pre_q[i], pre_a[i] = slay.find_bursts(
            times_multi[key], state_ratio=3, gamma=0.1
        )

    post_q = np.zeros(len(new2old.keys()), dtype="object")
    post_a = np.zeros(len(new2old.keys()), dtype="object")
    for i in range(len(new2old.keys())):
        if i % 2:
            print(i)
        spike_times = []
        for clust in new2old[list(new2old.keys())[i]]:
            spike_times.append(times_multi[clust])
        spike_times = np.concatenate(spike_times)

        _, post_q[i], post_a[i] = slay.find_bursts(
            spike_times, state_ratio=3, gamma=0.1
        )

    thresh = 50
    pre_bursts = np.zeros(n_clust)
    pre_int = np.zeros(n_clust)
    for i in range(len(old2new.keys())):
        a_thresh = np.where(pre_a[i] > thresh)[0]
        key = int(list(old2new.keys())[i])
        if a_thresh.shape[0] == 0:
            pre_bursts[key] = 0
        else:
            lvl_thresh = a_thresh.min()
            labels = label_spikes(pre_q[i], lvl_thresh).sum()
            pre_bursts[key] = labels.sum()
            pre_q[i][pre_q[i] >= pre_a[i].shape[0]] = pre_a[i].shape[0] - 1
            if labels > 0:
                pre_int[key] = pre_a[i][
                    pre_q[i][pre_q[i] >= lvl_thresh].astype("int")
                ].mean()
            else:
                pre_int[key] = 0

    new_bursts = np.zeros(post_q.shape[0])
    new_bursts_pct = np.zeros(post_q.shape[0])
    post_bursts = np.zeros(post_q.shape[0])
    ch_int = np.zeros(post_q.shape[0])
    post_int = np.zeros(post_q.shape[0])

    for i in range(post_q.shape[0]):
        old_bursts = 0
        old_int = 0
        for clust in new2old[list(new2old.keys())[i]]:
            old_bursts += pre_bursts[clust]
            old_int += pre_int[clust] * pre_bursts[clust]

        if old_bursts > 0:
            old_int /= old_bursts

        a_thresh = np.where(post_a[i] > thresh)[0]
        key = int(list(new2old.keys())[i])
        if a_thresh.shape[0] == 0:
            post_bursts[i] = 0
        else:
            lvl_thresh = a_thresh.min()
            labels = label_spikes(post_q[i], lvl_thresh).sum()
            post_bursts[i] = labels.sum()

            new_bursts[i] = max(post_bursts[i] - old_bursts, 0)
            if old_bursts != 0:
                new_bursts_pct[i] = new_bursts[i] / old_bursts

            post_q[i][post_q[i] >= post_a[i].shape[0]] = post_a[i].shape[0] - 1
            post_int[i] = post_a[i][
                post_q[i][post_q[i] >= lvl_thresh].astype("int")
            ].mean()

            if new_bursts[i] > 0:
                ch_int[i] = max(post_int[i] - old_int, 0) / old_int

    return new_bursts_pct, ch_int


if __name__ == "__main__":
    generate_burst_fig()
