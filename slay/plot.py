import math
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm import tqdm
import json

import slay
import npx_utils as npx

matplotlib.use("Agg")


### PLOTTING CODE
def plot_merges(
    data: NDArray[np.float_],
    times_multi: list,
    mean_wf: NDArray[np.float_],
    params: dict[str, Any],
    nchan: int = 20,
    start: int = 10,
    stop: int = 60,
) -> None:
    """
    Plot the merges by generating waveform and correlation plots.
    Args:
        merges (list): List of merges to plot.
        times_multi (list): List of timestamps for each merge.
        mean_wf (NDArray): Mean waveform data.
        std_wf (NDArray): Standard deviation of waveform data.
        spikes (NDArray): Spike data.
        params (dict): Parameters for plotting.
        nchan (int, optional): Number of channels. Defaults to 20.
        start (int, optional): Start time for plotting. Defaults to 10.
        stop (int, optional): Stop time for plotting. Defaults to 60.
    """
    new2old = os.path.join(params["KS_folder"], "automerge", "new2old.json")
    with open(new2old, "r") as f:
        merges = json.load(f)

    for merge in tqdm(merges.values(), desc="Plotting merges"):
        merge.sort()
        spikes = {}
        for id in merge:
            spikes[id] = npx.extract_spikes(
                data,
                times_multi,
                id,
                params["pre_samples"],
                params["post_samples"],
                200,
            )

        wf_plot = plot_wfs(merge, mean_wf, spikes, nchan, start, stop)
        corr_plot = plot_corr(merge, times_multi, params)

        merge_str = "_".join(map(str, merge))
        name = os.path.join(
            params["KS_folder"],
            "automerge",
            "merges",
            f"merge_{merge_str}.pdf",
        )

        with PdfPages(name) as file:
            file.savefig(wf_plot, dpi=300)
            file.savefig(corr_plot, dpi=300)

        plt.close(wf_plot)
        plt.close(corr_plot)


def plot_wfs(
    clust: list[int],
    mean_wf: NDArray[np.float_],
    spikes: NDArray,
    nchan: int = 10,
    start: int = 10,
    stop: int = 60,
) -> Figure:
    """
    Plot waveforms for specified clusters.
    Args:
        clust (list): List of cluster indices.
        mean_wf (NDArray): Array of mean waveforms for each cluster.
        std_wf (NDArray): Array of standard deviation of waveforms for each cluster.
        spikes (NDArray): Array of spike waveforms for each cluster.
        nchan (int, optional): Number of channels to plot. Defaults to 10.
        start (int, optional): Start index of waveform plot. Defaults to 10.
        stop (int, optional): Stop index of waveform plot. Defaults to 60.
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    peak_chans = np.argmax(np.ptp(mean_wf, axis=2), axis=1)
    peak = int(np.mean(peak_chans[clust]))

    fig, axes = plt.subplots(
        math.ceil(nchan / 2),
        2 * len(clust),
        figsize=(5 * len(clust), 6),
        constrained_layout=True,
    )
    ch_start = max(int(peak - nchan / 2), 0)
    ch_stop = min(int(peak + nchan / 2), mean_wf.shape[1] - 1)
    lines = []

    for i in range(ch_start, ch_stop):
        ind = i - ch_start
        or_ind = ind * len(clust) + 1

        for cl, id in enumerate(clust):
            p_ind = or_ind + cl
            ax = plt.subplot(math.ceil(nchan / 2), 2 * len(clust), p_ind)
            ax.set_facecolor("black")
            for j in range(spikes[id].shape[0]):
                (line,) = ax.plot(
                    spikes[id][j, i, start:stop],
                    label=str(id),
                    color=COLORS[cl],
                    linewidth=0.25,
                )
            lines.append(line)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
            )

    for ax, row in zip(axes[:, 0], range(ch_start, ch_stop, 2)):
        ax.set_ylabel(row, rotation=90, size="large")
    for ax, row in zip(axes[:, len(clust)], range(ch_start + 1, ch_stop, 2)):
        ax.set_ylabel(row, rotation=90, size="large")

    fig.legend(lines, clust)

    return fig


def plot_corr(
    clust: list[int],
    times_multi: list[NDArray[np.float_]],
    params: dict[str, Any],
) -> Figure:
    """
    Plots the auto and cross correlograms for a given set of clusters.
    Args:
        clust (list): List of cluster indices.
        times_multi (list): List of spike times for each cluster.
        params (dict): Dictionary of parameters.
    Returns:
        fig (Figure): The generated figure.
    """
    n_clust = len(clust)
    fig, axes = plt.subplots(n_clust, n_clust, figsize=(10, 5))

    window_size = 0.1
    bin_size = params["ref_pen_bin_width"]
    overlap_tol = params["overlap_tol"]

    # auto correlograms
    for i in range(n_clust):
        acg = npx.auto_correlogram(
            times_multi[clust[i]] / 30000,
            window_size,
            bin_size / 1000,
            overlap_tol,
        )

        ax = plt.subplot(n_clust, n_clust, (i * n_clust) + (i + 1))
        ax.set_facecolor("black")
        ax.set_yticks([0, acg.max()])
        plt.bar(range(len(acg)), acg, width=1, color=COLORS[i])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )

    cross_pairs = [(i, j) for i in range(n_clust) for j in range(n_clust) if i != j]

    # cross correlograms
    for i, j in cross_pairs:
        ccg = npx.x_correlogram(
            times_multi[clust[i]] / 30000,
            times_multi[clust[j]] / 30000,
            window_size,
            bin_size / 1000,
            overlap_tol,
        )
        ax = plt.subplot(n_clust, n_clust, i * n_clust + (j + 1))
        ax.set_facecolor("black")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_yticks([0, ccg.max()])

        if i > j:
            ax.bar(range(len(ccg)), ccg[::-1], width=1, color=LIGHT_COLORS[i])
        else:
            ax.bar(range(len(ccg)), ccg, width=1, color=LIGHT_COLORS[i])

        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )

    if n_clust > 1:
        for ax, col in zip(axes[-1], clust):
            ax.set_xlabel(col, size="large")

        for ax, row in zip(axes[:, 0], clust):
            ax.set_ylabel(row, rotation=0, size="large")

    fig.tight_layout()

    return fig


## COLOR ARRAYS

LIGHT_COLORS = [
    (229 / 255, 239 / 255, 254 / 255),
    (254 / 255, 231 / 255, 230 / 255),
    (234 / 255, 249 / 255, 234 / 255),
    (254 / 255, 250 / 255, 229 / 255),
    (254 / 255, 241 / 255, 229 / 255),
    (246 / 255, 238 / 255, 245 / 255),
    (254 / 255, 229 / 255, 239 / 255),
    (249 / 255, 242 / 255, 235 / 255),
    (233 / 255, 251 / 255, 249 / 255),
    (243 / 255, 243 / 255, 240 / 255),
]

COLORS = [
    (0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
    (0.9832565730779054, 0.3694984452949815, 0.3488265255379734),
    (0.4666666666666667, 0.8666666666666667, 0.4666666666666667),
    (0.999, 0.8666666666666666, 0.23116059580240228),
    (0.999, 0.62745156, 0.3019607),
    (0.656421832660253, 0.35642078793464527, 0.639125729774389),
    (0.999, 0.6509803921568628, 0.788235294117647),
    (0.8352941176470589, 0.6392156862745098, 0.4470588235294118),
    (0.25098039215686274, 0.8784313725490196, 0.8156862745098039),
    (0.7098039215686275, 0.7411764705882353, 0.6470588235294118),
]
