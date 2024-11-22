import os

import matplotlib.pyplot as plt
import numpy as np
from marshmallow import EXCLUDE
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import slay
from slay.schemas import PlotUnitsParams


def main() -> None:
    args = slay.parse_cmd_line_args()
    schema = PlotUnitsParams(unknown=EXCLUDE)
    params = schema.load(args)

    automerge = os.path.join(params["KS_folder"], "automerge")
    os.makedirs(os.path.join(automerge, "units"), exist_ok=True)

    times = np.load(os.path.join(params["KS_folder"], "spike_times.npy")).flatten()
    clusters = np.load(
        os.path.join(params["KS_folder"], "spike_clusters.npy")
    ).flatten()
    n_clust = clusters.max() + 1

    # load recording
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data = np.reshape(rawData, (int(rawData.size / params["n_chan"]), params["n_chan"]))

    # count spikes per cluster, load quality labels
    times_multi = slay.find_times_multi(
        times,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )
    counts = np.array([len(times_multi[i]) for i in range(n_clust)])

    # filter out low-spike/noise units
    good_ids = np.where(counts > params["min_spikes"])[0]
    mean_wf, std_wf, spikes = slay.calc_mean_and_std_wf(
        params, n_clust, good_ids, times_multi, data, return_spikes=True
    )

    for id in tqdm(good_ids, desc="Plotting units"):
        wf_plot = slay.plot_wfs([id], mean_wf, std_wf, spikes)
        acg_plot = slay.plot_corr([id], times_multi, params)

        name = os.path.join(params["KS_folder"], "automerge", f"units{id}.pdf")
        with PdfPages(name) as file:
            file.savefig(wf_plot, dpi=300)
            file.savefig(acg_plot, dpi=300)

        plt.close(wf_plot)
        plt.close(acg_plot)


if __name__ == "__main__":
    main()
