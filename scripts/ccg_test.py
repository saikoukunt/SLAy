import slay
import numpy as np
from slay.schemas import RunParams
import matplotlib.pyplot as plt

from scipy.signal import sosfiltfilt, butter, find_peaks_cwt
from scipy.stats import wasserstein_distance
import npx_utils as npx

if __name__ == "__main__":
    ks_dir = r"D:\SLAy_data\mouse_hippocampus\catgt_ANM480363_20210527_pfc320_hpc180_g0\ANM480363_20210527_pfc320_hpc180_g0_imec1\imec1_ks2"
    args = {"KS_folder": ks_dir}
    args = slay.parse_kilosort_params(args)
    schema = RunParams()
    params = schema.load(args)
    clusters, cl_labels, _, _, n_clust, times_multi, counts = slay.load_ks_files(params)
    for i in range(100):
        if (counts[i] > 0) and (cl_labels.loc[i, "label"] in params["good_lbls"]):
            print(i)
            cl_times = times_multi[i] / params["sample_rate"]

            acg = npx.auto_correlogram(
                cl_times,
                0.1,
                params["xcorr_bin_width"],
                params["overlap_tol"],
            )

            fs = 1 / params["xcorr_bin_width"]
            cutoff_freq = 100
            nyqist = fs / 2
            cutoff = cutoff_freq / nyqist
            peak_width = 0.002 / params["xcorr_bin_width"]

            acg_2d = np.diff(acg, 2)
            sos = butter(4, cutoff, output="sos")
            acg_2d = sosfiltfilt(sos, acg_2d)

            peaks = find_peaks_cwt(-acg_2d, peak_width, noise_perc=90) + 1
            peaks = np.abs(peaks - acg.shape[0] / 2)
            peaks = peaks[peaks > 0.5 * peak_width]
            min_peaks = np.sort(peaks)

            window_width = min_peaks * 1.5
            starts = np.maximum(acg.shape[0] / 2 - window_width, 0)
            ends = np.minimum(acg.shape[0] / 2 + window_width, acg.shape[0] - 1)

            ind = 0
            acg_window = acg[int(starts[0]) : int(ends[0] + 1)]
            acg_sum = acg_window.sum()
            window_size = acg_window.shape[0] * params["xcorr_bin_width"] / 2
            while (acg_sum < (params["min_xcorr_rate"] * window_size * 10)) and (
                ind < starts.shape[0]
            ):
                acg_window = acg[int(starts[ind]) : int(ends[ind] + 1)]
                acg_sum = acg_window.sum()
                window_size = acg_window.shape[0] * params["xcorr_bin_width"] / 2
                ind += 1

            edges = None
            sig = 0
            if ind != starts.shape[0]:
                edges = [starts[ind], ends[ind] + 1]
                sig = (
                    wasserstein_distance(
                        np.arange(acg_window.shape[0]) / acg_window.shape[0],
                        np.arange(acg_window.shape[0]) / acg_window.shape[0],
                        acg_window,
                        np.ones_like(acg_window),
                    )
                    * 4
                )

            plt.figure()
            plt.subplot(1, 2, 1)
            if edges is not None:
                plt.vlines(edges, 0, acg.max(), color="r", linewidth=1)
            plt.bar(np.arange(acg.shape[0]), acg)
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(acg.shape[0] - 2), acg_2d)
            if edges is not None:
                plt.vlines(edges, acg_2d.min(), acg_2d.max(), color="r", linewidth=1)
            plt.suptitle(sig)

            plt.savefig(f"../results/ccg_test/cl_{i}.png", dpi=300)
            plt.close()
