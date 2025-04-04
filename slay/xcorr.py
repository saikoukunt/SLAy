"""
Functions to efficiently calculate auto- and cross- correlograms.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks_cwt, sosfiltfilt
from scipy.stats import wasserstein_distance


def bin_spike_trains(
    c1_times: NDArray[np.int_], c2_times: NDArray[np.int_], bin_width: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Splits two input spike trains into bins.

    Args:
        c1_times (NDArray): Spike trains in seconds.
        c2_times (NDArray): Spike trains in seconds.
        bin_width (float): The width of bins in seconds.

    Returns:
        c1_counts (NDArray): Binned spike counts
        c2_counts (NDArray): Binned spike counts.
    """
    c1_counts: NDArray[np.float64] = np.zeros(
        (math.ceil(max(c1_times) / bin_width)), dtype="int32"
    )
    c2_counts: NDArray[np.float64] = np.zeros(
        (math.ceil(max(c2_times) / bin_width)), dtype="int32"
    )

    for time in c1_times:
        c1_counts[math.floor(time / bin_width)] += 1
    for time in c2_times:
        c2_counts[math.floor(time / bin_width)] += 1

    return c1_counts, c2_counts


def xcorr_sig(
    xgram: NDArray[np.float64],
    xcorr_bin_width: float,
    min_xcorr_rate: float,
) -> float:
    """
    Calculates a cross-correlation significance metric for a cluster pair.

    Uses the wasserstein distance between an observed cross-correlogram and a null
    distribution as an estimate of how significant the dependence between
    two neurons is. Low spike count cross-correlograms have large wasserstein
    distances from null by chance, so we first try to expand the window size. If
    that fails to yield enough spikes, we apply a penalty to the metric.

    Args:
        xgram (NDArray): The raw cross-correlogram for the cluster pair.
        xcorr_bin_width (float): The width in seconds of the bin size of the
            input ccgs.
        max_window (float): The largest allowed window size during window
            expansion.
        min_xcorr_rate (float): The minimum ccg firing rate in Hz.

    Returns:
        sig (float): The calculated cross-correlation significance metric.
    """
    # calculate low-pass filtered second derivative of ccg
    fs = 1 / xcorr_bin_width
    cutoff_freq = 100
    nyqist = fs / 2
    cutoff = cutoff_freq / nyqist
    peak_width = 0.002 / xcorr_bin_width

    xgram_2d = np.diff(xgram, 2)
    sos = butter(4, cutoff, output="sos")
    xgram_2d = sosfiltfilt(sos, xgram_2d)

    if xgram.sum() == 0:
        return 0

    # find negative peaks of second derivative of ccg, these are the edges of dips in ccg
    peaks = find_peaks_cwt(-xgram_2d, peak_width, noise_perc=90) + 1
    peaks = np.abs(peaks - xgram.shape[0] / 2)
    peaks = peaks[peaks > 0.5 * peak_width]
    min_peaks = np.sort(peaks)

    # start with peaks closest to 0 and move to the next set of peaks if the event count is too low
    window_width = min_peaks * 1.5
    starts = np.maximum(xgram.shape[0] / 2 - window_width, 0)
    ends = np.minimum(xgram.shape[0] / 2 + window_width, xgram.shape[0] - 1)
    ind = 0
    xgram_window = xgram[int(starts[0]) : int(ends[0] + 1)]
    xgram_sum = xgram_window.sum()
    window_size = xgram_window.shape[0] * xcorr_bin_width
    while (xgram_sum < (min_xcorr_rate * window_size * 10)) and (ind < starts.shape[0]):
        xgram_window = xgram[int(starts[ind]) : int(ends[ind] + 1)]
        xgram_sum = xgram_window.sum()
        window_size = xgram_window.shape[0] * xcorr_bin_width
        ind += 1
    # use the whole ccg if peak finding fails
    if ind == starts.shape[0]:
        xgram_window = xgram

    sig = (
        wasserstein_distance(
            np.arange(xgram_window.shape[0]) / xgram_window.shape[0],
            np.arange(xgram_window.shape[0]) / xgram_window.shape[0],
            xgram_window,
            np.ones_like(xgram_window),
        )
        * 4
    )

    # apply a penalty if we had to use the whole ccg and event count is too low
    if (ind == starts.shape[0]) and xgram_window.sum() < (min_xcorr_rate * window_size):
        sig *= (xgram_window.sum() / (min_xcorr_rate * window_size)) ** 2

    return sig
