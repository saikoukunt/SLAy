"""
Functions to efficiently calculate auto- and cross- correlograms.
"""

import math

import numpy as np
import scipy
from numpy.typing import NDArray


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
    null_xgram: NDArray[np.float64],
    window_size: float,
    xcorr_bin_width: float,
    max_window: float,
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
        null_dist (NDArray): The null cross-correlogram for the cluster pair.
            In practice, this is usually a uniform distribution.
        window_size (float): The width in seconds of the default ccg window.
        xcorr_bin_width (float): The width in seconds of the bin size of the
            input ccgs.
        max_window (float): The largest allowed window size during window
            expansion.
        min_xcorr_rate (float): The minimum ccg firing rate in Hz.

    Returns:
        sig (float): The calculated cross-correlation significance metric.
    """
    num_bins_half: int = math.ceil(round(window_size / xcorr_bin_width) / 2)
    start_idx = int(xgram.shape[0] / 2 - num_bins_half)
    end_idx = int(xgram.shape[0] / 2 - 1 + num_bins_half)
    xgram_win: NDArray[np.float64] = xgram[start_idx : end_idx + 1]
    null_win: NDArray[np.float64] = null_xgram[start_idx : end_idx + 1]

    # If the ccg doesn't contain enough spikes, we double the window size until
    # it does, or until window_size == max_window.
    while (xgram_win.sum() < min_xcorr_rate * window_size) and (
        window_size < max_window
    ):
        window_size = min(max_window, 2 * window_size)
        num_bins_half = math.ceil(round(window_size / xcorr_bin_width) / 2)
        start_idx = int(xgram.shape[0] / 2 - num_bins_half)
        end_idx = int(xgram.shape[0] / 2 - 1 + num_bins_half)
        xgram_win = xgram[start_idx : end_idx + 1]
        null_win = null_xgram[start_idx : end_idx + 1]
    if (xgram_win.sum() == 0) or (null_win.sum() == 0):
        return 0

    # To normalize the wasserstein distance, we divide by 0.25, which is the
    # wasserstein distance between uniform and delta distributions.
    num_bins_half *= 2
    sig: float = (
        scipy.stats.wasserstein_distance(
            np.arange(num_bins_half) / num_bins_half,
            np.arange(num_bins_half) / num_bins_half,
            xgram_win,
            null_win,
        )
        / 0.25
    )

    # Low spike count penalty is the squared ratio of the observed spikes to the minimum
    # number of spikes.
    if xgram_win.sum() < (min_xcorr_rate * window_size):
        sig *= (xgram_win.sum() / (min_xcorr_rate * window_size)) ** 2

    return sig
