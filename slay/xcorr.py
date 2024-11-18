"""
Functions to efficiently calculate auto- and cross- correlograms.
"""

import math

import numpy as np
import scipy
from numpy.typing import NDArray


def bin_spike_trains(
    c1_times: NDArray[np.int_], c2_times: NDArray[np.int_], bin_width: float
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
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
    c1_counts: NDArray[np.float_] = np.zeros(
        (math.ceil(max(c1_times) / bin_width)), dtype="int32"
    )
    c2_counts: NDArray[np.float_] = np.zeros(
        (math.ceil(max(c2_times) / bin_width)), dtype="int32"
    )

    for time in c1_times:
        c1_counts[math.floor(time / bin_width)] += 1
    for time in c2_times:
        c2_counts[math.floor(time / bin_width)] += 1

    return c1_counts, c2_counts


def x_correlogram(
    c1_times: NDArray[np.float_],
    c2_times: NDArray[np.float_],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> tuple[NDArray[np.float_], int]:
    """
    Calculates the cross correlogram between two spike trains.

    Args:
        c1_times (NDArray): Spike times in seconds.
        c2_times (NDArray): Spike times in seconds.
        window_size (float, optional): Width of cross correlogram window in seconds.
            Defaults to 100 ms.
        bin_width (float, optional): Width of cross correlogram bins in seconds.
            Defaults to 1 ms.
        overlap_tol (float, optional): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        corrgram (NDArray): The calculated cross-correlogram.
        overlap (int): The number of overlapping spikes.
    """
    # Call the cluster with more spikes c1.
    corrgram = np.zeros((math.ceil(window_size / bin_width)))

    overlap = 0
    c2_start = 0
    if c1_times.shape[0] < c2_times.shape[0]:
        c1_times, c2_times = c2_times, c1_times

    # To calculate the cross-correlogram, we iterate over c1 spikes as reference spikes
    # and count the number of c2 spikes that fall within window_size of the
    # reference spike.
    for ref_spk in range(c1_times.shape[0]):
        while (c2_start < c2_times.shape[0]) and (
            c2_times[c2_start] < (c1_times[ref_spk] - window_size / 2)
        ):
            c2_start += 1  # c2_start tracks the first in-window spike.

        spk_idx = c2_start  # spk_idx iterates over in-window c2 spikes.
        if spk_idx >= c2_times.shape[0]:
            continue

        while (spk_idx < c2_times.shape[0]) and (
            c2_times[spk_idx] < (c1_times[ref_spk] + window_size / 2)
        ):
            if abs(c1_times[ref_spk] - c2_times[spk_idx]) > overlap_tol:
                bin_idx = min(
                    math.floor(
                        (c1_times[ref_spk] - c2_times[spk_idx] + window_size / 2)
                        / bin_width
                    ),
                    corrgram.shape[0] - 1,
                )
                corrgram[bin_idx] += 1
            else:
                overlap += 1
            spk_idx += 1

    return corrgram, overlap


def auto_correlogram(
    c1_times: NDArray[np.float_],
    window_size: float,
    bin_width: float,
    overlap_tol: float,
) -> NDArray[np.float_]:
    """
    Calculates the auto correlogram for a spike train.

    Args:
        c1_times (NDArray): Spike times (sorted least to greatest)
            in seconds.
        window_size (float): Width of cross correlogram window in seconds.
        bin_width (float): Width of cross correlogram bins in seconds.
        overlap_tol (float): Overlap tolerance in seconds. Spikes within
            the tolerance of the reference spike time will not be counted for cross
            correlogram calculation.

    Returns:
        corrgram (NDArray): The calculated cross-correlogram.
    """
    corrgram = np.zeros((math.ceil(window_size / bin_width)))
    start = 0

    # To calculate the auto-correlogram, we iterate over spikes as reference spikes
    # and count the number of other spikes that fall within window_size of the
    # reference spike.
    for ref_spk in range(c1_times.shape[0]):
        while (start < c1_times.shape[0]) and (
            c1_times[start] < (c1_times[ref_spk] - window_size / 2)
        ):
            start += 1  # start tracks the first in-window spike.

        spk_idx = start  # spk_idx iterates over in-window spikes.
        if spk_idx >= c1_times.shape[0]:
            continue

        while (spk_idx < c1_times.shape[0]) and (
            c1_times[spk_idx] < (c1_times[ref_spk] + window_size / 2)
        ):
            if (ref_spk != spk_idx) and (
                abs(c1_times[ref_spk] - c1_times[spk_idx]) > overlap_tol
            ):
                gram_ind = min(
                    math.floor(
                        (c1_times[ref_spk] - c1_times[spk_idx] + window_size / 2)
                        / bin_width
                    ),
                    corrgram.shape[0] - 1,
                )
                corrgram[gram_ind] += 1
            spk_idx += 1

    return corrgram


def xcorr_sig(
    xgram: NDArray[np.float_],
    null_xgram: NDArray[np.float_],
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
    xgram_win: NDArray[np.float_] = xgram[start_idx : end_idx + 1]
    null_win: NDArray[np.float_] = null_xgram[start_idx : end_idx + 1]

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
