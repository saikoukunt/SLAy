"""
General utilities for ephys data-wrangling.

Assumes that ephys data is stored in the phy output format.
"""

import numpy as np
from numpy.typing import NDArray
from spikeinterface.core import SortingAnalyzer


def get_closest_channels(
    channel_positions: NDArray[np.float64], ref_chan: int, num_close: int | None = None
) -> NDArray[np.int_]:
    """
    Gets the channels closest to a specified channel on the probe.

    Args:
        channel_positions (NDArray): The XY coordinates of each channel on the
            probe (in arbitrary units)
        ref_chan (int): The index of the channel to calculate distances relative to.
        num_close (int, optional): The number of closest channels to return,
            including `ref_chan`.

    Returns:
        close_chans (NDArray): The indices of the closest channels, sorted from
            closest to furthest. Includes the ref_chan.

    """
    x, y = channel_positions[:, 0], channel_positions[:, 1]
    x0, y0 = channel_positions[ref_chan]

    dists: NDArray[np.float64] = (x - x0) ** 2 + (y - y0) ** 2
    close_chans = np.argsort(dists)
    if num_close:
        close_chans = close_chans[:num_close]
    return close_chans


def get_channels_by_distance(
    peak_channel: int,
    sorting_analyzer: SortingAnalyzer,
    num_chan: int,
) -> NDArray[np.int_]:
    """
    For a given waveform template, finds the channels with largest amplitude and
    orders them by distance from the peak channel.

    Args:
        template (NDArray): The waveform template to find the best channels for.
        sorting_analyzer (SortingAnalyzer): SpikeInterface SortingAnalyzer containing
            the recording with channel position information.
        num_chan (int): The number of closest channels to return.

    Returns:
        ordered_chans (NDArray): Channel IDs ordered by distance from peak,
            with peak channel first.
    """
    channel_pos = sorting_analyzer.recording.get_channel_locations()
    close_chans: NDArray[np.int_] = get_closest_channels(
        channel_pos, peak_channel, num_chan
    )

    # Calculate distances and sort channels by distance
    x: NDArray[np.float64] = channel_pos[:, 0]
    y: NDArray[np.float64] = channel_pos[:, 1]
    x0, y0 = channel_pos[peak_channel]
    dists: NDArray[np.float64] = (x - x0) ** 2 + (y - y0) ** 2
    dists_subset = dists[close_chans]

    return sorting_analyzer.recording.channel_ids[close_chans[np.argsort(dists_subset)]]


# TODO: move these out of source code and refactor to use SI
# def calc_fr_unif(
#     spike_times: list[np.int_],
#     old2new: dict[int, int],
#     new2old: dict[int, list[int]],
#     times_multi: list[NDArray[np.int_]],
# ) -> tuple[NDArray, NDArray]:
#     """
#     Calculate firing rate uniformity metrics for spike trains.
#     Args:
#         spike_times (list): List of spike times for each cluster.
#         old2new (dict): Dictionary mapping old cluster IDs to new cluster IDs.
#         new2old (dict): Dictionary mapping new cluster IDs to old cluster IDs.
#         times_multi (dict): Dictionary mapping cluster IDs to spike times.
#     Returns:
#         single_ds (NDArray): Array of firing rate uniformity metrics for individual clusters.
#         merged_ds (NDArray): Array of firing rate uniformity metrics for merged clusters.
#     """
#     merged_ds = np.zeros(len(new2old.keys()))
#     for i in range(len(new2old.keys())):
#         spike_times = []
#         for clust in new2old[list(new2old.keys())[i]]:
#             spike_times.append(times_multi[clust])

#         spike_times = np.concatenate(spike_times)
#         c1, _ = bin_spike_trains(spike_times, spike_times, 20)
#         n = c1.shape[0]
#         merged_ds[i] = 1 - wasserstein_distance(
#             u_values=np.arange(n) / n,
#             v_values=np.arange(n) / n,
#             u_weights=c1,
#             v_weights=np.ones(n),
#         )

#     single_ds = np.zeros(len(old2new.keys()))
#     for i in range(len(old2new.keys())):
#         clust = int(list(old2new.keys())[i])
#         spike_times = times_multi[clust]
#         c1, _ = bin_spike_trains(spike_times, spike_times, 20)
#         n = c1.shape[0]
#         single_ds[i] = 1 - wasserstein_distance(
#             u_values=np.arange(n) / n,
#             v_values=np.arange(n) / n,
#             u_weights=c1,
#             v_weights=np.ones(n),
#         )

#     return single_ds, merged_ds


# def temp_mismatch(
#     clust_id: int,
#     templates: list,
#     channel_pos: NDArray[np.float64],
#     n_chan: int,
#     num_close: int,
#     mean_wf: NDArray[np.float64],
# ) -> float:
#     """
#     Calculate the temporal mismatch between the proximity ranks and amplitude ranks of a given cluster.
#     Args:
#         clust_id (int): The ID of the cluster.
#         templates (list): List of templates.
#         channel_pos (NDArray): Array of channel positions.
#         n_chan (int): Number of channels.
#         num_close (int): Number of closest channels to consider.
#         mean_wf (NDArray): Array of mean waveforms.
#     Returns:
#         mismatch (float): The magnitude and direction of the temporal mismatch.
#     """
#     ch_ids, peak_channel = find_best_channels(
#         templates[clust_id], channel_pos, n_chan, num_close
#     )

#     # calculate and rank distances (proximity)
#     dists = get_dists(channel_pos, peak_channel, ch_ids)
#     prox_order = np.argsort(dists)
#     prox_ranks = np.argsort(prox_order)

#     # calculate and rank amplitudes
#     means = mean_wf[clust_id, ch_ids, :]
#     amp = means.max(axis=1) - means.min(axis=1)
#     amp_order = np.argsort(amp)
#     amp_ranks = np.argsort(amp_order)

#     # calculate magnitude and direction of mismatch
#     mismatch = np.abs(
#         (prox_ranks[prox_order] - amp_ranks[prox_order])[
#             int(ch_ids.shape[0] / 2) :
#         ].sum()
#         / 26
#     )  # 26 is maximum possible raw mismatch
#     return mismatch
