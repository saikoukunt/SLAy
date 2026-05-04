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
    Finds the num_chan channels closest to a given peak channel, ordered by distance.

    Args:
        peak_channel (int): Index of the peak channel to calculate distances from.
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
