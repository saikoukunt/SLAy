import numpy as np
import pandas as pd
from spikeinterface.core import SortingAnalyzer
from spikeinterface.core.sorting_tools import (
    generate_unit_ids_for_split,
    spike_vector_to_indices,
)
from spikeinterface.qualitymetrics import compute_quality_metrics

from slay.metrics import _sliding_RP_viol_pair


def make_artificial_splits(
    sorting_analyzer: SortingAnalyzer,
    splitting_probability,
    random_seed=0,
):
    """
    Create artificial splits using multiple splitting strategies.

    Applies splits in order of most to least restrictive candidate criteria:
    burst -> amplitude -> drift -> random

    Only units with >= 1000 spikes are eligible for splitting.
    Each unit can only be split once, and units created from splits
    cannot be split again. Split pairs where either resulting unit has
    SNR < 2, firing rate < 0.1 Hz, or high refractory period violations
    are discarded.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing units to split.
    splitting_probability : float
        Total fraction of units to split. Each of the four split strategies
        receives an equal share (splitting_probability / 4).
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    split_analyzer : SortingAnalyzer
        New sorting analyzer with artificial splits applied, using the
        same recording as the input analyzer.
    split_ids : dict[unit_id, list[unit_id]]
        Maps each original unit ID to the two new unit IDs created by its split.
        Unit ID type matches `sorting_analyzer.unit_ids` (int or str).
    split_types : dict[unit_id, str]
        Maps each original unit ID to the split strategy used
        ("burst", "amplitude", "drift", or "random").
    """
    splittable_ids = []
    for unit_id in sorting_analyzer.unit_ids:
        spike_train = sorting_analyzer.sorting.get_unit_spike_train(unit_id)
        if len(spike_train) >= 1000:
            splittable_ids.append(unit_id)

    all_split_indices = {}
    split_pipeline = [
        ("burst", get_burst_splits, splitting_probability / 4),
        ("amplitude", get_amplitude_splits, splitting_probability / 4),
        ("drift", get_drift_splits, splitting_probability / 4),
        ("random", get_random_splits, splitting_probability / 4),
    ]

    for split_name, split_function, probability in split_pipeline:
        if len(splittable_ids) == 0:
            break

        split_indices = split_function(
            sorting_analyzer,
            splitting_probability=probability,
            splittable_ids=splittable_ids,
            random_seed=random_seed,
        )

        all_split_indices = all_split_indices | split_indices
        splittable_ids = [
            uid for uid in splittable_ids if uid not in list(split_indices.keys())
        ]

    split_analyzer, split_ids, split_types = _create_splits(
        sorting_analyzer, all_split_indices
    )
    invalid_splits = get_invalid_splits(split_analyzer, split_ids)
    for id in invalid_splits:
        del all_split_indices[id]
    split_analyzer, split_ids, split_types = _create_splits(
        sorting_analyzer, all_split_indices
    )

    return split_analyzer, split_ids, split_types


def _create_splits(sorting_analyzer, all_split_indices):
    """Assign new unit IDs to split indices and return a new SortingAnalyzer with the splits applied."""
    new_ids = generate_unit_ids_for_split(
        sorting_analyzer.unit_ids,
        {key: [value[0], value[1]] for key, value in all_split_indices.items()},
        new_id_strategy="append",
    )
    splits = {}
    for i, original_id in enumerate(list(all_split_indices.keys())):
        split_type = all_split_indices[original_id][2]
        splits[original_id] = [
            new_ids[i][0],
            new_ids[i][1],
            split_type,
        ]

    split_analyzer = sorting_analyzer.split_units(
        {key: [value[0], value[1]] for key, value in all_split_indices.items()},
        new_unit_ids=new_ids,
    )
    split_ids = {key: [value[0], value[1]] for key, value in splits.items()}
    split_types = {key: value[2] for key, value in splits.items()}

    return split_analyzer, split_ids, split_types


def get_invalid_splits(split_analyzer, split_ids):
    """Return original unit IDs whose splits produced at least one low-quality unit (low SNR, low firing rate, or high refractory violations)."""
    quality_metrics = compute_quality_metrics(
        split_analyzer, metric_names=["snr", "firing_rate"]
    )
    ccg_ext = split_analyzer.get_extension("correlograms")
    ccgs, ccg_bins = ccg_ext.get_data()
    acgs = np.zeros((ccgs.shape[0], ccgs.shape[-1]))
    for i in range(ccgs.shape[0]):
        acgs[i] = ccgs[i, i, :]

    smoothed_slid_rps = np.zeros(acgs.shape[0])
    for i in range(acgs.shape[0]):
        smoothed_slid_rps[i] = _sliding_RP_viol_pair(
            acgs[i], bin_size_ms=np.diff(ccg_bins)[0]
        )
    smoothed_slid_rps = pd.Series(smoothed_slid_rps, index=split_analyzer.unit_ids)

    # unit ids created by splitting (dtype-agnostic; works for int or str unit ids)
    new_split_unit_ids = [uid for pair in split_ids.values() for uid in pair]
    split_quality_metrics = quality_metrics.loc[new_split_unit_ids]
    split_quality_metrics.insert(
        2, "sliding_rp_viol", smoothed_slid_rps.loc[new_split_unit_ids]
    )

    good_units = (
        (split_quality_metrics["snr"] > 2)
        & (split_quality_metrics["sliding_rp_viol"] < 0.1)
        & (split_quality_metrics["firing_rate"] > 0.1)
    )
    good_units = set(split_quality_metrics[good_units].index)
    good_splits = []
    for original_id, new_ids in split_ids.items():
        if new_ids[0] in good_units and new_ids[1] in good_units:
            good_splits.append(original_id)

    return np.setdiff1d(list(split_ids.keys()), good_splits)


def get_drift_splits(
    sorting_analyzer,
    splitting_probability,
    splittable_ids=None,
    random_seed=0,
):
    """
    Create artificial splits simulating drift-related oversplitting.

    Splits units by time with a linearly increasing probability of splitting spikes
    as the recording progresses. Spikes in the first 40% of the recording have a
    0-20% chance of being split, spikes between 40-60% have a 20-80% chance, and
    spikes in the latter 40% have an 80-100% chance.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing the units to split.
    splitting_probability : float
        The proportion of splittable units to split (0-1).
    splittable_ids : array-like of unit_id, optional
        Specific unit IDs to consider for splitting. If None, all units in
        `sorting_analyzer` are considered.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    split_indices : dict[unit_id, list]
        Maps each split unit ID to `[unit_1_spike_indices, unit_2_spike_indices, "drift"]`,
        where the spike indices (into the unit's own spike train) indicate which
        spikes were assigned to each of the two resulting units.

    Notes
    -----
    Only single-segment recordings are supported.
    Only units with at least 30% of spikes in each half of the recording are
    considered as split candidates.
    """
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting_analyzer.sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    total_samples = sorting_analyzer.get_num_samples()

    split_candidates = _get_drift_split_candidates(
        splittable_ids, spikes, spike_indices, total_samples
    )
    num_splits = min(num_splits, len(split_candidates))

    # perform the splits
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)
    split_indices = {}

    drift_cutoff_1 = int(total_samples * 0.4)
    drift_cutoff_2 = int(total_samples * 0.6)

    for unit_id in unit_ids_to_split:
        original_spike_indices = spike_indices[0][unit_id]
        original_spike_times = spikes[0][original_spike_indices]["sample_index"]

        # spikes in the first 40% of recording have a 0-20% chance of being split
        first_portion_splits = _split_spike_portion(
            original_spike_indices, original_spike_times, 0, drift_cutoff_1, 0, 0.2, rng
        )
        # spikes between 40-60% have a 20-80% chance of being split
        middle_portion_splits = _split_spike_portion(
            original_spike_indices,
            original_spike_times,
            drift_cutoff_1,
            drift_cutoff_2,
            0.2,
            0.8,
            rng,
        )
        # spikes in the latter 40% have a 80-100% chance of being split
        last_portion_splits = _split_spike_portion(
            original_spike_indices,
            original_spike_times,
            drift_cutoff_2,
            total_samples,
            0.8,
            1,
            rng,
        )
        unit_2_spike_indices = np.concatenate(
            [first_portion_splits, middle_portion_splits, last_portion_splits]
        )
        unit_1_spike_indices = np.setdiff1d(
            np.arange(original_spike_indices.shape[0]), unit_2_spike_indices
        )
        split_indices[unit_id] = [unit_1_spike_indices, unit_2_spike_indices, "drift"]

    return split_indices


def get_amplitude_splits(
    sorting_analyzer,
    splitting_probability,
    splittable_ids=None,
    random_seed=0,
):
    """
    Create artificial splits simulating amplitude-based oversplitting.

    Splits units by separating spikes with higher amplitudes from those with lower
    amplitudes.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing the units to split. The "spike_amplitudes"
        extension is computed automatically if not already present.
    splitting_probability : float
        The proportion of splittable units to split (0-1).
    splittable_ids : array-like of unit_id, optional
        Specific unit IDs to consider for splitting. If None, all units in
        `sorting_analyzer` are considered.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    split_indices : dict[unit_id, list]
        Maps each split unit ID to `[unit_1_spike_indices, unit_2_spike_indices, "amplitude"]`,
        where the spike indices (into the unit's own spike train) indicate which
        spikes were assigned to each of the two resulting units.

    Notes
    -----
    Only single-segment recordings are supported.
    Only units with amplitude variance between the 75th and 95th percentile are
    considered as split candidates.
    """
    if not sorting_analyzer.has_extension("spike_amplitudes"):
        sorting_analyzer.compute("spike_amplitudes")
    spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()

    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting_analyzer.sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    split_candidates = _get_amplitude_split_candidates(
        splittable_ids, spike_indices, spike_amplitudes
    )
    num_splits = min(num_splits, len(split_candidates))

    # perform the splits
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)
    split_indices = {}

    for unit_id in unit_ids_to_split:
        original_spike_indices = spike_indices[0][unit_id]
        original_spike_amplitudes = spike_amplitudes[original_spike_indices]

        split_ratio = rng.uniform(0.3, 0.5)
        amplitude_cutoff = np.quantile(original_spike_amplitudes, split_ratio)

        unit_2_spike_indices = np.argwhere(
            original_spike_amplitudes >= amplitude_cutoff
        ).flatten()
        unit_1_spike_indices = np.setdiff1d(
            np.arange(original_spike_indices.shape[0]), unit_2_spike_indices
        )
        split_indices[unit_id] = [
            unit_1_spike_indices,
            unit_2_spike_indices,
            "amplitude",
        ]

    return split_indices


def get_burst_splits(
    sorting_analyzer,
    splitting_probability,
    splittable_ids=None,
    random_seed=0,
):
    """
    Create artificial splits simulating burst-related oversplitting.

    Splits units by separating the second half of each burst into a new unit.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing the units to split.
    splitting_probability : float
        The proportion of splittable units to split (0-1).
    splittable_ids : array-like of unit_id, optional
        Specific unit IDs to consider for splitting. If None, all units in
        `sorting_analyzer` are considered.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    split_indices : dict[unit_id, list]
        Maps each split unit ID to `[unit_1_spike_indices, unit_2_spike_indices, "burst"]`,
        where the spike indices (into the unit's own spike train) indicate which
        spikes were assigned to each of the two resulting units.

    Notes
    -----
    Only single-segment recordings are supported.
    Only units with at least 30% of spikes in bursts are considered as split candidates.
    Bursts are defined as 3 or more consecutive spikes with inter-spike intervals < 20ms.
    """
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting_analyzer.sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )

    split_candidates, bursts = _get_burst_split_candidates(
        splittable_ids,
        spikes,
        spike_indices,
        sorting_analyzer.sampling_frequency,
    )
    num_splits = min(num_splits, len(split_candidates))

    # perform the splits
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)
    split_indices = {}

    for unit_id in unit_ids_to_split:
        original_spike_indices = spike_indices[0][unit_id]
        unit_2_spike_indices = []
        unit_bursts = bursts[unit_id]
        # split the second half of each burst into new unit
        for burst_start, num_isis in unit_bursts.items():
            split_start = burst_start + num_isis // 2
            unit_2_spike_indices.append(
                np.arange(split_start, burst_start + num_isis + 1)
            )

        unit_2_spike_indices = np.concatenate(unit_2_spike_indices)
        unit_1_spike_indices = np.setdiff1d(
            np.arange(original_spike_indices.shape[0]), unit_2_spike_indices
        )
        split_indices[unit_id] = [unit_1_spike_indices, unit_2_spike_indices, "burst"]

    return split_indices


def get_random_splits(
    sorting_analyzer,
    splitting_probability,
    splittable_ids=None,
    random_seed=0,
):
    """
    Create artificial splits by randomly assigning spikes to a new unit.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing the units to split.
    splitting_probability : float
        The proportion of splittable units to split (0-1).
    splittable_ids : array-like of unit_id, optional
        Specific unit IDs to consider for splitting. If None, all units in
        `sorting_analyzer` are considered.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    split_indices : dict[unit_id, list]
        Maps each split unit ID to `[unit_1_spike_indices, unit_2_spike_indices, "random"]`,
        where the spike indices (into the unit's own spike train) indicate which
        spikes were assigned to each of the two resulting units.

    Notes
    -----
    Only single-segment recordings are supported.
    All units in splittable_ids are candidates for splitting.
    """
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids

    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting_analyzer.sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    num_splits = min(num_splits, len(splittable_ids))

    # perform the splits
    unit_ids_to_split = rng.choice(splittable_ids, num_splits, replace=False)
    split_indices = {}

    for unit_id in unit_ids_to_split:
        original_spike_indices = spike_indices[0][unit_id]
        split_ratio = rng.uniform(0.3, 0.5)

        unit_2_spike_indices = rng.choice(
            np.arange(original_spike_indices.shape[0]),
            int(split_ratio * original_spike_indices.shape[0]),
            replace=False,
        )
        unit_1_spike_indices = np.setdiff1d(
            np.arange(original_spike_indices.shape[0]), unit_2_spike_indices
        )
        split_indices[unit_id] = [unit_1_spike_indices, unit_2_spike_indices, "random"]

    return split_indices


def _split_spike_portion(
    spike_idxs, spike_times, start_sample, end_sample, start_prob, end_prob, rng
):
    """
    Split spikes in a time window with linearly varying probability.

    The probability of splitting increases linearly from start_prob at start_sample
    to end_prob at end_sample.
    """
    portion_idxs = np.intersect1d(
        np.argwhere(spike_times >= start_sample).flatten(),
        np.argwhere(spike_times < end_sample).flatten(),
    )
    portion_times = spike_times[portion_idxs]
    split_draws = rng.uniform(size=portion_idxs.shape[0])

    # split probability increases linearly from start_prob (at start_sample) to end prob (at end_sample)
    split_probs = start_prob + (end_prob - start_prob) * (
        portion_times - start_sample
    ) / (end_sample - start_sample)

    portion_split_idxs = np.argwhere(
        split_draws < split_probs
    ).flatten()  # indices in the portion to split
    unit_split_idxs = portion_idxs[portion_split_idxs]  # indices in the unit to split

    return unit_split_idxs


def _get_drift_split_candidates(
    splittable_ids, spikes, spike_indices, total_samples, min_percent_per_half=0.3
):
    """
    Identify units suitable for drift-based splitting.

    Returns units that have at least min_percent_per_half of spikes in each half
    of the recording, ensuring the unit is active throughout the recording.
    """
    split_candidates = []

    # drift split candidates have at least min_percent_per_half of spikes in each half of the recording
    for unit_id in splittable_ids:
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_times = spikes[0][unit_spike_idxs]["sample_index"]
        percent_in_first_half = (
            unit_spike_times < total_samples // 2
        ).sum() / unit_spike_times.shape[0]

        if (
            percent_in_first_half >= min_percent_per_half
            and percent_in_first_half <= 1 - min_percent_per_half
        ):
            split_candidates.append(unit_id)
    return np.array(split_candidates)


def _get_amplitude_split_candidates(
    splittable_ids,
    spike_indices,
    spike_amplitudes,
    low_cutoff=75,
    high_cutoff=95,
):
    """
    Identify units suitable for amplitude-based splitting.

    Returns units with amplitude variance between the low_cutoff and high_cutoff
    percentiles, ensuring sufficient amplitude variation for meaningful splits.
    """
    amplitude_variances = []

    for unit_id in splittable_ids:
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_amplitudes = spike_amplitudes[unit_spike_idxs]
        amplitude_variances.append(np.var(unit_spike_amplitudes))
    amplitude_variances = np.array(amplitude_variances)

    # amplitude split candidates have amplitude variance betweeen the low and high percentile of all units
    candidate_indices = np.intersect1d(
        np.argwhere(
            amplitude_variances >= np.percentile(amplitude_variances, low_cutoff)
        ).flatten(),
        np.argwhere(
            amplitude_variances <= np.percentile(amplitude_variances, high_cutoff)
        ).flatten(),
    )

    return np.array(splittable_ids)[candidate_indices]


def _get_burst_split_candidates(
    splittable_ids,
    spikes,
    spike_indices,
    fs,
    isi_threshold_s=0.02,
    min_burst_fraction=0.3,
):
    """
    Identify units suitable for burst-based splitting.

    Returns units with at least min_burst_fraction of spikes in bursts, where bursts
    are defined as 3+ consecutive spikes with ISI < isi_threshold_s. Also returns
    the burst locations for each candidate unit.
    """
    split_candidates = []
    bursts = {}
    for unit_id in splittable_ids:
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_times = spikes[0][unit_spike_idxs]["sample_index"] / fs
        isis = np.diff(unit_spike_times)
        num_burst_isis = 0
        is_burst = False
        burst_idxs = {}
        burst_start = -1

        # burst split candidates have at least min_burst_fraction of spikes in bursts
        # defined as 3 or more consecutive spikes with ISI < isi_threshold_s
        if np.quantile(isis, min_burst_fraction) < isi_threshold_s:
            for i in range(isis.shape[0]):
                if isis[i] < isi_threshold_s:
                    if not is_burst:
                        burst_start = i
                        is_burst = True
                    num_burst_isis += 1
                elif isis[i] > isi_threshold_s and is_burst:
                    if num_burst_isis >= 2:
                        burst_idxs[burst_start] = num_burst_isis
                    is_burst = False
                    num_burst_isis = 0
            # edge case where burst extends to end of recording
            if is_burst and num_burst_isis >= 2:
                burst_idxs[burst_start] = num_burst_isis

            if (
                sum(count + 1 for count in burst_idxs.values())
                / unit_spike_times.shape[0]
                >= min_burst_fraction
            ):
                split_candidates.append(unit_id)
                bursts[unit_id] = burst_idxs

    return split_candidates, bursts
