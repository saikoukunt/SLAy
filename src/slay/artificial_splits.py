import numpy as np
from spikeinterface.core.sorting_tools import spike_vector_to_indices
from spikeinterface.core import SortingAnalyzer


# TODO delete some partners to get better testing for false positives
def make_artificial_splits(
    sorting_analyzer: SortingAnalyzer,
    splitting_probability,
    random_seed=0,
    widow_probability=0.5,
):
    """
    Create artificial splits using multiple splitting strategies.

    Applies splits in order of most to least restrictive candidate criteria:
    burst -> amplitude -> drift -> random

    Only units with >= 1000 spikes are eligible for splitting.
    Each unit can only be split once, and units created from splits
    cannot be split again.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing units to split
    random_seed : int, default: 0
        Random seed for reproducibility

    Returns
    -------
    new_analyzer : SortingAnalyzer
        New sorting analyzer with artificial splits applied, using the
        same recording as the input analyzer
    all_split_pairs : list of tuple
        All (original_unit_id, new_unit_id, split_type) tuples from all split types
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

    split_ids = {}
    new_id = max(sorting_analyzer.unit_ids) + 1
    for original_id in all_split_indices.keys():
        split_ids[original_id] = [new_id, new_id + 1]
        new_id += 2

    split_analyzer = sorting_analyzer.split_units(all_split_indices)

    # randomly delete a percentage of split partners (tests false positive merges)
    widowed_ids = []
    delete_ids = []
    rng = np.random.default_rng(random_seed)
    num_splits = len(list(split_ids.keys()))
    num_widows = int(widow_probability * num_splits)
    widow_parent_ids = rng.choice(list(split_ids.keys()), num_widows, replace=False)
    for parent_id in widow_parent_ids:
        unit_id1, unit_id2 = split_ids[parent_id]
        widowed_ids.append(unit_id1)
        delete_ids.append(unit_id2)

        del split_ids[parent_id]

    split_analyzer.remove_units(delete_ids)

    return split_analyzer, list(split_ids.values()), widowed_ids


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
    sorting : BaseSorting
        The sorting object containing the units to split.
    sampling_frequency : float
        The sampling frequency of the recording.
    splitting_probability : float
        The proportion of units to split (0-1).
    splittable_ids : array-like, optional
        Specific unit IDs to consider for splitting. If None, all units are considered.
    spike_amplitudes : array-like, optional
        Ignored for drift splits, included for consistent API.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    new_sorting : NumpySorting
        A new sorting object with the artificial splits applied.
    split_pairs : list of tuple
        List of (original_unit_id, new_unit_id) pairs indicating which units were split.

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
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_times = spikes[0][unit_spike_idxs]["sample_index"]

        # spikes in the first 40% of recording have a 0-20% chance of being split
        first_portion_splits = _split_spike_portion(
            unit_spike_idxs, unit_spike_times, 0, drift_cutoff_1, 0, 0.2, rng
        )
        # spikes between 40-60% have a 20-80% chance of being split
        middle_portion_splits = _split_spike_portion(
            unit_spike_idxs,
            unit_spike_times,
            drift_cutoff_1,
            drift_cutoff_2,
            0.2,
            0.8,
            rng,
        )
        # spikes in the latter 40% have a 80-100% chance of being split
        last_portion_splits = _split_spike_portion(
            unit_spike_idxs,
            unit_spike_times,
            drift_cutoff_2,
            total_samples,
            0.8,
            1,
            rng,
        )
        split_spike_idxs = np.concatenate(
            [first_portion_splits, middle_portion_splits, last_portion_splits]
        )
        unsplit_spike_idxs = np.setdiff1d(
            np.arange(unit_spike_idxs.shape[0]), split_spike_idxs
        )
        split_indices[unit_id] = [unsplit_spike_idxs, split_spike_idxs]

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
    sorting : BaseSorting
        The sorting object containing the units to split.
    sampling_frequency : float
        The sampling frequency of the recording.
    splitting_probability : float
        The proportion of units to split (0-1).
    splittable_ids : array-like, optional
        Specific unit IDs to consider for splitting. If None, all units are considered.
    spike_amplitudes : array-like, required
        Pre-computed spike amplitudes for all spikes.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    new_sorting : NumpySorting
        A new sorting object with the artificial splits applied.
    split_pairs : list of tuple
        List of (original_unit_id, new_unit_id) pairs indicating which units were split.

    Notes
    -----
    Only single-segment recordings are supported.
    Requires spike_amplitudes to be provided.
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
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_amplitudes = spike_amplitudes[unit_spike_idxs]

        split_ratio = rng.uniform(0.3, 0.5)
        amplitude_cutoff = np.quantile(unit_spike_amplitudes, split_ratio)

        split_spike_idxs = np.argwhere(
            unit_spike_amplitudes >= amplitude_cutoff
        ).flatten()
        unsplit_spike_idxs = np.setdiff1d(
            np.arange(unit_spike_idxs.shape[0]), split_spike_idxs
        )
        split_indices[unit_id] = [unsplit_spike_idxs, split_spike_idxs]

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
    sorting : BaseSorting
        The sorting object containing the units to split.
    sampling_frequency : float
        The sampling frequency of the recording.
    splitting_probability : float
        The proportion of units to split (0-1).
    splittable_ids : array-like, optional
        Specific unit IDs to consider for splitting. If None, all units are considered.
    spike_amplitudes : array-like, optional
        Ignored for burst splits, included for consistent API.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    new_sorting : NumpySorting
        A new sorting object with the artificial splits applied.
    split_pairs : list of tuple
        List of (original_unit_id, new_unit_id) pairs indicating which units were split.

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
        unit_spike_idxs = spike_indices[0][unit_id]
        split_spike_idxs = []
        unit_bursts = bursts[unit_id]
        # split the second half of each burst into new unit
        for burst_start, num_isis in unit_bursts.items():
            split_start = burst_start + num_isis // 2
            split_spike_idxs.append(np.arange(split_start, burst_start + num_isis + 1))

        split_spike_idxs = np.concatenate(split_spike_idxs)
        unsplit_spike_idxs = np.setdiff1d(
            np.arange(unit_spike_idxs.shape[0]), split_spike_idxs
        )
        split_indices[unit_id] = [unsplit_spike_idxs, split_spike_idxs]

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
    sorting : BaseSorting
        The sorting object containing the units to split.
    sampling_frequency : float
        The sampling frequency of the recording.
    splitting_probability : float
        The proportion of units to split (0-1).
    splittable_ids : array-like, optional
        Specific unit IDs to consider for splitting. If None, all units are considered.
    spike_amplitudes : array-like, optional
        Ignored for random splits, included for consistent API.
    random_seed : int, default: 0
        Random seed for reproducibility.

    Returns
    -------
    new_sorting : NumpySorting
        A new sorting object with the artificial splits applied.
    split_pairs : list of tuple
        List of (original_unit_id, new_unit_id) pairs indicating which units were split.

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
        unit_spike_idxs = spike_indices[0][unit_id]
        split_ratio = rng.uniform(0.3, 0.5)

        split_spike_idxs = rng.choice(
            np.arange(unit_spike_idxs.shape[0]),
            int(split_ratio * unit_spike_idxs.shape[0]),
            replace=False,
        )
        unsplit_spike_idxs = np.setdiff1d(
            np.arange(unit_spike_idxs.shape[0]), split_spike_idxs
        )
        split_indices[unit_id] = [unsplit_spike_idxs, split_spike_idxs]

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
