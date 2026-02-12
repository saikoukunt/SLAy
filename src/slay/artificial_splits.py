import numpy as np
from spikeinterface.core.sorting_tools import spike_vector_to_indices
from spikeinterface.core.numpyextractors import NumpySorting


def make_artificial_splits(sorting_analyzer, random_seed=0):

    pass


def make_drift_splits(
    sorting_analyzer, splitting_probability, splittable_ids=None, random_seed=0
):
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    sorting = sorting_analyzer.sorting
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    total_samples = sorting_analyzer.recording.get_num_samples()

    split_candidates = _get_drift_split_candidates(
        splittable_ids, spikes, spike_indices, total_samples
    )
    num_splits = min(num_splits, len(split_candidates))

    # perform the splits
    new_spikes = spikes[0].copy()
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)
    drift_cutoff_1 = int(total_samples * 0.4)
    drift_cutoff_2 = int(total_samples * 0.6)

    new_id = max(sorting_analyzer.unit_ids) + 1
    new_unit_ids = list(sorting_analyzer.unit_ids)
    new_idx = len(sorting_analyzer.unit_ids)
    split_pairs = []
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

        new_spikes["unit_index"][split_spike_idxs] = new_idx
        split_pairs.append((unit_id, new_id))
        new_unit_ids.append(new_id)
        new_id += 1
        new_idx += 1

    new_sorting = NumpySorting(
        new_spikes,
        sampling_frequency=sorting_analyzer.sampling_frequency,
        unit_ids=new_unit_ids,
    )

    return new_sorting, split_pairs


def make_amplitude_splits(
    sorting_analyzer, splitting_probability, splittable_ids=None, random_seed=0
):
    if not sorting_analyzer.has_extension("spike_amplitudes"):
        sorting_analyzer.compute("spike_amplitudes")
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    sorting = sorting_analyzer.sorting
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    split_candidates = _get_amplitude_split_candidates(
        splittable_ids, spike_indices, spike_amplitudes
    )
    num_splits = min(num_splits, len(split_candidates))

    # perform the splits
    new_spikes = spikes[0].copy()
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)

    new_id = max(sorting_analyzer.unit_ids) + 1
    new_unit_ids = list(sorting_analyzer.unit_ids)
    new_idx = len(sorting_analyzer.unit_ids)
    split_pairs = []

    for unit_id in unit_ids_to_split:
        unit_spike_idxs = spike_indices[0][unit_id]
        unit_spike_amplitudes = spike_amplitudes[unit_spike_idxs]

        split_ratio = rng.uniform(0.3, 0.5)
        amplitude_cutoff = np.quantile(unit_spike_amplitudes, split_ratio)
        split_spike_idxs = unit_spike_idxs[unit_spike_amplitudes >= amplitude_cutoff]

        new_spikes["unit_index"][split_spike_idxs] = new_idx
        split_pairs.append((unit_id, new_id))
        new_unit_ids.append(new_id)
        new_id += 1
        new_idx += 1

    new_sorting = NumpySorting(
        new_spikes,
        sampling_frequency=sorting_analyzer.sampling_frequency,
        unit_ids=new_unit_ids,
    )

    return new_sorting, split_pairs


def make_burst_splits(
    sorting_analyzer, splitting_probability, splittable_ids=None, random_seed=0
):
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids
    sorting = sorting_analyzer.sorting
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting.to_spike_vector(concatenated=False)
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
    new_spikes = spikes[0].copy()
    unit_ids_to_split = rng.choice(split_candidates, num_splits, replace=False)

    new_id = max(sorting_analyzer.unit_ids) + 1
    new_unit_ids = list(sorting_analyzer.unit_ids)
    new_idx = len(sorting_analyzer.unit_ids)
    split_pairs = []

    for unit_id in unit_ids_to_split:
        unit_spike_idxs = spike_indices[0][unit_id]
        split_spike_idxs = []
        unit_bursts = bursts[unit_id]
        # split the second half of each burst into new unit
        for burst_start, num_isis in unit_bursts.items():
            split_start = burst_start + num_isis // 2
            split_spike_idxs.append(np.arange(split_start, burst_start + num_isis + 1))

        split_spike_idxs = unit_spike_idxs[np.concatenate(split_spike_idxs)]
        new_spikes["unit_index"][split_spike_idxs] = new_idx
        split_pairs.append((unit_id, new_id))
        new_unit_ids.append(new_id)
        new_id += 1
        new_idx += 1

    new_sorting = NumpySorting(
        new_spikes,
        sampling_frequency=sorting_analyzer.sampling_frequency,
        unit_ids=new_unit_ids,
    )

    return new_sorting, split_pairs


def make_random_splits(
    sorting_analyzer, splitting_probability, splittable_ids=None, random_seed=0
):
    if splittable_ids is None:
        splittable_ids = sorting_analyzer.unit_ids

    sorting = sorting_analyzer.sorting
    rng = np.random.default_rng(random_seed)

    num_splits = int(splitting_probability * len(sorting_analyzer.unit_ids))
    spikes = sorting.to_spike_vector(concatenated=False)
    assert len(spikes) == 1, "Only single-segment recordings supported"

    spike_indices = spike_vector_to_indices(
        spikes, sorting_analyzer.unit_ids, absolute_index=True
    )
    num_splits = min(num_splits, len(splittable_ids))

    # perform the splits
    new_spikes = spikes[0].copy()
    unit_ids_to_split = rng.choice(splittable_ids, num_splits, replace=False)

    new_id = max(sorting_analyzer.unit_ids) + 1
    new_unit_ids = list(sorting_analyzer.unit_ids)
    new_idx = len(sorting_analyzer.unit_ids)
    split_pairs = []

    for unit_id in unit_ids_to_split:
        unit_spike_idxs = spike_indices[0][unit_id]
        split_ratio = rng.uniform(0.3, 0.5)
        split_spike_idxs = rng.choice(
            unit_spike_idxs,
            int(split_ratio * unit_spike_idxs.shape[0]),
            replace=False,
        )

        new_spikes["unit_index"][split_spike_idxs] = new_idx
        split_pairs.append((unit_id, new_id))
        new_unit_ids.append(new_id)
        new_id += 1
        new_idx += 1

    new_sorting = NumpySorting(
        new_spikes,
        sampling_frequency=sorting_analyzer.sampling_frequency,
        unit_ids=new_unit_ids,
    )

    return new_sorting, split_pairs


def _split_spike_portion(
    spike_idxs, spike_times, start_sample, end_sample, start_prob, end_prob, rng
):
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

    return spike_idxs[unit_split_idxs]


def _get_drift_split_candidates(
    splittable_ids, spikes, spike_indices, total_samples, min_percent_per_half=0.3
):
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
