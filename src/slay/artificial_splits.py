import numpy as np
from spikeinterface.core.sorting_tools import spike_vector_to_indices
from spikeinterface.core.numpyextractors import NumpySorting


def make_artificial_splits(sorting_analyzer, random_seed=0):

    pass


# reimplementing bc spikeinterface version duplicates unit IDs if original unit IDs are non-contiguous
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
    unit_ids_to_split = rng.choice(split_candidates, num_splits)
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
        print(split_spike_idxs)

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

    return new_sorting, new_spikes, split_pairs


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

    # a unit is a candidate for time splits if it has at least min_percent_per_half of its spikes in each half of the recording
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
    return split_candidates


def make_amplitude_splits(sorting_analyzer, random_seed=0):
    pass


def make_burst_splits(sorting_analyzer, random_seed=0):
    pass


def make_random_splits(sorting_analyzer, random_seed=0):
    pass
