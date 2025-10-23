import json
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import trange

import slay


def run_split_experiment(ks_dir: str, split_ratio: float = 0.15, num_runs=5):
    merge_params = {
        "plot_merges": False,
        "max_spikes": 500,
        "auto_accept_merges": False,
        "ref_pen_bin_width": 0.5,
    }

    # if ks_dir_as doesn't exist, copy ks_dir to ks_dir_as
    ks_dir_as = ks_dir + "_as"
    if not os.path.exists(ks_dir_as):
        shutil.copytree(ks_dir, ks_dir_as)

    tp_avg_all = np.zeros(num_runs, dtype="float")
    tp_bursts_all = np.zeros(num_runs, dtype="float")
    tp_amp_all = np.zeros(num_runs, dtype="float")
    tp_drift_all = np.zeros(num_runs, dtype="float")
    tp_random_all = np.zeros(num_runs, dtype="float")
    fp_all = np.zeros(num_runs, dtype="float")

    for i in trange(num_runs):
        if i == num_runs - 1:
            merge_params["plot_merges"] = True
        # cleanup old files and make backup copies if necessary
        if os.path.exists(os.path.join(ks_dir_as, "automerge")):
            shutil.rmtree(os.path.join(ks_dir_as, "automerge"))
        if os.path.exists(os.path.join(ks_dir_as, "mean_waveforms.npy")):
            os.remove(os.path.join(ks_dir_as, "mean_waveforms.npy"))
        if os.path.exists(os.path.join(ks_dir_as, "split_info.json")):
            os.remove(os.path.join(ks_dir_as, "split_info.json"))
        if not os.path.exists(os.path.join(ks_dir_as, "spike_clusters_old.npy")):
            shutil.copyfile(
                os.path.join(ks_dir_as, "spike_clusters.npy"),
                os.path.join(ks_dir_as, "spike_clusters_old.npy"),
            )
        if not os.path.exists(os.path.join(ks_dir_as, "cluster_group_old.tsv")):
            shutil.copyfile(
                os.path.join(ks_dir_as, "cluster_group.tsv"),
                os.path.join(ks_dir_as, "cluster_group_old.tsv"),
            )

        # load clean ks files
        (
            times,
            amplitudes,
            clusters,
            cl_labels,
            n_clust,
            times_multi,
            counts,
            good_ids,
            data_shape,
        ) = load_ks_files(ks_dir_as)

        # make artificial splits and run slay
        (
            split_dict,
            num_split,
            burst_split_ids,
            amp_split_ids,
            drift_split_ids,
            random_split_ids,
        ) = make_splits(
            split_ratio,
            ks_dir_as,
            times,
            amplitudes,
            clusters,
            cl_labels,
            n_clust,
            times_multi,
            counts,
            good_ids,
            data_shape,
        )
        slay.run.main({"KS_folder": ks_dir_as, **merge_params})

        new2old = json.load(open(os.path.join(ks_dir_as, "automerge", "new2old.json")))
        merges = list(new2old.values())

        (tp_bursts, tp_amplitude, tp_drift, tp_random, tp_avg, fp_rate) = (
            quantify_performance(
                merges,
                num_split,
                split_dict,
                burst_split_ids,
                amp_split_ids,
                drift_split_ids,
                random_split_ids,
                n_clust,
            )
        )

        tp_avg_all[i] = tp_avg
        tp_bursts_all[i] = tp_bursts
        tp_amp_all[i] = tp_amplitude
        tp_drift_all[i] = tp_drift
        tp_random_all[i] = tp_random
        fp_all[i] = fp_rate

    os.makedirs(os.path.join(ks_dir, "split_experiment"), exist_ok=True)
    np.save(os.path.join(ks_dir, "split_experiment", "tp_avg.npy"), tp_avg_all)
    np.save(os.path.join(ks_dir, "split_experiment", "tp_bursts.npy"), tp_bursts_all)
    np.save(os.path.join(ks_dir, "split_experiment", "tp_amp.npy"), tp_amp_all)
    np.save(os.path.join(ks_dir, "split_experiment", "tp_drift.npy"), tp_drift_all)
    np.save(os.path.join(ks_dir, "split_experiment", "tp_random.npy"), tp_random_all)
    np.save(os.path.join(ks_dir, "split_experiment", "fp.npy"), fp_all)


def quantify_performance(
    merges,
    num_split,
    split_dict,
    burst_split_ids,
    amp_split_ids,
    drift_split_ids,
    random_split_ids,
    n_clust,
):
    # divide splits by type
    amplitude_splits = []
    for id in amp_split_ids:
        split = []
        split.append(id)
        split.append(*split_dict[id][1:-1])
        amplitude_splits.append(split)

    burst_splits = []
    for id in burst_split_ids:
        split = []
        split.append(id)
        split.append(*split_dict[id][1:-1])
        burst_splits.append(split)

    drift_splits = []
    for id in drift_split_ids:
        split = []
        split.append(id)
        split.append(*split_dict[id][1:-1])
        drift_splits.append(split)

    random_splits = []
    for id in random_split_ids:
        split = []
        split.append(id)
        split.append(*split_dict[id][1:-1])
        random_splits.append(split)

    # true positives
    num_found_bursts = 0
    for split in burst_splits:
        is_found = [set(split).issubset(merge) for merge in merges]
        if any(is_found):
            num_found_bursts += 1

    num_found_amplitude = 0
    for split in amplitude_splits:
        is_found = [set(split).issubset(merge) for merge in merges]
        if any(is_found):
            num_found_amplitude += 1

    num_found_drift = 0
    for split in drift_splits:
        is_found = [set(split).issubset(merge) for merge in merges]
        if any(is_found):
            num_found_drift += 1

    num_found_random = 0
    for split in random_splits:
        is_found = [set(split).issubset(merge) for merge in merges]
        if any(is_found):
            num_found_random += 1

    tp_bursts = num_found_bursts / num_split
    tp_amplitude = num_found_amplitude / num_split
    tp_drift = num_found_drift / num_split
    tp_random = num_found_random / num_split
    tp_avg = (tp_bursts + tp_amplitude + tp_drift + tp_random) / 4

    all_splits = np.concatenate(
        (burst_splits, amplitude_splits, random_splits, drift_splits)
    )
    # remove
    for split in all_splits:
        try:
            is_found = np.argwhere(
                [set(split).issubset(merge) for merge in merges]
            ).flatten()[0]
            merges.pop(is_found)
        except IndexError:
            pass

    try:
        remaining_merge_ids = np.concatenate(merges)
        n_fp = (remaining_merge_ids >= n_clust).sum()
    except ValueError:
        n_fp = 0

    fp_rate = n_fp / all_splits.shape[0]

    return tp_bursts, tp_amplitude, tp_drift, tp_random, tp_avg, fp_rate


def make_splits(
    split_ratio,
    ks_dir_as,
    times,
    amplitudes,
    clusters,
    cl_labels,
    n_clust,
    times_multi,
    counts,
    good_ids,
    data_shape,
):
    # organize time index and amplitudes by cluster id
    times_multi_idx = {}
    amplitudes_multi = {}
    for i in np.arange(n_clust):
        cl_spike_idx = np.argwhere(clusters == i).flatten()
        cl_spike_times = times[cl_spike_idx]
        cl_spike_idx = cl_spike_idx[
            (cl_spike_times >= 20) & (cl_spike_times < data_shape - 62)
        ]
        times_multi_idx[i] = cl_spike_idx
        amplitudes_multi[i] = amplitudes[cl_spike_idx]

    random_candidates = np.intersect1d(np.argwhere(counts > 1000).flatten(), good_ids)

    # find candidates for burst splits
    burst_candidates = []
    burst_times_all = {}
    for id in random_candidates:
        isis = np.diff(times_multi[id]) / 30000
        if np.percentile(isis, 30) < 0.02:
            # find times of bursts, strict 0.02 cutoff, at least 2 subthreshold isis in a row
            burst_times = {}
            num_spikes = 0
            burst_start = -1
            is_burst = False
            for i in range(isis.shape[0]):
                if isis[i] < 0.02:
                    if not is_burst:
                        burst_start = i
                        is_burst = True
                    num_spikes += 1
                elif isis[i] > 0.02 and is_burst:
                    if num_spikes >= 2:
                        burst_times[burst_start] = num_spikes
                    is_burst = False
                    num_spikes = 0
            if sum(list(burst_times.values())) / 2 >= 300:
                burst_candidates.append(id)
                burst_times_all[id] = burst_times
    burst_candidates = np.array(burst_candidates)

    # find candidates for amplitude splits
    amplitude_vars = np.array([amplitudes_multi[i].var() for i in np.arange(n_clust)])
    cutoff_low = np.percentile(amplitude_vars[good_ids], 75)
    cutoff_high = np.percentile(amplitude_vars[good_ids], 95)
    amplitude_split_candidates = np.intersect1d(
        np.argwhere(amplitude_vars > cutoff_low).flatten(),
        np.argwhere(amplitude_vars < cutoff_high).flatten(),
    )
    amplitude_split_candidates = np.intersect1d(
        random_candidates, amplitude_split_candidates
    )

    # find drift split candidates
    drift_candidates = []
    drift_cutoff = data_shape // 2
    for id in random_candidates:
        if (times_multi[id] > drift_cutoff).sum() / times_multi[id].shape[0] >= 0.3:
            drift_candidates.append(id)
    drift_candidates = np.array(drift_candidates)

    # pick clusters to split and assign them to split types
    split_ratio = 0.15
    num_split = round(
        min(split_ratio * good_ids.shape[0], random_candidates.shape[0]) / 4
    )
    burst_split_ids = np.random.choice(
        burst_candidates,
        size=min(int(num_split), burst_candidates.shape[0]),
        replace=False,
    )
    amplitude_split_candidates = np.setdiff1d(
        amplitude_split_candidates, burst_split_ids
    )
    amp_split_ids = np.random.choice(
        amplitude_split_candidates, size=int(num_split), replace=False
    )
    drift_candidates = np.setdiff1d(
        drift_candidates, np.concatenate((amp_split_ids, burst_split_ids))
    )
    drift_split_ids = np.random.choice(
        drift_candidates, size=int(num_split), replace=False
    )
    random_candidates = np.setdiff1d(
        random_candidates,
        np.concatenate((amp_split_ids, drift_split_ids, burst_split_ids)),
    )
    random_split_ids = np.random.choice(
        random_candidates, size=int(num_split), replace=False
    )

    # perform the splits
    next_id = n_clust
    split_dict = {}
    # burst splits
    for split_id in burst_split_ids:
        split_dict[int(split_id)] = [int(split_id), int(next_id), "burst"]
        split_idx = []
        for start in list(burst_times_all[split_id].keys()):
            num_spikes = burst_times_all[split_id][start]
            split_start = start + num_spikes // 2
            split_idx.append(np.arange(split_start, start + num_spikes))
        split_idx = np.concatenate(split_idx)

        clusters[times_multi_idx[split_id][split_idx]] = next_id
        next_id += 1
    # random splits
    for split_id in random_split_ids:
        split_dict[int(split_id)] = [int(split_id), int(next_id), "random"]
        split_ratio = np.random.uniform(0.3, 0.5)
        split_idx = np.random.choice(
            np.arange(times_multi[split_id].shape[0]),
            size=int(split_ratio * times_multi[split_id].shape[0]),
            replace=False,
        )

        clusters[times_multi_idx[split_id][split_idx]] = next_id
        next_id += 1
    # amplitude splits
    for split_id in amp_split_ids:
        split_dict[int(split_id)] = [int(split_id), int(next_id), "amplitude"]
        split_ratio = np.random.uniform(0.3, 0.5)
        amplitude_cutoff = np.quantile(amplitudes_multi[split_id], split_ratio)
        clusters[
            times_multi_idx[split_id][amplitudes_multi[split_id] > amplitude_cutoff]
        ] = next_id
        next_id += 1
    # drift splits
    drift_cutoff_1 = int(data_shape * 0.4)
    drift_cutoff_2 = int(data_shape * 0.6)
    end = data_shape
    for split_id in drift_split_ids:
        split_dict[int(split_id)] = [int(split_id), int(next_id), "drift"]
        # first 40% is mostly in the original cluster
        first_idx = np.argwhere((times_multi[split_id] < drift_cutoff_1)).flatten()
        for idx in first_idx:
            prob = 0.1 * (times_multi[split_id][idx]) / (drift_cutoff_1)
            if np.random.rand() < prob:
                clusters[times_multi_idx[split_id][idx]] = next_id

        # middle 3rd is a steeper transition
        middle_idx = np.intersect1d(
            np.argwhere(times_multi[split_id] < drift_cutoff_2).flatten(),
            np.argwhere(times_multi[split_id] > drift_cutoff_1).flatten(),
        )
        for idx in middle_idx:
            prob = 0.1 + 0.8 * (times_multi[split_id][idx] - drift_cutoff_1) / (
                drift_cutoff_2 - drift_cutoff_1
            )
            if np.random.rand() < prob:
                clusters[times_multi_idx[split_id][idx]] = next_id

        # last 3rd is mostly in the new cluster
        last_idx = np.argwhere(times_multi[split_id] > drift_cutoff_2).flatten()
        for idx in last_idx:
            prob = 0.9 + 0.1 * (times_multi[split_id][idx] - drift_cutoff_2) / (
                end - drift_cutoff_2
            )
            if np.random.rand() < prob:
                clusters[times_multi_idx[split_id][idx]] = next_id

        next_id += 1

    # save new clusters
    np.save(os.path.join(ks_dir_as, "spike_clusters.npy"), clusters)

    # add rows to cl_labels and save
    for split_id, new_ids in split_dict.items():
        new_id = new_ids[1]
        cl_labels.loc[new_id] = cl_labels.loc[split_id].copy()
        cl_labels.at[new_id, "label"] = "good"
    cl_labels.to_csv(
        os.path.join(ks_dir_as, "cluster_group.tsv"),
        sep="\t",
        index_label="cluster_id",
    )

    # save split info
    split_dict = dict(sorted(split_dict.items(), key=lambda x: x[0]))
    json.dump(
        split_dict,
        open(os.path.join(ks_dir_as, "split_info.json"), "w"),
        separators=(",", ":"),
        indent=2,
    )

    return (
        split_dict,
        num_split,
        burst_split_ids,
        amp_split_ids,
        drift_split_ids,
        random_split_ids,
    )


def load_ks_files(ks_dir):
    args = slay.parse_kilosort_params({"KS_folder": ks_dir})
    schema = slay.schemas.RunParams()
    params = schema.load(args)
    params["meta_path"] = params["data_filepath"].replace(".bin", ".meta")

    times = np.load(os.path.join(ks_dir, "spike_times.npy")).flatten()
    amplitudes = np.load(os.path.join(ks_dir, "amplitudes.npy")).flatten()
    clusters = np.load(os.path.join(ks_dir, "spike_clusters_old.npy")).flatten()
    cl_labels: pd.DataFrame = pd.read_csv(
        os.path.join(ks_dir, "cluster_group_old.tsv"),
        sep="\t",
        index_col="cluster_id",
    )

    if "label" not in cl_labels.columns:
        try:
            cl_labels["label"] = cl_labels["KSLabel"]
        except KeyError:
            cl_labels["label"] = cl_labels["group"]

        # Compute useful cluster info.
        # Load the ephys recording.
    rawData = np.memmap(params["data_filepath"], dtype=params["dtype"], mode="r")
    data = np.reshape(rawData, (int(rawData.size / params["n_chan"]), params["n_chan"]))

    n_clust = clusters.max() + 1
    times_multi = slay.find_times_multi(
        times,
        clusters,
        np.arange(n_clust),
        data,
        params["pre_samples"],
        params["post_samples"],
    )
    counts = np.zeros(n_clust, dtype=int)
    for k, v in times_multi.items():
        counts[k] = len(v)

    # update cl_labels to be list with all cluster_ids
    cl_labels = cl_labels.reindex(np.arange(n_clust))
    good_ids = np.argwhere(
        (counts > params["min_spikes"]) & (cl_labels["label"].isin(params["good_lbls"]))
    ).flatten()

    return (
        times,
        amplitudes,
        clusters,
        cl_labels,
        n_clust,
        times_multi,
        counts,
        good_ids,
        data.shape[0],
    )


if __name__ == "__main__":
    data_dir = "D:/SLAY_data/midbrain_cullen"

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if "ks" in dir_name and "as" not in dir_name and "orig" not in dir_name:
                ks_dir = os.path.join(root, dir_name)
                print(ks_dir)
                run_split_experiment(ks_dir)
