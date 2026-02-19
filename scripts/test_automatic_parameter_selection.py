import os

from spikeinterface.core import load_sorting_analyzer

from slay import autoselect_merge_parameters, compute_slay_merges
from slay.autoselect_params import evaluate_merge_predictions


def compare_default_auto_params(sorting_analyzer, model_path):
    default_merges, sorting_analyzer, default_metrics = compute_slay_merges(
        sorting_analyzer,
        merge_parameters={"k1": 0.25, "k2": 1, "merge_threshold": 0.5},
        model_path=model_path,
        # retrain_autoencoder=True,
    )
    print(f"Default original merges: {default_merges}")

    (
        autoselected_parameters,
        autoselected_scores,
        autoselected_merges,
        split_analyzer,
        split_pairs,
        widowed_ids,
    ) = autoselect_merge_parameters(sorting_analyzer, model_path=model_path)
    print(f"Autoselected parameters: {autoselected_parameters}")
    print(f"Autoselected artificial scores: {autoselected_scores}")
    print(f"Autoselected artificial merges: {autoselected_merges}")

    autoselected_merges, sorting_analyzer, autoselected_metrics = compute_slay_merges(
        sorting_analyzer,
        merge_parameters=autoselected_parameters,
        model_path=model_path,
    )
    print(f"Autoselected original merges: {autoselected_merges}")

    default_split_merges, split_analyzer, default_split_metrics = compute_slay_merges(
        split_analyzer,
        merge_parameters={"k1": 0.25, "k2": 1, "merge_threshold": 0.5},
        model_path=model_path,
    )
    print(f"Default split merges: {default_split_merges}")

    (
        num_tp,
        num_fp,
        num_fn,
        precision,
        recall,
        f1,
        tp_merges,
        fp_merges,
        fn_merges,
    ) = evaluate_merge_predictions(default_split_merges, split_pairs, widowed_ids)
    print(num_tp, num_fp, num_fn, precision, recall, f1)


if __name__ == "__main__":
    ks_folder = r"D:\SLAy_data\bijan_monkey_motor\catgt_rec_bank0_dense_g0\rec_bank0_dense_g0_imec0\imec0_ks4"
    path_to_analyzer = os.path.join(ks_folder, "clean_analyzer")
    model_path = os.path.join(ks_folder, "automerge", "autoencoder.pt")

    sorting_analyzer = load_sorting_analyzer(path_to_analyzer)

    compare_default_auto_params(sorting_analyzer, model_path)
