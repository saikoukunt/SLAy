import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    data_dir = "D:/SLAY_data"

    precision_avg = []
    precision_burst = []
    precision_drift = []
    precision_amplitude = []
    precision_random = []
    recall = []

    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if "ks" in dir_name and "as" not in dir_name and "orig" not in dir_name:
                ks_dir = os.path.join(root, dir_name)
                print(ks_dir)
                file_dir = os.path.join(ks_dir, "split_experiment")

                tp_avg = np.load(os.path.join(file_dir, "tp_avg.npy"))

                precision_avg.append(np.load(os.path.join(file_dir, "tp_avg.npy")))
                precision_burst.append(np.load(os.path.join(file_dir, "tp_bursts.npy")))
                precision_drift.append(np.load(os.path.join(file_dir, "tp_drift.npy")))
                precision_amplitude.append(
                    np.load(os.path.join(file_dir, "tp_amp.npy"))
                )
                precision_random.append(
                    np.load(os.path.join(file_dir, "tp_random.npy"))
                )
                recall.append(
                    tp_avg / (np.load(os.path.join(file_dir, "fp.npy")) + tp_avg)
                )

    print(precision_avg)
    sns.barplot(
        data=precision_avg,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Precision across all splits")
    plt.savefig("../results/figures/as_precision.svg", transparent=True, dpi=300)

    plt.figure()
    sns.barplot(
        data=recall,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Recall across all splits")
    plt.savefig("../results/figures/as_recall.svg", transparent=True, dpi=300)

    plt.figure()
    sns.barplot(
        data=precision_burst,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Precision for Burst Splits")
    plt.savefig("../results/figures/as_precision_burst.svg", transparent=True, dpi=300)

    plt.figure()
    sns.barplot(
        data=precision_amplitude,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Precision for Amplitude Splits")
    plt.savefig(
        "../results/figures/as_precision_amplitude.svg", transparent=True, dpi=300
    )

    plt.figure()
    sns.barplot(
        data=precision_drift,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Precision for Drift Splits")
    plt.savefig("../results/figures/as_precision_drift.svg", transparent=True, dpi=300)

    plt.figure()
    sns.barplot(
        data=precision_random,
    )
    plt.xlabel("Dataset")
    plt.ylabel("Precision for Random Splits")
    plt.savefig("../results/figures/as_precision_random.svg", transparent=True, dpi=300)
