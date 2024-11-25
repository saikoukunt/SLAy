"""
Functions and helpers to calculate normalized inner products between waveforms. 

These functions are not used in the merge finder unless the option to use mean
similarity is selected (autoencoder-based similarity is the default).
"""

import numpy as np
from numpy.typing import NDArray


def calc_wf_norms(wfs: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Calculates the Frobenius norm of each waveform in an array of waveforms.

    Args:
        wfs (NDArray): Array containing waveforms. The first dimension of the
            array must index the  waveforms [i.e. shape = (# of waveforms, # channels,
            # timepoints) OR (# of waveforms, # timepoints, # channels)]

    Returns:
        wf_norm (NDArray): Array of waveform norms.
    """
    wf_norms = np.zeros(wfs.shape[0])

    for i in range(wfs.shape[0]):
        wf_norms[i] = np.linalg.norm(wfs[i])

    return wf_norms


def wf_means_similarity(
    mean_wf: NDArray[np.float_],
    cl_good: NDArray[np.bool_],
    use_jitter,
    max_jitter,
) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]]:
    """
    Calculates the normalized pairwise similarity (inner product) between pairs of
    waveforms in an array of waveforms.

    Args:
        mean_wf (NDArray): Array containing unnormalized waveforms. The first dim
            of the array must index the waveforms [i.e. shape = (# of waveforms,
            # channels, # timepoints) OR (# of waveforms, # timepoints, # channels)].
        cl_good (NDArray): Array containing cluster quality labels. Bad clusters
            will be skipped during similarity calculations.
        jitter (bool): True if similarity calculations should check for time shifts
            between waveforms. Defaults to False. Note that runtime scales linearly
            with jitter_amt if enabled.
        jitter_amt (int): The maximum amount of shift to search in both directions.
            Note that runtime scales linearly with jitter_amt if enabled.

    Returns:
        mean_sim (NDArray): The (maximum) pairwise similarity for each pair of
            (normalized) waveforms.
        wf_norms (NDArray): Calculated waveform norms.
        shifts (NDArray): If jitter is enabled, the shift that produces the
            max inner product for each pair of waveforms.
    """
    n_clust: int = mean_wf.shape[0]
    mean_sim: NDArray[np.float_] = np.zeros((n_clust, n_clust))
    shifts: NDArray[np.int_] = np.zeros((n_clust, n_clust), dtype="int16")
    wf_norms: NDArray[np.float_] = calc_wf_norms(mean_wf)

    for i in range(n_clust):
        for j in range(n_clust):
            if cl_good[i] and cl_good[j]:
                if i != j:
                    norm: NDArray[np.float_] = max(wf_norms[i], wf_norms[j])
                    if norm == 0:
                        continue

                    if use_jitter:
                        sim: float
                        off: int
                        sim, off = find_jitter(mean_wf[i], mean_wf[j], max_jitter)
                        mean_sim[i, j] = sim / (norm**2)
                        shifts[i, j] = off
                    else:
                        mean_sim[i, j] = np.dot(
                            mean_wf[i].flatten(), mean_wf[j].flatten()
                        ) / (norm**2)

    return mean_sim, wf_norms, shifts


def find_jitter(
    m1: NDArray[np.float_], m2: NDArray[np.float_], max_jitter: int
) -> tuple[float, int]:
    """
    Finds the time shift that maximizes the inner product between two waveforms.

    To avoid spurious shifts, a nonzero shift must increase the inner product by < 0.1
    to be counted.

    Args:
        m1 (NDArray): Input waveform with shape (# channels, # timepoints)
        m2 (NDArray): Input waveforms with shape (# channels, # timepoints)
        max_jitter (int): The maximum shift to consider (in samples).

    Returns:
        mean_sim (float): The maximum inner product.
        jitter (int): The amount of shift that produced the maximum inner product.
            Specifies the amount that m2 shifted (negative value signifies
            m2 had to be shifted left).
    """
    jitter = 0
    mean_sim: float = np.dot(m1.flatten(), m2.flatten())
    t_length: int = m1.shape[1]

    for i in range(-1 * max_jitter, max_jitter + 1):
        if i < 0:
            c1_shift: NDArray[np.float_] = m1[:, : t_length + i]
            c2_shift: NDArray[np.float_] = m2[:, i * -1 :]
        else:
            c1_shift = m1[:, i:]
            c2_shift: NDArray[np.float_] = m2[:, : t_length - i]

        off_sim: float = np.dot(c1_shift.flatten(), c2_shift.flatten())
        if off_sim > mean_sim:
            mean_sim = off_sim
            jitter: int = i
        if (mean_sim - np.dot(m1.flatten(), m2.flatten())) < 0.1:
            mean_sim = np.dot(m1.flatten(), m2.flatten())
            jitter = 0
    return mean_sim, jitter
