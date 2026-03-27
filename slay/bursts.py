from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def base_algo(
    spike_times: NDArray[np.float64], state_ratio: float = 5, gamma: float = 0.3
) -> tuple[pd.DataFrame, NDArray[np.int_]]:
    """
    Uses the original Kleinberg algorithm to estimate the optimal HMM state sequence,
    and returns an interpretable output.

    Args:
        spike_times (NDArray): Spike times in seconds.
        state_ratio (float, optional): The geometric ratio between the firing rates of
            adjacent HMM states. Defaults to 5, which seems appropriate empirically.
        gamma (float, optional): The cost coefficient of state transitions. Transition cost
            is 0 if transitioning to a lower state and gamma*(j-i) if transitioning
            to a higher state. Defaults to 0.3, which seems appropriate empirically.

    Returns:
        bursts (pd.DataFrame): Onset and offsets of detected bursts.
        q (NDArray): The inferred optimal state sequence.
    """
    q, _ = find_sequence(spike_times, state_ratio, gamma)
    bursts = create_output(spike_times, q)
    return bursts, q


def find_bursts(
    spike_times: NDArray[np.float64],
    state_ratio: float = 5,
    gamma: float = 0.3,
    max_iter: int = 5,
) -> tuple[pd.DataFrame, NDArray[np.int_], NDArray[np.float64]]:
    """
    Uses our EM-Kleinberg algorithm to iteratively estimate HMM state firing rates
    and the optimal state sequence, and returns an interpretable output.

    Args:
        spike_times (NDArray): Spike times in seconds.
        state_ratio (float, optional): The geometric ratio between the firing rates of
            adjacent HMM states. Defaults to 5, which seems appropriate empirically.
        gamma (float, optional): The cost coefficient of state transitions. Transition cost
            is 0 if transitioning to a lower state and gamma*(j-i) if transitioning
            to a higher state. Defaults to 0.3, which seems appropriate empirically.
        max_iter (int, optional): The maximum number of EM iterations to run. Defaults to 5.

    Returns:
        bursts (pd.DataFrame): Onset and offsets of detected bursts. Dataframe with columns 'level', 'start', 'end'
        q (NDArray): The inferred optimal state sequence. (length is number of spikes - 1)
        a (NDArray): The inferred HMM state firing rates. (length is number of states used)
    """
    q, a = find_sequence(spike_times, state_ratio, gamma)
    gaps = np.diff(spike_times)

    max_level = -1
    ind = 1

    # EM LOOP
    while True:
        q_old = q

        # Update firing rate estimates given state sequence.
        k = a.shape[0]
        for i in range(1, k):
            inds = np.nonzero(q_old == i)[0]
            if inds.shape[0] == 0:  # Stop looking higher if a state is unused.
                max_level = i
                break
            a_hat = 1 / (np.mean(gaps[inds]))  # MLE estimator for exponential dist
            a[i] = max(state_ratio * a[i - 1], a_hat)

        # Update state sequence estimate given firing rates.
        ind += 1
        q, a = find_sequence(spike_times, state_ratio, gamma, a)

        if np.array_equal(q, q_old) or ind == max_iter:
            break

    bursts = create_output(spike_times, q)

    return bursts, q, a[: max_level + 1]


def find_sequence(
    spike_times: NDArray[np.float64],
    state_ratio: float,
    gamma: float,
    frs: NDArray[np.float64] = None,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """
    Infers the optimal state sequence of Kleinberg HMM bursting states.

    Args:
        spike_times (NDArray): Spike times in seconds.
        state_ratio (float): The geometric ratio between the firing rates of
            adjacent HMM states.
        gamma (float): The cost coefficient of state transitions. Transition cost
            is 0 if transitioning to a lower state and gamma*(j-i) if transitioning
            to a higher state.
        frs (NDArray, optional): Preset firing rates for each state.

    Returns:
        q (NDArray): The inferred optimal state sequence.
        frs (NDArray): HMM state firing rates. This is the same as the input
            parameter if provided. Otherwise, it is calculated from the baseline
            spike rate and `state_ratio`.
    """

    spike_times = np.sort(spike_times)
    gaps = np.diff(spike_times)

    # calculate base rate and number of HMM states (k)
    T = np.sum(gaps)
    n = gaps.shape[0]
    base_fr = n / T

    try:
        k = int(
            np.ceil(
                1
                + np.emath.logn(state_ratio, T)
                + np.emath.logn(state_ratio, 1 / gaps.min())
            )
        )
    except OverflowError:
        print(
            f"spike_times: {len(spike_times)}, state_ratio: {state_ratio}, T: {T}, gaps.min(): {gaps.min()}"
        )
        raise OverflowError

    # Precompute transition penalties between all state pairs.
    log_n = np.log(n)
    state_inds = np.arange(k)
    tau = np.where(
        state_inds[:, None] >= state_inds[None, :],
        0.0,
        (state_inds[None, :] - state_inds[:, None]) * gamma * log_n,
    )
    if frs is None:
        frs = (state_ratio ** np.arange(k)) * base_fr
    neg_log_frs = -np.log(frs)

    # Viterbi algorithm to estimate optimal state sequence
    C = np.zeros(k)  # cost
    backptr = np.zeros((n, k), dtype="int")
    k_inds = np.arange(k)

    for t in range(n):
        # candidate[ell, j] is the cost of previous state ell transitioning to j.
        candidate = C[:, None] + tau
        best_prev = np.argmin(candidate, axis=0)
        backptr[t, :] = best_prev

        # Cj = least previous cost - exponential pmf
        with np.errstate(divide="ignore"):
            C = candidate[best_prev, k_inds] + neg_log_frs + frs * gaps[t]

    # get optimal state sequence
    q = np.zeros(n, dtype="int")
    q[-1] = np.argmin(C)
    for t in range(n - 1, 0, -1):
        q[t - 1] = backptr[t, q[t]]

    return q, frs


def create_output(
    spike_times: NDArray[np.float64], q: NDArray[np.int_]
) -> pd.DataFrame:
    """
    Finds bursts (continuous portions with the same state) from an inferred state
    sequence.

    Args:
        spike_times (NDArray): Spike times in seconds.
        q (NDArray): Inferred state sequence.

    Returns:
        bursts (pd.DataFrame): Onset and offsets of detected bursts.
    """
    # calculate output size
    prev_q = -1
    N = 0
    n: int = np.diff(spike_times).shape[0]

    # calculate output size
    for t in range(n - 1):
        if (q[t] > prev_q) and (q[t + 1] >= q[t]):
            N += q[t] - prev_q
        prev_q: int = q[t]
    N = int(N)

    if N == 0:
        bursts = pd.DataFrame(
            {"level": [0], "start": [spike_times[0]], "end": [spike_times[-1]]}
        )
        return bursts

    # create output
    level: NDArray[np.int_] = np.zeros(N, dtype="int")
    start: NDArray[np.float64] = np.zeros(N, dtype="float64")
    end: NDArray[np.float64] = np.zeros(N, dtype="float64")

    # populate output
    burst_ind = 0
    prev_q = -1
    stack: NDArray[np.int_] = np.zeros(N, dtype="int")
    stack_ind = 0

    for t in range(n - 1):
        if (q[t] > prev_q) and (q[t + 1] >= q[t]):  # if new bursts need to be added
            n_new = int(q[t] - prev_q)
            new_inds = np.arange(burst_ind, burst_ind + n_new)
            level[new_inds] = np.arange(prev_q + 1, q[t] + 1)
            start[new_inds] = spike_times[t]
            stack[stack_ind : stack_ind + n_new] = new_inds
            burst_ind += n_new
            stack_ind += n_new

        elif q[t] < prev_q:  # if old bursts need to be closed
            n_close = int(prev_q - q[t])
            for i in range(n_close):
                stack_ind = max(0, stack_ind - 1)
                end[stack[stack_ind]] = spike_times[t]

        prev_q = q[t]

    # close bursts that include last spike
    end[stack[:stack_ind]] = spike_times[-1]

    bursts = pd.DataFrame({"level": level, "start": start, "end": end})

    return bursts
