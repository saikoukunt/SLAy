from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def base_algo(
    spike_times: NDArray[np.float_], state_ratio: float = 5, gamma: float = 0.3
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
    bursts: pd.DataFrame = create_output(spike_times, q)
    return bursts, q


def find_bursts(
    spike_times: NDArray[np.float_],
    state_ratio: float = 5,
    gamma: float = 0.3,
    max_iter: int = 5,
) -> tuple[pd.DataFrame, NDArray[np.int_], NDArray[np.float_]]:
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
        bursts (pd.DataFrame): Onset and offsets of detected bursts.
        q (NDArray): The inferred optimal state sequence.
        a (NDArray): The inferred HMM state firing rates.
    """
    q, a = find_sequence(spike_times, state_ratio, gamma)
    gaps: NDArray[np.float_] = np.diff(spike_times)

    max_level = -1
    ind = 1

    # EM LOOP
    while True:
        q_old = q

        # Update firing rate estimates given state sequence.
        k: int = a.shape[0]
        for i in range(1, k):
            inds = np.nonzero(q_old == i)[0]
            if inds.shape[0] == 0:  # Stop looking higher if a state is unused.
                max_level: int = i
                break
            a_hat: NDArray[np.float_] = 1 / (
                np.mean(gaps[inds])
            )  # MLE estimator for exponential dist
            a[i] = max(state_ratio * a[i - 1], a_hat)

        # Update state sequence estimate given firing rates.
        ind += 1
        q, a = find_sequence(spike_times, state_ratio, gamma, a)

        if np.array_equal(q, q_old) or ind == max_iter:
            break

    bursts: pd.DataFrame = create_output(spike_times, q)

    return bursts, q, a[:max_level+1]


def find_sequence(
    spike_times: NDArray[np.float_],
    state_ratio: float,
    gamma: float,
    frs: NDArray[np.float_] = None,
) -> tuple[NDArray[np.int_], NDArray[np.float_]]:
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
    gaps: NDArray[np.float_] = np.diff(spike_times)

    # calculate base rate and number of HMM states (k)
    T: np.float_ = np.sum(gaps)
    n: int = gaps.shape[0]
    base_fr: float = n / T  # type: ignore

    k = int(
        np.ceil(
            1
            + np.emath.logn(state_ratio, T)
            + np.emath.logn(state_ratio, 1 / gaps.min())
        )
    )

    # Lambdas for transition costs (tau) and probability density (fire_pdf)
    tau: Callable = lambda i, j: 0 if i >= j else (j - i) * gamma * np.log(n)
    if frs is None:
        frs = (state_ratio ** np.arange(k)) * base_fr
    fire_pdf: Callable = lambda j, x: frs[j] * np.exp(-frs[j] * x)

    # Viterbi algorithm to estimate optimal state sequence
    C: NDArray[np.float_] = np.zeros(k)  # cost
    q: NDArray[np.int_] = np.zeros((k, 1), dtype="int")  # state sequence

    for t in range(n):
        Cprime: NDArray[np.float_] = np.zeros(k)
        qprime: NDArray[np.int_] = np.zeros((k, t + 1), dtype="int")

        for j in range(k):
            # Calculate the cost of transitioning from all possible previous states
            cost = list(map(lambda ell: C[ell] + tau(ell, j), np.arange(k)))
            ell: int = np.argmin(cost)  # type: ignore

            # Cj = least previous cost - exponential pmf
            with np.errstate(divide="ignore"):
                Cprime[j] = cost[ell] - np.log(fire_pdf(j, gaps[t]))
            # update state sequences
            if t == 0:
                qprime[j, 0] = j
            else:
                qprime[j, : t + 1] = np.concatenate((q[ell, :], [j]))  # type: ignore

        C = Cprime
        q = qprime

    # get optimal state sequence
    q = q[np.argmin(C), :]

    return q, frs  # type: ignore


def create_output(spike_times: NDArray[np.float_], q: NDArray[np.int_]) -> pd.DataFrame:
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
    start: NDArray[np.float_] = np.zeros(N, dtype="float64")
    end: NDArray[np.float_] = np.zeros(N, dtype="float64")
    bursts = pd.DataFrame({"level": level, "start": start, "end": end})

    # populate output
    burst_ind = 0
    prev_q = -1
    stack: NDArray[np.int_] = np.zeros(N, dtype="int")
    stack_ind = 0

    for t in range(n - 1):
        if (q[t] > prev_q) and (q[t + 1] >= q[t]):  # if new bursts need to be added
            n_new = int(q[t] - prev_q)
            for i in range(n_new):
                bursts.at[burst_ind, "level"] = prev_q + i + 1
                bursts.at[burst_ind, "start"] = spike_times[t]

                stack[stack_ind] = burst_ind
                burst_ind += 1
                stack_ind += 1

        elif q[t] < prev_q:  # if old bursts need to be closed
            n_close = int(prev_q - q[t])
            for i in range(n_close):
                stack_ind = max(0, stack_ind - 1)
                bursts.at[stack[stack_ind], "end"] = spike_times[t]

        prev_q = q[t]

    # close bursts that include last spike
    while stack_ind > 0:
        stack_ind -= 1
        bursts.at[stack[stack_ind], "end"] = spike_times[-1]

    return bursts
