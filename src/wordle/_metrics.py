"""Module of metrics for Wordle greedy search algorithm"""

from typing import Union

import numpy as np

from wordle._types import FloatArrayType, IntArrayType, UIntArrayType

ResponseDType = Union[np.uint8, np.uint16, np.uint32]


def score(
    metric: str,
    array: UIntArrayType,
    possible_guesses: IntArrayType,
    possible_answers: IntArrayType,
) -> FloatArrayType:
    """
    Score the array of responses on the provided metric,
    given the possible guesses and answers.
    """
    return _metrics[metric](array, possible_guesses, possible_answers)


def _expected_value(
    array: UIntArrayType, possible_guesses: IntArrayType, possible_answers: IntArrayType
) -> FloatArrayType:
    """
    Calculate the expected number of possible answers for each possible guess, given
    an existing set of possible answers, corrected for the possibility that the guess
    can be correct.
    """
    mean_num_left = []
    for guess_ind in possible_guesses:
        row = array[guess_ind]
        _, counts = np.unique(row, return_counts=True)  # type: ignore
        mean_num_left.append(counts.dot(counts))
    can_be_correct = np.isin(possible_guesses, possible_answers, assume_unique=True)  # type: ignore
    return (np.array(mean_num_left) - can_be_correct) / array.shape[1]  # type: ignore


def _minimax(
    array: UIntArrayType, possible_guesses: IntArrayType, possible_answers: IntArrayType
) -> FloatArrayType:
    """
    Calculate the maximum number of possible answers for each possible guess, given
    an existing set of possible answers. The possibility that the guess can be correct
    is used as a potential tiebreaker. See also:
    https://en.wikipedia.org/wiki/Minimax
    """
    max_num_left = []
    for guess_ind in possible_guesses:
        row = array[guess_ind]
        _, counts = np.unique(row, return_counts=True)  # type: ignore
        max_num_left.append(counts.max())
    can_be_correct = np.isin(possible_guesses, possible_answers, assume_unique=True)  # type: ignore
    return np.array(max_num_left) - can_be_correct * 0.1  # type: ignore


def _partitions(
    array: UIntArrayType, possible_guesses: IntArrayType, possible_answers: IntArrayType
) -> FloatArrayType:
    """
    Calculate the number of possible responses for each possible guess, given
    an existing set of possible answers. This is negated since we want to maximize this, i.e.
    we'd like to spread out the possible answers into as many groups as possible. The
    possibility that the guess can be correct is used as a potential tiebreaker.
    """
    partitions = []
    for guess_ind in possible_guesses:
        row = array[guess_ind]
        partitions.append(np.unique(row).shape[0])  # type: ignore
    can_be_correct = np.isin(possible_guesses, possible_answers, assume_unique=True)  # type: ignore
    return -np.array(partitions) - can_be_correct * 0.1  # type: ignore


def _entropy(
    array: UIntArrayType, possible_guesses: IntArrayType, _unused: IntArrayType
) -> FloatArrayType:
    """
    Calculate the Shannon entropy of the probability distribution of responses for each guess, given
    an existing set of possible answers. Shannon entropy can be thought of as the average level of "information"
    contained in a response, so this is negated since we want to maximize this. See also:
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    entropies = []
    for guess_ind in possible_guesses:
        row = array[guess_ind]
        _, counts = np.unique(row, return_counts=True)  # type: ignore
        probs = counts / row.shape[0]
        entropies.append(probs.dot(np.log(probs)))
    return np.array(entropies)


_metrics = {
    "expected_value": _expected_value,
    "minimax": _minimax,
    "partitions": _partitions,
    "entropy": _entropy,
}
