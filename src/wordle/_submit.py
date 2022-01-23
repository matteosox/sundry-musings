"""Module of functions for calculating the Wordle response"""

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np

from wordle._types import UIntArrayType, UIntType

__all__ = ["submit"]


def submit(guess: str, answer: str) -> str:
    """
    Submit a guess for a given answer, producing the Wordle response
    """
    match_inds = []
    leftover_answer_letters = answer
    for ind, (g_letter, a_letter) in enumerate(zip(guess, answer)):
        if g_letter == a_letter:
            match_inds.append(ind)
            leftover_answer_letters = leftover_answer_letters.replace(g_letter, "", 1)

    response = ""
    for ind, letter in enumerate(guess):
        if ind in match_inds:
            response += "🟩"
        elif letter in leftover_answer_letters:
            response += "🟨"
            leftover_answer_letters = leftover_answer_letters.replace(letter, "", 1)
        else:
            response += "⬛"

    return response


def calc_array(
    guesses: tuple[str, ...],
    response_index: Mapping[str, UIntType],
    max_workers: Optional[int] = None,
) -> UIntArrayType:
    """
    Calculate an array of responses, given the provided tuple of allowed guesses,
    a mapping to encode the response strings into indices, and an optional
    maximum number of workers to spread the work across.
    """
    calc_row = partial(_calc_row, answers=guesses, response_index=response_index)
    with ProcessPoolExecutor(max_workers) as executor:
        num_workers = executor._max_workers  # type: ignore # pylint: disable=protected-access
        chunksize = np.ceil(len(guesses) / num_workers).astype(int)
        return np.array(list(executor.map(calc_row, guesses, chunksize=chunksize)))


def _calc_row(
    guess: str, answers: tuple[str], response_index: Mapping[str, UIntType]
) -> list[UIntType]:
    return [response_index[submit(guess, answer)] for answer in answers]
