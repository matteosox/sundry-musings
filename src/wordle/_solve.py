"""Module of utilities for solving Wordle"""

import itertools
from collections.abc import Iterable
from typing import Optional, Type

import numpy as np
from pkg_resources import resource_filename

from wordle import _metrics as metrics
from wordle._submit import calc_array
from wordle._types import FloatArrayType, IntArrayType, UIntArrayType, UIntType

__all__ = ["ANSWERS", "OTHER_GUESSES", "Wordle"]

with open(
    resource_filename("wordle", "answers.txt"), "r", encoding="utf-8"
) as answers_file:
    ANSWERS = tuple(answers_file.read().splitlines())

with open(
    resource_filename("wordle", "other_guesses.txt"), "r", encoding="utf-8"
) as guesses_file:
    OTHER_GUESSES = tuple(guesses_file.read().splitlines())


class _SolutionMap(dict[str, "_SolutionMap"]):
    """
    Subclassing dict is a code smell, but I'm being lazy right now
    and this is relatively benign monkeypatching.
    """

    def __init__(self, guess: str, is_leaf: bool = False) -> None:
        super().__init__()
        self.guess = guess
        self.is_leaf = is_leaf

    @property
    def leaf_depths(self) -> dict[str, int]:
        """Returns a dictionary of the depth of each leaf"""
        solution_maps = [self]

        counter = itertools.count(1)
        scores = {}
        while guess_num := next(counter):
            if not solution_maps:
                break
            next_solution_maps: list["_SolutionMap"] = []
            for solution_map in solution_maps:
                if solution_map.is_leaf:
                    scores[solution_map.guess] = guess_num
                next_solution_maps.extend(val for val in solution_map.values())
            solution_maps = next_solution_maps
        return scores

    @property
    def leaf_depth_distribution(self) -> dict[int, int]:
        """Returns a dictionary of the count of leaves for each depth"""
        depths, counts = np.unique(list(self.leaf_depths.values()), return_counts=True)  # type: ignore
        return dict(zip(depths, counts))


class Wordle:
    """Object for solving a given configuration of Wordle"""

    def __init__(
        self,
        answers: Optional[Iterable[str]] = None,
        other_guesses: Optional[Iterable[str]] = None,
        metric: str = "expected_value",
        hard_mode: bool = False,
    ) -> None:
        self.answers = ANSWERS if answers is None else tuple(answers)
        self.guesses = self.answers + (
            OTHER_GUESSES if other_guesses is None else tuple(other_guesses)
        )
        self.metric = metric
        self.hard_mode = hard_mode

        word_len = len(self.guesses[0])
        self._correct_response = "🟩" * word_len

        self._responses = [
            "".join(letters) for letters in itertools.product(*(["🟩🟨⬛"] * word_len))
        ]
        if word_len < 6:
            response_ind_type: Type[UIntType] = np.uint8
        elif word_len < 11:
            response_ind_type = np.uint16
        else:
            response_ind_type = np.uint32

        self._word_index = {guess: ind for ind, guess in enumerate(self.guesses)}
        self._response_index = {
            response: response_ind_type(ind)
            for ind, response in enumerate(self._responses)
        }
        self._array = calc_array(self.guesses, self._response_index)

    def submit(self, guess: str, answer: str) -> str:
        """
        Get the response for a given guess and answer.
        """
        guess_ind = self._word_index[guess]
        answer_ind = self._word_index[answer]
        response_ind = self._array[guess_ind, answer_ind]
        response: str = self._responses[response_ind]
        return response

    def find_possible_answers(
        self, guesses: Optional[dict[str, str]] = None
    ) -> list[str]:
        """
        Find the possible answers given a dictionary where each key is a guess,
        and each values is the response.
        """
        possible_answers = self._find_possible_answers(guesses)
        return [self.answers[answer_ind] for answer_ind in possible_answers]

    def _find_possible_answers(
        self, guesses: Optional[dict[str, str]] = None
    ) -> IntArrayType:
        possible_answers = np.arange(len(self.answers))
        if guesses:
            for guess, response in guesses.items():
                response_ind = self._response_index[response]
                guess_ind = self._word_index[guess]
                possible_answers = self._next_possible_answers(
                    guess_ind, response_ind, possible_answers
                )
        return possible_answers

    def find_possible_guesses(
        self, guesses: Optional[dict[str, str]] = None
    ) -> list[str]:
        """
        Find the possible guesses given a dictionary where each key is a guess,
        and each values is the response.
        """
        possible_guesses = self._find_possible_guesses(guesses)
        return [self.guesses[guess_ind] for guess_ind in possible_guesses]

    def _find_possible_guesses(
        self, guesses: Optional[dict[str, str]] = None
    ) -> IntArrayType:
        possible_guesses = np.arange(len(self.guesses))
        if guesses:
            for guess, response in guesses.items():
                response_ind = self._response_index[response]
                guess_ind = self._word_index[guess]
                possible_guesses = self._next_possible_guesses(
                    guess_ind, response_ind, possible_guesses
                )
        return possible_guesses

    def _next_possible_answers(
        self, guess_ind: int, response_ind: UIntType, possible_answers: IntArrayType
    ) -> IntArrayType:
        subset = self._array[guess_ind, possible_answers]
        next_possible_answers: IntArrayType = possible_answers[subset == response_ind]
        return next_possible_answers

    def _next_possible_guesses(
        self, guess_ind: int, response_ind: UIntType, possible_guesses: IntArrayType
    ) -> IntArrayType:
        if not self.hard_mode:
            return possible_guesses
        subset = self._array[guess_ind, possible_guesses]
        next_possible_guesses: IntArrayType = possible_guesses[subset == response_ind]
        return next_possible_guesses

    def _score(
        self, possible_guesses: IntArrayType, possible_answers: IntArrayType
    ) -> FloatArrayType:
        subset = self._array[:, possible_answers]
        return metrics.score(self.metric, subset, possible_guesses, possible_answers)

    def _find_best_guess(
        self, possible_guesses: IntArrayType, possible_answers: IntArrayType
    ) -> int:
        scores = self._score(possible_guesses, possible_answers)
        best_guess: int = possible_guesses[np.argmin(scores)]
        return best_guess

    def solve(
        self,
        answer: Optional[str] = None,
        prev_guesses: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Find the solution to Wordle.

        If supplied, the possible guesses and answers are narrowed based
        on the previous guesses dictionary `prev_guesses`, where each key is a guess,
        and each values is the response.

        The solver will calculate the best guesses and print them for you. If an
        `answer` is supplied, the solver will choose the best guess, print the
        response, and repeat until the solution is found. If no `answer` is
        supplied, the solver will ask you to choose a guess after printing
        the best guesses, then ask you to supply the response.
        """
        possible_guesses = self._find_possible_guesses(prev_guesses)
        possible_answers = self._find_possible_answers(prev_guesses)
        counter = itertools.count(1)

        while guess_num := next(counter):
            print(f"Round {guess_num}")
            print(f"Currently possible answers: {len(possible_answers)}")

            scores = self._score(possible_guesses, possible_answers)
            scores_index = {
                possible_guesses[ind]: scores[ind] for ind in np.argsort(scores)
            }
            if possible_answers.shape[0] <= 2:
                top_guesses = possible_answers.tolist()
            else:
                top_guesses = list(scores_index.keys())[:5]

            print(
                f"Best guesses, sorted by {self.metric} (out of {len(possible_guesses)}):"
            )
            for top_guess_ind in top_guesses:
                print(
                    f"  - {self.guesses[top_guess_ind]}: {scores_index[top_guess_ind]:.1f}"
                )

            if answer:
                guess = self.guesses[top_guesses[0]]
                guess_ind = self._word_index[guess]
                print(f"Guess: {guess}")
                response = self.submit(guess, answer)
                print(f"Response: {response}")
            else:
                guess = input("Your guess: ")
                guess_ind = self._word_index[guess]
                print(f"Guess score: {scores_index[guess_ind]}")
                response = input("Response (🟩🟨⬛): ")

            if response == self._correct_response:
                print("Yay!")
                break

            print()
            response_ind = self._response_index[response]
            possible_answers = self._next_possible_answers(
                guess_ind, response_ind, possible_answers
            )
            possible_guesses = self._next_possible_guesses(
                guess_ind, response_ind, possible_guesses
            )

    def map_solutions(
        self,
        possible_guesses: Optional[IntArrayType] = None,
        possible_answers: Optional[IntArrayType] = None,
    ) -> _SolutionMap:
        """
        Map the solutions, optionally given a constrained set of possible guesses and answers.
        """
        if possible_guesses is None:
            possible_guesses = self._find_possible_guesses(None)
        if possible_answers is None:
            possible_answers = self._find_possible_answers(None)

        # Optimization for the 1 and 2 possible answer cases
        if possible_answers.shape[0] == 1:
            answer = self.answers[possible_answers[0]]
            solution_map = _SolutionMap(answer, is_leaf=True)
        elif possible_answers.shape[0] == 2:
            answer0 = self.answers[possible_answers[0]]
            answer1 = self.answers[possible_answers[1]]
            solution_map = _SolutionMap(answer0, is_leaf=True)
            solution_map[self.submit(answer0, answer1)] = _SolutionMap(
                answer1, is_leaf=True
            )
        else:
            guess_ind = self._find_best_guess(possible_guesses, possible_answers)
            solution_map = _SolutionMap(self.guesses[guess_ind])
            possible_response_inds: UIntArrayType = np.unique(
                self._array[guess_ind, possible_answers]
            )
            for response_ind in possible_response_inds:
                response = self._responses[response_ind]
                if response == self._correct_response:
                    solution_map.is_leaf = True
                else:
                    next_possible_guesses = self._next_possible_guesses(
                        guess_ind, response_ind, possible_guesses
                    )
                    next_possible_answers = self._next_possible_answers(
                        guess_ind, response_ind, possible_answers
                    )
                    solution_map[response] = self.map_solutions(
                        next_possible_guesses, next_possible_answers
                    )

        return solution_map
