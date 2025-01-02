from typing import List, Tuple

import numpy as np
import pytest

from linalg.matrix.matrix import Matrix
from linalg.matrix.search import main_diagonal


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [1, 2],
                        [3, 4]
                    ]
                )
            ),
            [(0, 0), (1, 1)]
        ),
        (
            "3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                )
            ),
            [(0, 0), (1, 1), (2, 2)]
        ),
        (
            "2x2 augmented matrix",
            Matrix.create_augmented(
                np.array(
                    [
                        [1, 2],
                        [3, 4]
                    ]
                ),
                np.array([[10], [20]]),
            ),
            [(0, 0), (1, 1)]
        ),
        (
            "3x3 augmented matrix",
            Matrix.create_augmented(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
                np.array([[10], [20], [30]]),
            ),
            [(0, 0), (1, 1), (2, 2)]
        ),
    ]
)
def test_main_diagonal(test_case: str, input: Matrix, expected: List[Tuple[int, int]]):
    actual = main_diagonal(input)
    assert expected == actual, f"{test_case}: wrong diagonal returned"
