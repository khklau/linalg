from typing import List, Tuple

import numpy as np
import pytest

from linalg.matrix.matrix import Matrix
from linalg.matrix.search import main_diagonal, find_pivots


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
                    ],
                    dtype=np.dtype(float)
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
                    ],
                    dtype=np.dtype(float)
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
                    ],
                    dtype=np.dtype(float)
                ),
                np.array([[10], [20]], dtype=np.dtype(float)),
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
                    ],
                    dtype=np.dtype(float)
                ),
                np.array([[10], [20], [30]], dtype=np.dtype(float)),
            ),
            [(0, 0), (1, 1), (2, 2)]
        ),
    ]
)
def test_main_diagonal(test_case: str, input: Matrix, expected: List[Tuple[int, int]]):
    actual = main_diagonal(input)
    assert expected == actual, f"{test_case}: wrong diagonal returned"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "All zero 2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [0, 0],
                        [0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            []
        ),
        (
            "Some zero but not in diagonal of 2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [0, 1],
                        [0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            [(0, 1)]
        ),
        (
            "Some zero in diagonal of 2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [2, 0],
                        [0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            [(0, 0)]
        ),
        (
            "Non-singular 2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [1, 2],
                        [0, 4],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            [(0, 0), (1, 1)]
        ),
        (
            "All zero 3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            []
        ),
        (
            "Some zero but not in diagonal of 3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [0, 1, 5],
                        [0, 0, 4],
                        [0, 0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            [(0, 1), (1, 2)]
        ),
        (
            "Some zero in diagonal of 3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [2, 0, 3],
                        [0, 7, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            [(0, 0), (1, 1)]
        ),
        (
            "Non-singular 3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [2, 0, 3],
                        [0, 7, 0],
                        [0, 0, 4],
                    ],
                    dtype=np.dtype(float)
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
                        [0, 0],
                    ],
                    dtype=np.dtype(float)
                ),
                np.array([[10], [20]], dtype=np.dtype(float)),
            ),
            [(0, 0)]
        ),
        (
            "3x3 augmented matrix",
            Matrix.create_augmented(
                np.array(
                    [
                        [0, 2, 3],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.dtype(float)
                ),
                np.array([[10], [20], [30]], dtype=np.dtype(float)),
            ),
            [(0, 1)]
        ),
    ]
)
def test_find_pivots(test_case: str, input: Matrix, expected: List[Tuple[int, int]]):
    actual = find_pivots(input)
    assert expected == actual, f"{test_case}: wrong pivots returned"
