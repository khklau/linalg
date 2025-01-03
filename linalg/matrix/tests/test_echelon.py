from copy import deepcopy

import numpy as np
import pytest

from linalg.matrix.echelon import to_row_echelon_form, shift_zero_pivots
from linalg.matrix.matrix import Matrix


@pytest.mark.parametrize(
    "test_case, input, row_range, col_idx, expected",
    [
        (
            "3x3 matrix requires 1st row swapped",
            Matrix.create(
                np.array(
                    [
                        [0, 8, 5],
                        [2, 6, 4],
                        [1, 11, 7],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            range(0, 3),
            0,
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 8, 5],
                        [1, 11, 7],
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
        (
            "3x3 matrix requires 2nd row swapped",
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 0, 3],
                        [0, 8, 5],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            range(1, 3),
            1,
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 8, 5],
                        [0, 0, 3],
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
        (
            "3x3 matrix no op on 3rd row",
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 8, 5],
                        [0, 0, 3],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            range(2, 3),
            2,
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 8, 5],
                        [0, 0, 3],
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
    ]
)
def test_shift_zero_pivots(test_case: str, input: Matrix, row_range: range, col_idx: int, expected: Matrix):
    actual = deepcopy(input)
    shift_zero_pivots(actual, row_range, col_idx)
    assert expected == actual, f"{test_case}: wrong row echelon form"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "2x2 matrix",
            Matrix.create(
                np.array(
                    [
                        [8, 10],
                        [4, 7],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            Matrix.create(
                np.array(
                    [
                        [8, 10],
                        [0, 2],
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
        (
            "3x3 matrix",
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [3, 10, 9],
                        [1, 11, 7],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 1, 3],
                        [0, 0, -19]
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
        (
            "3x3 matrix requires row swapped",
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [3, 9, 9],
                        [1, 11, 7],
                    ],
                    dtype=np.dtype(float)
                )
            ),
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 8, 5],
                        [0, 0, 3]
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
    ]
)
def test_to_row_echelon_form(test_case: str, input: Matrix, expected: Matrix):
    actual = deepcopy(input)
    to_row_echelon_form(actual)
    assert expected == actual, f"{test_case}: wrong row echelon form"
