from copy import deepcopy

import numpy as np
import pytest

from linalg.matrix.echelon import to_row_echelon_form
from linalg.matrix.matrix import Matrix


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
                    ]
                )
            ),
            Matrix.create(
                np.array(
                    [
                        [8, 10],
                        [0, 2],
                    ]
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
                    ]
                )
            ),
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 1, 3],
                        [0, 0, -19]
                    ]
                )
            )
        ),
    ]
)
def test_to_row_echelon_form(test_case: str, input: Matrix, expected: Matrix):
    actual = deepcopy(input)
    to_row_echelon_form(actual)
    assert expected == actual, f"{test_case}: wrong row echelon form"
