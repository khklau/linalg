from copy import deepcopy

import numpy as np
import pytest

from linalg.matrix.elimination import eliminate_value
from linalg.matrix.matrix import Matrix


@pytest.mark.parametrize(
    "test_case, input, col_idx, src_row, tgt_row, expected",
    [
        (
            "eliminate_2nd_row_1st_col",
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
            0,
            0,
            1,
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 1, 3],
                        [1, 11, 7]
                    ],
                    dtype=np.dtype(float)
                )
            ),
        ),
        (
            "eliminate_3rd_row_1st_col",
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
            0,
            0,
            2,
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [3, 10, 9],
                        [0, 8, 5]
                    ],
                    dtype=np.dtype(float)
                )
            )
        ),
        (
            "eliminate_3rd_row_2nd_col",
            Matrix.create(
                np.array(
                    [
                        [2, 6, 4],
                        [0, 1, 3],
                        [0, 8, 5]
                    ],
                    dtype=np.dtype(float)
                )
            ),
            1,
            1,
            2,
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
    ]
)
def test_eliminate_value(test_case: str, input: Matrix, col_idx: int, src_row: int, tgt_row: int, expected: Matrix):
    actual = deepcopy(input)
    eliminate_value(actual, col_idx, src_row, tgt_row)
    assert np.array_equal(expected, actual), f"{test_case}: wrong elimination_result"
