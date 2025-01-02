
import numpy as np
import pytest

from linalg.matrix.elimination import eliminate_value


@pytest.mark.parametrize(
    "test_case, input, col_idx, src_row, tgt_row, expected",
    [
        (
            "eliminate_2nd_row_1st_col",
            np.array(
                [
                    [2, 6, 4],
                    [3, 10, 9],
                    [1, 11, 7],
                ]
            ),
            0,
            0,
            1,
            np.array(
                [
                    [2, 6, 4],
                    [0, 1, 3],
                    [1, 11, 7]
                ]
            )
        ),
        (
            "eliminate_3rd_row_1st_col",
            np.array(
                [
                    [2, 6, 4],
                    [3, 10, 9],
                    [1, 11, 7],
                ]
            ),
            0,
            0,
            2,
            np.array(
                [
                    [2, 6, 4],
                    [3, 10, 9],
                    [0, 8, 5]
                ]
            )
        ),
        (
            "eliminate_3rd_row_2nd_col",
            np.array(
                [
                    [2, 6, 4],
                    [0, 1, 3],
                    [0, 8, 5]
                ]
            ),
            1,
            1,
            2,
            np.array(
                [
                    [2, 6, 4],
                    [0, 1, 3],
                    [0, 0, -19]
                ]
            )
        ),
    ]
)
def test_eliminate_value(test_case: str, input: np.ndarray, col_idx: int, src_row: int, tgt_row: int, expected: np.ndarray):
    actual = input.copy()
    eliminate_value(actual, col_idx, src_row, tgt_row)
    assert np.array_equal(expected, actual), f"{test_case}: wrong elimination_result"
