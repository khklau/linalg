from typing import Optional

import numpy as np
import pytest

from linalg.column.search import find_non_zero_in_column


@pytest.mark.parametrize(
    "test_case, nd_array, row_range, col_idx, expected",
    [
        (
            "search full column with all non-zero",
            np.array(
                [
                    [9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]
                ]
            ),
            range(3),
            1,
            0
        ),
        (
            "search full column with all zero",
            np.array(
                [
                    [9, 0, 7],
                    [6, 0, 4],
                    [3, 0, 1]
                ]
            ),
            range(3),
            1,
            None
        ),
        (
            "search full column with some zero",
            np.array(
                [
                    [9, 0, 7],
                    [6, 0, 4],
                    [3, 2, 1]
                ]
            ),
            range(3),
            1,
            2
        ),
        (
            "search sub column with all non-zero",
            np.array(
                [
                    [9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]
                ]
            ),
            range(1, 3),
            2,
            1
        ),
        (
            "search sub column with all zero",
            np.array(
                [
                    [9, 8, 7],
                    [6, 5, 0],
                    [3, 2, 0]
                ]
            ),
            range(1, 3),
            2,
            None
        ),
        (
            "search sub column with some zero",
            np.array(
                [
                    [9, 8, 7],
                    [6, 5, 0],
                    [3, 2, 1]
                ]
            ),
            range(1, 3),
            2,
            2
        ),
    ]
)
def test_find_non_zero_in_column(test_case: str, nd_array: np.ndarray, row_range: range, col_idx: int, expected: Optional[int]):
    actual = find_non_zero_in_column(nd_array, row_range, col_idx)
    assert actual == expected, f"{test_case}: wrong row with non zero value found"
