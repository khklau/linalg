from copy import deepcopy

import numpy as np
import pytest

from linalg.row.reduction import reduce_row_with_pivot


@pytest.mark.parametrize(
    "test_case, input, pivot_idx, tgt_pivot, expected",
    [
        (
            "target_pivot_1",
            np.array([3, 21, 15], dtype=np.dtype(float)),
            0,
            1.0,
            np.array([1, 7, 5], dtype=np.dtype(float)),
        ),
        (
            "target_pivot_2",
            np.array([0, 8, 20], dtype=np.dtype(float)),
            1,
            2.0,
            np.array([0, 2, 5], dtype=np.dtype(float)),
        ),
    ]
)
def test_reduce_row_with_pivot(test_case: str, input: np.ndarray, pivot_idx: int, tgt_pivot: float, expected: np.ndarray):
    actual = reduce_row_with_pivot(input, pivot_idx, tgt_pivot)
    assert np.array_equal(expected, actual), f"{test_case}: wrong reduction result"
