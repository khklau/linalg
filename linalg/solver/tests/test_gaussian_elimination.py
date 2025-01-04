from copy import deepcopy

import numpy as np
import pytest

from linalg.solver.gaussian_elimination import solve


@pytest.mark.parametrize(
    "test_case, coefficients, targets, expected",
    [
        (
            "Non-singular 2x2 augmented matrix",
            np.array(
                [
                    [2, 5],
                    [8, 1],
                ],
                dtype=np.dtype(float)
            ),
            np.array([[46], [32]], dtype=np.dtype(float)),
            np.array([3, 8], dtype=np.dtype(float))
        ),
        (
            "Non-singular 3x3 matrix",
            np.array(
                [
                    [2, -1, -1],
                    [2, 2, 4],
                    [4, 1, 0],
                ],
                dtype=np.dtype(float)
            ),
            np.array([[1], [-2], [-1]], dtype=np.dtype(float)),
            np.array([0, -1, 0], dtype=np.dtype(float))
        ),
    ]
)
def test_solve(test_case: str, coefficients: np.ndarray, targets: np.ndarray, expected: np.ndarray):
    actual = solve(coefficients, targets)
    assert np.array_equal(actual, expected), f"{test_case}: wrong solution"
