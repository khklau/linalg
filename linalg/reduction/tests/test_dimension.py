from copy import deepcopy
import math

import numpy as np
import pytest

from linalg.reduction.dimension import (
    calc_best_eigen_vector,
    calc_covariance_matrix,
    calc_mean,
    calc_norm_scaling,
    reduce_by_pca,
)


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "1x2 matrix",
            np.array(
                [
                    [2, 5],
                ],
                dtype=np.dtype(float)
            ),
            np.array(
                [
                    [2, 5],
                ],
                dtype=np.dtype(float)
            ),
        ),
        (
            "2x2 matrix",
            np.array(
                [
                    [2, 5],
                    [4, 9],
                ],
                dtype=np.dtype(float)
            ),
            np.array(
                [
                    [3, 7],
                    [3, 7],
                ],
                dtype=np.dtype(float)
            ),
        ),
        (
            "5x3 matrix",
            np.array(
                [
                    [0, 1, 11],
                    [2, 3, 13],
                    [4, 5, 15],
                    [4, 5, 15],
                    [6, 7, 17],
                    [8, 9, 19],
                ],
                dtype=np.dtype(float)
            ),
            np.array(
                [
                    [4, 5, 15],
                    [4, 5, 15],
                    [4, 5, 15],
                    [4, 5, 15],
                    [4, 5, 15],
                    [4, 5, 15],
                ],
                dtype=np.dtype(float)
            ),
        ),
    ]
)
def test_calc_mean(test_case: str, input: np.ndarray, expected: np.ndarray):
    actual = calc_mean(input)
    assert np.array_equal(actual, expected), f"{test_case}: wrong mean calculated"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "4x2 matrix",
            np.array(
                [
                    [1.0, 1.0],
                    [1.2, 1.6],
                    [-0.5, 0.2],
                    [-1.3, -0.6],
                ],
                dtype=np.dtype(float)
            ),
            np.array(
                [
                    [4.34/3.0, 3.38/3.0],
                    [3.38/3.0, 2.75/3.0],
                ],
                dtype=np.dtype(float)
            ),
        ),
    ]
)
def test_calc_covariance_matrix(test_case: str, input: np.ndarray, expected: np.ndarray):
    actual = calc_covariance_matrix(input)
    assert np.allclose(actual, expected, rtol=1e-05), f"{test_case}: wrong covariance calculated"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "2x2 matrix",
            np.array(
                [
                    [4.34/3.0, 3.38/3.0],
                    [3.38/3.0, 2.75/3.0],
                ],
                dtype=np.dtype(float)
            ),
            np.array([0.78388745, 0.62090294])
        ),
    ]
)
def test_calc_best_eigen_vector(test_case: str, input: np.ndarray, expected: np.ndarray):
    actual = calc_best_eigen_vector(input)
    assert np.allclose(actual, expected, rtol=1e-05), f"{test_case}: wrong eigen vector selected"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "whole number L2 norm",
            np.array([3.0, 4.0], dtype=np.dtype(float)),
            1.0/5.0,
        ),
    ]
)
def test_calc_norm_scaling(test_case: str, input: np.ndarray, expected: float):
    actual = calc_norm_scaling(input)
    assert math.isclose(expected, actual, rel_tol=1e-05), f"{test_case}: wrong L2 norm scaling calculated"


@pytest.mark.parametrize(
    "test_case, input, expected",
    [
        (
            "4x2 matrix",
            np.array(
                [
                    [1.0, 1.0],
                    [1.2, 1.6],
                    [-0.5, 0.2],
                    [-1.3, -0.6],
                ],
                dtype=np.dtype(float)
            ),
            np.array([1.40479039, 1.93410965, -0.26776314, -1.39159545], dtype=np.dtype(float)),
        ),
    ]
)
def test_reduce_by_pca(test_case: str, input: np.ndarray, expected: np.ndarray):
    actual = reduce_by_pca(input)
    assert np.allclose(actual, expected, rtol=1e-05), f"{test_case}: wrong PCA reduction"
