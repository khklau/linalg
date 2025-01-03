import numpy as np
import pytest

from linalg.matrix.matrix import Matrix


@pytest.mark.parametrize(
    "test_case, input",
    [
        (
            "Empty matrix",
            np.zeros(())
        ),
        (
            "Empty 1D matrix",
            np.array([], dtype=np.dtype(float))
        ),
        (
            "1x3 matrix",
            np.array([[1, 2, 3]], dtype=np.dtype(float))
        )
    ]
)
def test_create_invalid(test_case: str, input: np.ndarray):
    with pytest.raises(np.linalg.LinAlgError) as exc:
        Matrix.create(input)


@pytest.mark.parametrize(
    "test_case, coefficients, targets",
    [
        (
            "Empty coefficients",
            np.zeros(()),
            np.array([[1], [2]], dtype=np.dtype(float))
        ),
        (
            "Empty targets",
            np.array([[1, 2], [3, 4]], dtype=np.dtype(float)),
            np.zeros(()),
        ),
        (
            "1D coefficients",
            np.array([[1, 2, 3]], dtype=np.dtype(float)),
            np.array([[1], [2]], dtype=np.dtype(float))
        ),
        (
            "1D targets",
            np.array([[1, 2], [3, 4]], dtype=np.dtype(float)),
            np.array([[1, 2]], dtype=np.dtype(float))
        )
    ]
)
def test_create_augmented_invalid(test_case: str, coefficients: np.ndarray, targets: np.ndarray):
    with pytest.raises(np.linalg.LinAlgError) as exc:
        Matrix.create(input)

