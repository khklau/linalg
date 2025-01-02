from dataclasses import dataclass

import numpy as np

from linalg.matrix.properties import Singularity


@dataclass
class Matrix:
    nd_array: np.array
    augmented: bool
    singularity: Singularity

    @staticmethod
    def create_augmented(coefficients: np.ndarray, targets: np.ndarray) -> "Matrix":
        singularity = Singularity.SINGULAR
        determinant = np.linalg.det(coefficients)
        if not np.isclose(determinant, 0):
            singularity = Singularity.NON_SINGULAR
        augmented = np.hstack((coefficients, targets))
        return Matrix(
            nd_array=augmented,
            augmented=True,
            singularity=singularity
        )

    @staticmethod
    def create(nd_array: np.ndarray) -> "Matrix":
        determinant = np.linalg.det(nd_array)
        singularity = Singularity.SINGULAR
        if not np.isclose(determinant, 0):
            singularity = Singularity.NON_SINGULAR
        return Matrix(
            nd_array=nd_array,
            augmented=False,
            singularity=singularity
        )
