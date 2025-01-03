from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np

from linalg.matrix.properties import Singularity


@dataclass
class Matrix:
    nd_array: np.array
    is_augmented: bool
    singularity: Singularity

    def __eq__(self, other: "Matrix") -> bool:
        is_equal = (
            np.array_equal(self.nd_array, other.nd_array)
            and self.is_augmented == other.is_augmented
            and self.singularity == other.singularity
        )
        return is_equal

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, deepcopy(value, memo))
        return new

    @staticmethod
    def create_augmented(coefficients: np.ndarray, targets: np.ndarray) -> "Matrix":
        singularity = Singularity.SINGULAR
        determinant = np.linalg.det(coefficients)
        if not np.isclose(determinant, 0, atol=1e-5):
            singularity = Singularity.NON_SINGULAR
        augmented = np.hstack((coefficients, targets))
        return Matrix(
            nd_array=augmented,
            is_augmented=True,
            singularity=singularity
        )

    @staticmethod
    def create(nd_array: np.ndarray) -> "Matrix":
        determinant = np.linalg.det(nd_array)
        singularity = Singularity.SINGULAR
        if not np.isclose(determinant, 0, atol=1e-5):
            singularity = Singularity.NON_SINGULAR
        return Matrix(
            nd_array=nd_array,
            is_augmented=False,
            singularity=singularity
        )
