import numpy as np

from linalg.matrix.matrix import Matrix
from linalg.matrix.echelon import to_reduced_row_echelon_form


def solve(coefficients: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = Matrix.create_augmented(coefficients.copy(), targets.copy())
    to_reduced_row_echelon_form(matrix)
    solution = matrix.nd_array[:, -1]
    return solution
