from typing import List, Tuple

import numpy as np

from linalg.matrix.matrix import Matrix


def main_diagonal(matrix: Matrix) -> List[Tuple[int, int]]:
    shape = np.shape(matrix.nd_array)
    if len(shape) < 1:
        return []
    diagonal_len = min(shape[-1], shape[-2])
    indices = [
        (index, index)
        for index in range(diagonal_len)
    ]
    return indices
