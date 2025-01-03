from typing import List, Tuple

import numpy as np

from linalg.matrix.matrix import Matrix
from linalg.row.search import find_left_most_non_zero


def main_diagonal(matrix: Matrix) -> List[Tuple[int, int]]:
    shape = np.shape(matrix.nd_array)
    if len(shape) < 2:
        return []
    diagonal_len = min(shape[-1], shape[-2])
    indices = [
        (index, index)
        for index in range(diagonal_len)
    ]
    return indices

def find_pivots(matrix: Matrix) -> List[Tuple[int, int]]:
    pivots = []
    shape = np.shape(matrix.nd_array)
    if len(shape) < 2:
        return pivots
    col_count = shape[-1]
    row_count = shape[-2]
    pivot_col_idx = 0
    for row_idx in range(row_count):
        if matrix.is_augmented:
            # Can't search for pivots in last column of augmented matrices since it contains targets
            col_range = range(pivot_col_idx, col_count - 1)
        else:
            col_range = range(pivot_col_idx, col_count)
        non_zero_col_idx = find_left_most_non_zero(matrix.nd_array[row_idx, :], col_range)
        if non_zero_col_idx is None:
            continue
        pivots.append((row_idx, non_zero_col_idx))
        # Next pivot must be right of the previous pivot
        pivot_col_idx = non_zero_col_idx + 1
    return pivots
