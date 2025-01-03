import sys
from typing import Optional

import numpy as np

from linalg.column.search import find_non_zero_in_column
from linalg.matrix.matrix import Matrix
from linalg.matrix.elimination import eliminate_value
from linalg.matrix.search import main_diagonal


def to_row_echelon_form(matrix: Matrix):
    row_count = np.shape(matrix.nd_array)[-2]
    for diag_row_idx, diag_col_idx in main_diagonal(matrix):
        non_zero_row = shift_zero_pivots(matrix, range(diag_row_idx, row_count), diag_col_idx)
        if non_zero_row is None:
            # all values in this column are zero, so nothing to do
            continue
        for elim_row_idx in range(diag_row_idx + 1, row_count):
            eliminate_value(matrix, diag_col_idx, diag_row_idx, elim_row_idx)

def shift_zero_pivots(matrix: Matrix, row_range: range, col_idx: int) -> Optional[int]:
    last_col_idx = np.shape(matrix.nd_array)[-1] - 1
    if matrix.is_augmented and col_idx == last_col_idx:
        # In augmented matrices the last column are targets so they cannot be pivots
        return None
    non_zero_row = find_non_zero_in_column(matrix.nd_array, row_range, col_idx)
    if non_zero_row is None:
        # The remaining values in the column are all 0 so there are no pivots for this column
        return None
    current_row = row_range[0]
    if non_zero_row != current_row:
        matrix.nd_array[[current_row, non_zero_row]] = matrix.nd_array[[non_zero_row, current_row]]
    return current_row
