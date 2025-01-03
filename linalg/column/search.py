from typing import Optional

import numpy as np


def find_non_zero_in_column(nd_array: np.ndarray, row_range: range, col_idx: int) -> Optional[int]:
    non_zero_row = None
    for row_idx in row_range:
        value = nd_array[row_idx, col_idx]
        if not np.isclose(value, 0, atol = 1e-5):
            non_zero_row = row_idx
            break
    return non_zero_row
