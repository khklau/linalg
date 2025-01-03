from typing import Optional

import numpy as np


def find_left_most_non_zero(row: np.ndarray, col_range: range) -> Optional[int]:
    non_zero = None
    for idx in col_range:
        if not np.isclose(row[idx], 0, atol=1e-5):
            non_zero = idx
            break
    return non_zero
