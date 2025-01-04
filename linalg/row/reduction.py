import numpy as np


def reduce_row_with_pivot(row: np.ndarray, pivot_col_idx: int, target_pivot_value: float) -> np.ndarray:
    reciprocal = target_pivot_value / row[pivot_col_idx]
    reduced_row = reciprocal * row
    return reduced_row
