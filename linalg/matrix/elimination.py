import numpy as np

from linalg.matrix.matrix import Matrix


def eliminate_value(matrix: Matrix, col_idx: int, src_row_idx: int, tgt_row_idx: int):
    reciprocal = matrix.nd_array[tgt_row_idx, col_idx] / matrix.nd_array[src_row_idx, col_idx]
    src_row_product = reciprocal * matrix.nd_array[src_row_idx]
    tgt_row_elim = matrix.nd_array[tgt_row_idx] - src_row_product
    matrix.nd_array[tgt_row_idx] = tgt_row_elim
