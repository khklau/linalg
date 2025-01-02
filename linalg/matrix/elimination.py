import numpy as np

from linalg.matrix.matrix import Matrix


def eliminate_value(matrix: Matrix, col_idx: int, src_row_idx: int, tgt_row_idx: int):
    reciprocal = matrix[tgt_row_idx, col_idx] / matrix[src_row_idx, col_idx]
    src_row_product = reciprocal * matrix[src_row_idx]
    tgt_row_elim = matrix[tgt_row_idx] - src_row_product
    matrix[tgt_row_idx] = tgt_row_elim
