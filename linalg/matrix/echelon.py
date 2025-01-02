import numpy as np

from linalg.matrix.matrix import Matrix
from linalg.matrix.elimination import eliminate_value
from linalg.matrix.search import main_diagonal


def to_row_echelon_form(matrix: Matrix):
    row_count = np.shape(matrix.nd_array)[-2]
    for diag_row_idx, diag_col_idx in main_diagonal(matrix):
        for elim_row_idx in range(diag_row_idx + 1, row_count):
            eliminate_value(matrix, diag_col_idx, diag_row_idx, elim_row_idx)
