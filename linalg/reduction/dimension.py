import math
import numpy as np
import scipy as sp

def reduce_by_pca(data_set: np.ndarray) -> np.ndarray:
    covariance_matrix = calc_covariance_matrix(data_set)
    best_eigvec = calc_best_eigen_vector(covariance_matrix)
    norm_scaling = calc_norm_scaling(best_eigvec)
    reduction = data_set @ best_eigvec * norm_scaling
    return reduction

def calc_covariance_matrix(data_set: np.ndarray) -> np.ndarray:
    mean_matrix = calc_mean(data_set)
    diff_to_mean_matrix = data_set - mean_matrix
    diff_to_mean_transposed = diff_to_mean_matrix.transpose()
    sample_scaling = 1.0 / float(data_set.shape[0] - 1.0)
    covariance_matrix = diff_to_mean_transposed @ diff_to_mean_matrix * sample_scaling
    return covariance_matrix

def calc_mean(data_set: np.ndarray) -> np.ndarray:
    mean_vector = np.apply_along_axis(np.mean, axis=0, arr=data_set)
    repeated_mean_vector = np.repeat(mean_vector, data_set.shape[0])
    mean_matrix = np.reshape(repeated_mean_vector, data_set.shape, order='F')
    return mean_matrix

def calc_best_eigen_vector(covariance_matrix: np.ndarray) -> np.ndarray:
    cov_eigvalues, cov_eigvec = sp.sparse.linalg.eigsh(covariance_matrix)
    # For highest variance use the largest eigen value
    max_index = np.argmax(cov_eigvalues)
    best_eigvec = cov_eigvec[max_index]
    if (best_eigvec < 0.0).all():
        # reverse the vector
        return best_eigvec * -1.0
    else:
        return best_eigvec

def calc_norm_scaling(eigen_vector: np.ndarray) -> float:
    vec_squared = np.apply_along_axis(np.square, arr=eigen_vector, axis=0)
    l2_norm = math.sqrt(np.sum(vec_squared))
    norm_scaling = 1.0 / l2_norm
    return norm_scaling
