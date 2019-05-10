# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la


def rand_tensor(*shape):
    return np.random.rand(*shape)


def ones_tensor(*shape):
    return np.ones(shape)


def unfold_tensor(tensor, rows_indexes_number=0):
    n = rows_indexes_number
    rows = int(np.product(tensor.shape[:n]))
    cols = int(np.product(tensor.shape[n:]))
    return np.reshape(tensor, (rows, cols))


def fold_matrix(matrix, shape):
    return np.reshape(matrix, shape)


def _svd(matrix, eps=1e-6):
    u, s, v = la.svd(matrix, full_matrices=False)
    id_ = s.size
    for i, s_val in enumerate(s):
        if s_val <= eps:
            id_ = i
            break
    id_ = max(id_, 1)
    return u[:, :id_], s[:id_], v[:id_, :]


def svd_decomposition(matrix, eps=1e-6):
    u, s, v = _svd(matrix, eps)
    return u, np.diag(s) @ v


def singular_amount(matrix):
    _, s, _ = _svd(matrix)
    return np.prod([i**2 for i in s])
