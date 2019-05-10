# *-* coding: utf-8 -*-
import copy
import itertools

import numpy as np
import numpy.linalg as la

import intervals.utils as iutils
from tt import TT
import tt.utils as tt_utils


def bruteforce_solver(func, vars, var_types='', default_type='i'):
    """
    var_types is a sequence of 'i' (interval) and/or 'r' (real) chars
    """
    var_types = var_types.ljust(len(vars), default_type)
    iters = []
    for i in range(len(vars)):
        if var_types[i] == 'i':
            iters.append(iter(iutils.iter_all_instances(vars[i])))
        elif var_types[i] == 'r':
            iters.append(iter((vars[i],)))
        else:
            raise ValueError('Unknown var type', var_types[i])

    it = itertools.product(*iters)
    up_bound = func(*next(it))
    low_bound = up_bound
    for i in it:
        v = func(*i)
        up_bound = np.maximum(up_bound, v)
        low_bound = np.minimum(low_bound, v)
    return iutils.to_iarray(low_bound, up_bound)


class LinearSystemSolver:
    @staticmethod
    def bruteforce(matrix, vec, types='ir'):
        return bruteforce_solver(
            lambda m, v: la.solve(m, v),
            [matrix, vec], types)

    def gauss(miatrix, vec):
        matrix = copy.deepcopy(m)
        vec = copy.deepcopy(vec)
        n = matrix.shape[0]
        for j in range(n - 1):
            for i in range(j + 1, n):
                r = matrix[i, j] / matrix[j, j]
                for k in range(j + 1, n):
                    matrix[i, k] -= r * matrix[j, k]
                vec[i] -= r * vec[i]
        x = iutils.rand_iarray(n)
        for i in range(n - 1, -1, -1):
            s = 0
            for j in range(i + 1, n):
                s += matrix[i, j] * x[j]
            x[i] = vec[i] - s
            x[i] /= matrix[i, i]
        return x

    def jacobi(matrix, vec, x0):
        n = matrix.shape[0]
        m = copy.deepcopy(matrix)
        v = copy.deepcopy(vec)
        for i in range(n):
            for j in range(n):
                if i == j:
                    m[i, i] = 0
                else:
                    m[i, j] /= matrix[i, i]
                    m[i, j] *= -1
            v[i] /= matrix[i, i]

        x_cur = copy.deepcopy(x0)
        for _ in range(10):
            x_next = np.dot(m, x_cur) + v
            print(x_cur)
            print(x_next)
            print('=======')
            for i in range(n):
                x_cur[i] &= x_next[i]
        return x_cur

    def gauss_seidel(matrix, vec, x_start):
        pass


class EigensSolver:
    @staticmethod
    def bruteforce_eigvals(matrix, type_='i'):
        return bruteforce_solver(
            lambda m: max(la.eigvals(m).real),
            [matrix], type_)

    def _max_eigenvalue(matrix, iter_num):
        v2, v1 = np.ones(matrix.shape[0]), None
        for _ in range(iter_num):
            v2 /= iutils.norm(v2)
            v2, v1 = np.dot(matrix, v2), v2
        return v2[0] / v1[0]

    @classmethod
    def bruteforce_iter(cls, matrix, type_='i', iter_num=100):
        return bruteforce_solver(
            lambda m: cls._max_eigenvalue(m, iter_num=iter_num),
            [matrix], type_)

    @staticmethod
    def interval_solver(imatrix):
        pass


def optimize_tt_imatrix(tt_imatrix, pow=1):
    shape = tt_imatrix.shape
    if shape[0] != shape[-1] and set(shape[1:-1]) != {2}:
        raise RuntimeError('Wrong tt imatrix shape', shape)

    cores = tt_imatrix._cores
    n = len(cores)
    assert n > 1

    upper = cores[0]
    lower = cores[0]
    for i in range(1, n - 1):
        new_lower = lower @ cores[i][:, 0, :]
        new_lower_val = tt_utils.singular_amount(new_lower)

        new_upper = upper @ cores[i][:, 0, :]
        new_upper_val = tt_utils.singular_amount(new_upper)

        for j in range(1, cores[i].shape[1]):
            m = lower @ cores[i][:, j, :]
            m_val = tt_utils.singular_amount(m)
            if m_val < new_lower_val:
                new_lower = m
                new_lower_val = m_val

            m = upper @ cores[i][:, j, :]
            m_val = tt_utils.singular_amount(m)
            if m_val > new_upper_val:
                new_upper = m
                new_upper_val = m_val

        lower = new_lower
        upper = new_upper

    lower = lower @ cores[-1]
    upper = upper @ cores[-1]

    return iutils.to_iarray(
        np.around(la.matrix_power(lower, pow), 5),
        np.around(la.matrix_power(upper, pow), 5))


if __name__ == '__main__':
    np.random.seed(0)

    a = np.array([
        [1, 0, -3],
        [1, -1, 0],
        [0, 0, 8],
    ])
    b = np.array([
        [1, 1, 0],
        [1, -1, 0],
        [5, 2, 9],
    ])
    m = iutils.to_iarray(a, b)

    mm = bruteforce_solver(lambda a: a @ a @ a, [m])
    print(mm)
    print(np.dot(np.dot(m, m), m))
    print(iutils.to_iarray(a @ a @ a, b @ b @ b))

    t = TT.from_tensor(iutils.imatrix_to_tensor(m))
    print(optimize_tt_imatrix(t, 3))

    ##########################################################

    # print(np.dot(np.dot(m, m), m))
    # print('------')
    # print(np.dot(m, np.dot(m, m)))
    # print('------')
    # print(bruteforce_solver(lambda m: m @ m @ m, [m], 'i'))

    # m = iutils.to_iarray(np.array([
    #     [[3.3, 3.3], [0, 2], [0, 2]],
    #     [[0, 2], [3.3, 3.3], [0, 2]],
    #     [[0, 2], [0, 2], [3.3, 3.3]],
    # ]))
    # b = iutils.to_iarray(np.array([
    #     [-1, 2],
    #     [-1, 2],
    #     [-1, 2],
    # ]))

    # x = LinearSystemSolver.gauss(m, b)
    # print(x)
    # print(LinearSystemSolver.jacobi(m, b, x))
    # print(LinearSystemSolver.bruteforce(m, b, 'ii'))
