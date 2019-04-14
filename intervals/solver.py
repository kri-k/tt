# *-* coding: utf-8 -*-

import copy
import itertools

import numpy as np
import numpy.linalg as la

import intervals.utils as iutils


def brute_force_solver(func, vars, var_types, default_type='i'):
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
    def brute_force(matrix, vec, types='ir'):
        return brute_force_solver(
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

    def gauss_seidel(matrix, vec):
        pass


if __name__ == '__main__':
    np.random.seed(0)

    m = iutils.to_iarray(np.array([
        [[3.3, 3.3], [0, 2], [0, 2]],
        [[0, 2], [3.3, 3.3], [0, 2]],
        [[0, 2], [0, 2], [3.3, 3.3]],
    ]))
    b = iutils.to_iarray(np.array([
        [-1, 2],
        [-1, 2],
        [-1, 2],
    ]))

    print(LinearSystemSolver.gauss(m, b))
    print(LinearSystemSolver.brute_force(m, b, 'ii'))
