# *-* coding: utf-8 -*-
import copy
import functools
import itertools

import numpy as np
import numpy.linalg as la

import intervals.utils as iutils
from ttlib import TT
from ttlib import ttcross
import ttlib.utils as tt_utils


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


def get_iarray_instance(iarray, ids):
    res = np.ones(iarray.shape)
    i = 0
    it = np.nditer(iarray, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        val = iarray[it.multi_index]
        if ids[i]:
            val = val.right
        else:
            val = val.left
        res[it.multi_index] = val
        i += 1
        it.iternext()
    return res


_CACHE_INFO = None


def get_blackbox_linear_solver(imatrix, ivec, fixed_ids, variety_shape):
    @functools.lru_cache(None)
    def _solve(imtrx_id, ivec_id):
        matrix = get_iarray_instance(imatrix, imtrx_id)
        vector = get_iarray_instance(ivec, ivec_id)
        x = la.solve(matrix, vector)
        return x

    @ttcross.blackbox_tensor(variety_shape)
    def _f(*ids):
        ids = fixed_ids + ids
        imtrx_id = ids[:imatrix.size]
        ivec_id = ids[imatrix.size:-1]
        id = ids[-1]
        global _CACHE_INFO
        _CACHE_INFO = _solve.cache_info()
        return _solve(imtrx_id, ivec_id)[id]

    return _f


class LinearSystemSolver:
    @staticmethod
    def bruteforce(matrix, vec, types='ir'):
        return bruteforce_solver(
            lambda m, v: la.solve(m, v),
            [matrix, vec], types)

    @staticmethod
    def gauss(imatrix, ivec):
        matrix = copy.deepcopy(imatrix)
        vec = copy.deepcopy(ivec)
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

    @staticmethod
    def ttcross(imatrix, ivec):
        assert imatrix.ndim == 2
        assert ivec.ndim == 1
        shape = [2] * (imatrix.size + ivec.size)
        shape.append(ivec.size)

        f = get_blackbox_linear_solver(imatrix, ivec, (), shape)
        tt = ttcross.ttcross(f, shape)
        del f
        
        x = []
        for i in range(ivec.size):
            c = copy.deepcopy(tt._cores)
            cn = c.pop(-1)
            c[-1] = np.tensordot(c[-1], cn[..., i, None], axes=([-1], [0]))
            c[-1] = c[-1].reshape(c[-1].shape[:-1])
            x.append(iutils.optimize_interval(TT(c)))
        return np.array(x)

        # t = tt.to_tensor()
        # print('{}/{}'.format(tt.size, t.size))
        # for i in range(imatrix.shape[0]):
        #     tmp = t[..., i]


    @staticmethod
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
            # print(x_cur)
            # print(x_next)
            # print('=======')
            for i in range(n):
                x_cur[i] &= x_next[i]
        return x_cur


if __name__ == '__main__':
    np.random.seed(0)

    # a = np.array([
    #     [1, 0, -3],
    #     [1, -1, 0],
    #     [0, 0, 8],
    # ])
    # b = np.array([
    #     [1, 1, 0],
    #     [1, -1, 0],
    #     [5, 2, 9],
    # ])
    # m = iutils.to_iarray(a, b)
    # # m = iutils.rand_iarray(2, 2)

    # import tt

    # print(m)
    # t = iutils.imatrix_to_tensor(m)
    # s = t.shape
    # f = ttcross.tensor_func(t)
    # f = ttcross.blackbox_tensor(s)(f)
    # tt = ttcross.ttcross(f, s)
    # tt = ttcross._translate_to_ttvec(tt)

    # # tt = tt.rand([3, 4, 5, 4, 3], 5, 3)
    # print(type(tt), tt)
    # print(iutils.min_tens(tt, rmax=10, nswp=30))

    ##########################################################

    # print(np.dot(np.dot(m, m), m))
    # print('------')
    # print(np.dot(m, np.dot(m, m)))
    # print('------')
    # print(bruteforce_solver(lambda m: m @ m @ m, [m], 'i'))

    m = iutils.to_iarray(np.array([
        [[3.3, 3.3], [0, 2], [0, 2], [1, 2], [3, 5]],
        [[0, 2], [3.3, 3.3], [1, 2], [0, 1], [9, 11]],
        [[0, 2], [0, 2], [3.3, 3.3], [0, 1], [11, 12]],
        [[0, 2], [0, 2], [3.3, 3.3], [2, 5], [1, 11]],
        [[-10, 8], [0, 1], [-8, 10], [-4, 2], [3, 4]],
    ]))
    b = iutils.to_iarray(np.array([
        [-1, 2],
        [-1, 10],
        [-32, 2],
        [1, 2],
        [1, 15],
    ]))


    # print(LinearSystemSolver.gauss(m, b))
    print(LinearSystemSolver.bruteforce(m, b, 'ii'))
    # print(LinearSystemSolver.ttcross(m, b))
    # print(_CACHE_INFO)
