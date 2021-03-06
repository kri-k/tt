# *-* coding: utf-8 -*-

import numpy as np
from tt.optimize.tt_min import min_tens

from intervals import Interval
from ttlib import ttcross


def to_iarray(array, array_upper=None):
    """
    iarray is for Interval Matrix
    """
    dim_1 = array.ndim == 1
    if array_upper is not None:
        array = np.dstack((array, array_upper))
    if array.shape[-1] != 2:
        raise ValueError('Last dimension should be 2', array.shape)
    return np.apply_along_axis(
        lambda x: Interval(*x), -1, array).reshape(array.shape[int(dim_1):-1])


def rand_iarray(*shape):
    a = np.random.rand(*shape)
    b = a + np.random.rand(*shape)
    return to_iarray(a, b)


def iter_all_instances(iarray):
    def update_iarray_instance(iarray, out, cur_state, req_state):
        diff = cur_state ^ req_state
        linear_index = 0
        while diff:
            if diff & 1:
                index = np.unravel_index(linear_index, out.shape)
                out[index] = iarray[index].get(req_state & 1)
            diff >>= 1
            req_state >>= 1
            linear_index += 1

    out = np.array([i.left for i in iarray.flatten()]).reshape(iarray.shape)
    yield np.copy(out)
    for i in range(1, 1 << iarray.size):
        update_iarray_instance(iarray, out, i - 1, i)
        yield np.copy(out)


def iarray_to_tensor(iarray):
    s = [2] * iarray.size
    t = np.empty(s + list(iarray.shape))
    for id, m in enumerate(iter_all_instances(iarray)):
        t[np.unravel_index(id, s)] = m
    return t


def imatrix_to_tensor(imatrix):
    if imatrix.ndim != 2:
        raise TypeError('ndim should be equal 2', imatrix.ndim)
    s = [2] * imatrix.size
    t = np.empty([imatrix.shape[0]] + s + [imatrix.shape[1]])
    for id, m in enumerate(iter_all_instances(imatrix)):
        t[(slice(None), *np.unravel_index(id, s), slice(None))] = m
    return t


def tensor_to_imatrix(tensor):
    r, c = tensor.shape[0], tensor.shape[-1]
    m = [[None] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            t = tensor[i, ..., j]
            m[i][j] = Interval(np.min(t), np.max(t))
    return np.array(m)


def norm(array):
    if array.dtype.type is np.object_:
        # consider this case as interval array
        return sum(i.width() for i in array.flat)
    else:
        return np.linalg.norm(array, ord=1)


def optimize_interval(tt):
    ttvec = ttcross._translate_to_ttvec(tt)
    a = min_tens(ttvec)[0]
    tt *= -1
    ttvec = ttcross._translate_to_ttvec(tt)
    b = -min_tens(ttvec)[0]
    return Interval(a, b)
