# -*- coding: utf-8 -*-
import numpy as np
import tt as ttpy

from ttlib import TT
from ttlib.utils import unravel_index


def _translate_from_ttvec(t):
    cores = ttpy.vector.to_list(t)
    cores[0] = cores[0].reshape(cores[0].shape[1:])
    cores[-1] = cores[-1].reshape(cores[-1].shape[:-1])
    return TT(cores)


def _translate_to_ttvec(t):
    cores = t._cores
    cores[0] = cores[0].reshape((1, *cores[0].shape))
    cores[-1] = cores[-1].reshape((*cores[-1].shape, 1))
    return ttpy.vector.from_list(cores)


def ttcross(func, shape, eps=1e-6, verb=True):
    init_t = ttpy.xfun(shape)
    t = ttpy.multifuncrs([init_t], func, verb=verb)
    return _translate_from_ttvec(t)


def blackbox_tensor(shape):
    def _wrapper(func):
        def __wrapper(linear_index):
            linear_index = round(float(linear_index))
            i = unravel_index(linear_index, shape, order='F')
            return func(*i)
        return np.vectorize(__wrapper)
    return _wrapper


def tensor_func(tensor):
    def f(*ids):
        return tensor[ids]
    return f
