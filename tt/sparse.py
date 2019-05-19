# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np

from tt import TT


class Peekorator:
    def __init__(self, iterable):
        self._iterator = iter(iterable)
        self._empty = False
        self.peek = None
        try:
            self.peek = next(self._iterator)
        except StopIteration:
            self._empty = True

    def __iter__(self):
        return self

    def __next__(self):
        if self._empty:
            raise StopIteration()
        tmp = self.peek
        try:
            self.peek = next(self._iterator)
        except StopIteration:
            self._empty = True
            self.peek = None
        return tmp


def _form_matrix(items: Peekorator, matrix_shape, cur_dim, cur_id):
    m = np.zeros(matrix_shape)
    end = False
    while items.peek is not None:
        item = items.peek
        if item.dims[cur_dim] != cur_id:
            break
        m[item.dims[cur_dim+1:]] = item.val
        next(items)
    return m


def _tt_sparse(items: Peekorator, shape, cur_dim, cur_id):
    if items.peek is None:
        return TT.zeros(*shape[cur_dim+1:])

    if items.peek.dims[cur_dim] != cur_id:
        return TT.zeros(*shape[cur_dim+1:])

    if cur_dim == len(shape) - 3:
        return TT.tt_svd(
            _form_matrix(items, shape[cur_dim+1:], cur_dim, cur_id))

    result = None
    for i in range(shape[cur_dim + 1]):
        result = TT.stack(
            result, _tt_sparse(items, shape, cur_dim + 1, i))

    return result


def tt_sparse(items: Peekorator, shape):
    if len(shape) < 2:
        raise ValueError('ndim should be >= 2', len(shape))

    if len(shape) == 2:
        m = np.zeros(shape)
        for item in items:
            m[item.dims[cur_dim+1:]] = item.val
        return TT.tt_svd(m)

    result = None
    for i in range(shape[0]):
        result = TT.stack(
            result, _tt_sparse(items, shape, 0, i))
    return result


Item = namedtuple('Item', ['dims', 'val'])


if __name__ == '__main__':
    import timeit
    from scipy import sparse

    shape = (2,) * 20
    m = np.product(shape[:-1])
    sp = sparse.rand(m, shape[-1], 0.0001)
    t = sp.toarray().reshape(shape)

    print('{}/{}'.format(np.count_nonzero(t), t.size))
    print(TT.tt_svd(t).size)

    items = []
    it = np.nditer(t, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            items.append(Item(it.multi_index, it[0]))
        it.iternext()
    print('len =', len(items))

    print('====Timings===')
    print(timeit.timeit(lambda: TT.tt_svd(t), number=10))
    print(timeit.timeit(lambda: tt_sparse(Peekorator(items), shape), number=10))
