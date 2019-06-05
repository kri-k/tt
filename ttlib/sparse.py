# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np

from ttlib import TT


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


Item = namedtuple('Item', ['dims', 'val'])


def _form_matrix(items: Peekorator, matrix_shape, prefix_indexes):
    m = np.zeros(matrix_shape)
    while items.peek is not None:
        item = items.peek
        if item.dims[:-2] != prefix_indexes:
            break
        m[item.dims[-2:]] = item.val
        next(items)
    return m


def _tt_sparse(items: Peekorator, shape, cur_dim, prefix_indexes):
    if items.peek is None:
        return TT.zeros(*shape[cur_dim:])

    if items.peek.dims[:cur_dim] > prefix_indexes:
        return TT.zeros(*shape[cur_dim:])

    assert items.peek.dims[:cur_dim] == prefix_indexes
    if cur_dim == len(shape) - 2:
        return TT.tt_svd(_form_matrix(items, shape[cur_dim:], prefix_indexes))

    result = None
    for i in range(shape[cur_dim]):
        result = TT.stack(
            result,
            _tt_sparse(items, shape, cur_dim + 1, prefix_indexes + (i,)))

    return result


def tt_sparse(items: Peekorator, shape):
    if len(shape) < 2:
        raise ValueError('ndim should be >= 2', len(shape))

    if len(shape) == 2:
        m = np.zeros(shape)
        for item in items:
            m[item.dims] = item.val
        return TT.tt_svd(m)

    result = None
    for i in range(shape[0]):
        result = TT.stack(
            result, _tt_sparse(items, shape, 1, (i,)))
    return result


if __name__ == '__main__':
    import timeit
    from scipy import sparse
    np.random.seed(1)

    shape = (2,) * 20
    m = np.product(shape[:-1])
    sp = sparse.random(m, shape[-1], 0.0001)
    t = sp.toarray().reshape(shape)

    print('{}/{}'.format(np.count_nonzero(t), t.size))
    print(TT.tt_svd(t).size)

    items = []
    it = np.nditer(t, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            items.append(Item(it.multi_index, it[0]))
        it.iternext()

    print('====Timings===')
    print(timeit.timeit(lambda: TT.tt_svd(t), number=1))
    print(timeit.timeit(lambda: tt_sparse(Peekorator(items), shape), number=1))
