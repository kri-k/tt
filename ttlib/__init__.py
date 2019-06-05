# -*- coding: utf-8 -*-
import copy

import numpy as np
import numpy.linalg as la

from ttlib import utils


class TT:
    def __init__(self, cores, shape=None):
        self._cores = copy.deepcopy(cores)
        if shape is None:
            shape = [c.shape[1] for c in cores]
            shape[0] = cores[0].shape[0]
        self._shape = list(shape)

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return sum(c.size for c in self._cores)

    @classmethod
    def tt_svd(cls, tensor):
        decomposition = []

        t = tensor
        while True:
            mtrx = utils.unfold_tensor(t, 1)
            u, v = utils.svd_decomposition(mtrx)
            decomposition.append(u)

            if t.ndim == 2:
                decomposition.append(v)
                break

            # Get next step tensor of size (r_i * n_{i+1}, n_{i+2}, ...)
            r = v.shape[0]
            n = t.shape[1]
            t = utils.fold_matrix(v, (r * n, *t.shape[2:]))

        for i in range(1, len(decomposition) - 1):
            # Transform (r_{i-1} * n_i, r_i) matrix
            # to (r_{i-1}, n_i, r_i) tensor
            prev_r = decomposition[i - 1].shape[-1]
            r = decomposition[i].shape[1]
            n = decomposition[i].shape[0] // prev_r
            u = utils.fold_matrix(decomposition[i], (prev_r, n, r))
            decomposition[i] = u

        return cls(decomposition, shape=tensor.shape)

    @classmethod
    def ones(cls, *shape):
        cores = [np.ones((shape[0], 1))]
        for i in range(1, len(shape) - 1):
            cores.append(np.ones((1, shape[i], 1)))
        if len(shape) > 1:
            cores.append(np.ones((1, shape[-1])))
        return cls(cores, shape)

    @classmethod
    def zeros(cls, *shape):
        cores = [np.zeros((shape[0], 1))]
        for i in range(1, len(shape) - 1):
            cores.append(np.zeros((1, shape[i], 1)))
        if len(shape) > 1:
            cores.append(np.zeros((1, shape[-1])))
        return cls(cores, shape)

    def to_tensor(self):
        cores_iterator = iter(self._cores)
        tensor = next(cores_iterator)
        for core in cores_iterator:
            # Dot product over rank indexes
            # of tensor of size (n_1, ..., n_{i-1}, r_{i-1})
            # and core of size (r_{i-1}, n_i, r_i)
            # Result is a tensor of size (n_1, ..., n_{i-1}, n_i, r_i)
            tensor = np.tensordot(tensor, core, axes=([-1], [0]))
        return tensor

    def _round(self, eps=1e-6):
        for i in range(len(self._cores) - 1, 1, -1):
            Q, R = la.qr(utils.unfold_tensor(self._cores[i], 1))
            self._cores[i] = utils.fold_matrix(
                R, (R.shape[0], *self._cores[i].shape[1:]))
            self._cores[i - 1] = np.tensordot(
                self._cores[i - 1], Q, axes=([-1], [0]))

        for i in range(len(self._cores) - 1):
            u, v = utils.svd_decomposition(
                utils.unfold_tensor(self._cores[i], self._cores[i].ndim - 1),
                eps=eps)
            self._cores[i] = utils.fold_matrix(
                u, (*self._cores[i].shape[:-1], u.shape[-1]))
            self._cores[i + 1] = np.tensordot(
                v, self._cores[i + 1], axes=([-1], [0]))

        return self

    @staticmethod
    def _union_cores(a, b):
        """
        Create from cores (a1, n, a2) and (b1, n, b2)
        new core (a1 + b1, n, a2 + b2)
        Core `a` in upper left corner, `b` in bottom right.
        Other elements are zeroes.
        """
        part1 = np.concatenate(
            (a, np.zeros(a.shape[:2] + (b.shape[2],))),
            axis=2)
        part2 = np.concatenate(
            (np.zeros(b.shape[:2] + (a.shape[2],)), b),
            axis=2)
        return np.concatenate((part1, part2))

    def __imul__(self, number):
        self._cores[int(number) % len(self._cores)] *= number
        return self

    def __mul__(self, number):
        result = type(self)(
            [np.copy(c) for c in self._cores], shape=self.shape)
        result *= number
        return result

    def __itruediv__(self, number):
        self *= 1.0 / number

    def __truediv__(self, number):
        return self * (1 / number)

    def __add__(self, number):
        return self.sum(self, self.ones(*self.shape) * number)

    def __len__(self):
        return len(self._cores)

    @classmethod
    def sum(cls, a, b):
        if a.shape != b.shape:
            raise ValueError('Shape mismatch', a.shape, b.shape)

        new_cores = [np.concatenate((a._cores[0], b._cores[0]), axis=1)]
        for i in range(1, len(a) - 1):
            new_cores.append(cls._union_cores(a._cores[i], b._cores[i]))
        if len(a) > 1:
            new_cores.append(np.concatenate((a._cores[-1], b._cores[-1])))
        return cls(new_cores, shape=a.shape)._round()

    def sum_elements(self):
        """
        Returns sum of tensor elements
        """
        it = iter(self._cores)
        s = np.sum(next(it), axis=0)
        for c in it:
            val = np.sum(c, axis=1)
            s = s @ val
        return s

    def _imatmul_tt(self, other):
        c = self._cores.pop() @ other._cores[0]
        self._cores.append(self._cores.pop() @ c)
        self._shape.pop()
        self._shape += other._shape[1:]
        for i in range(1, len(other._cores)):
            self._cores.append(np.copy(other._cores[i]))

    def _imatmul_ndarray(self, other):
        if other.ndim > 2:
            raise ValueError(
                'Matmul of TT tensor and ndarray '
                'with ndim > 2 is not supported')

        if other.ndim == 1:
            other = other[:, np.newaxis]

        c = self._cores.pop()
        c = c @ other
        if c.shape[-1] == 1:
            c = (self._cores.pop() @ c)[..., 0]
            self._shape.pop()
        else:
            self._shape[-1] = c.shape[-1]
        self._cores.append(c)

    def __imatmul__(self, other):
        if isinstance(other, type(self)):
            self._imatmul_tt(copy.deepcopy(other))
        else:
            self._imatmul_ndarray(other)
        return self

    def __matmul__(self, other):
        t = type(self)(self._cores, self.shape)
        t @= other
        return t

    @classmethod
    def stack(cls, a, b):
        """
        Stack tensor `b` in tensor `a`

        If tensors `a` and `b` have sizes (i1, i2, ..., in) then
        tensor `t = stack(a, b)` will have size (2, i1, i2, ..., in)
        where `t[0] == a` and `t[1] == b`

        If tensor `a` have size (i0, i1, i2, ..., in) and
        tensor `b` have size (i1, i2, ..., in) then
        tensor `t = stack(a, b)` will have size (i0 + 1, i1, i2, ..., in)
        where `t[i] == a[i]` for 0 <= i < i0 and `t[i0] == b`
        """
        if a is None or b is None:
            return a or b

        if a.ndim != b.ndim and a.ndim != b.ndim + 1:
            raise ValueError(
                'Number of dimensions mismatch '
                '(should be equal or n + 1 and n)',
                a.ndim, b.ndim)

        shift = int(a.ndim != b.ndim)

        if a.shape[shift:] != b.shape:
            raise ValueError('Shape mismatch', a.shape, b.shape)

        def f(core):
            if core.ndim == 2:
                return core.reshape((1, *core.shape))
            return core

        if a.ndim == b.ndim:
            new_cores = [np.eye(2)]
            new_shape = (2, *a.shape)
        else:
            new_cores = [np.pad(a._cores[0], ((0, 1), (0, 1)), 'constant')]
            new_cores[0][-1, -1] = 1.0
            new_shape = list(a._shape)
            new_shape[0] += 1

        new_cores.extend(
            cls._union_cores(f(a._cores[i + shift]), f(b._cores[i]))
            for i in range(len(b._cores) - 1)
        )
        new_cores.append(np.concatenate((a._cores[-1], b._cores[-1])))

        return cls(new_cores, new_shape)._round()
