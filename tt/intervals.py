import numpy as np

from tt.core import TT


class Interval:
    def __init__(self, left, right):
        if left > right:
            left, right = right, left
        self.l = left
        self.r = right

    @property
    def interval(self):
        return (self.l, self.r)

    def __repr__(self):
        return 'I({}, {})'.format(self.l, self.r)

    def __str__(self):
        return self.__repr__()

    def __iadd__(self, other):
        if isinstance(other, Interval):
            self.l += other.l
            self.r += other.r
        else:
            self.l += other
            self.r += other
        return self

    def __add__(self, other):
        res = type(self)(self.l, self.r)
        res += other
        return res

    def __isub__(self, other):
        if isinstance(other, Interval):
            self.l -= other.r
            self.r -= other.l
        else:
            self.l -= other
            self.r -= other
        return self

    def __sub__(self, other):
        res = type(self)(self.l, self.r)
        res -= other
        return res

    def __imul__(self, other):
        if isinstance(other, Interval):
            t = sorted(i * j for i in self.interval for j in other.interval)
            self.l = t[0]
            self.r = t[-1]
        else:
            self.l *= other
            self.r *= other
            if other < 0:
                self.r, self.l = self.interval
        return self

    def __mul__(self, other):
        res = type(self)(self.l, self.r)
        res *= other
        return res

    def __itruediv__(self, other):
        i = type(self)(1.0 / other.r, 1.0 / other.l)
        self *= i
        return self

    def __truediv__(self, other):
        res = type(self)(self.l, self.r)
        res /= other
        return res

    def get(self, n):
        assert n in (0, 1)
        return self.r if n else self.l


def to_iarray(array, array_upper=None):
    """
    iarray is for Interval Matrix
    """
    if array_upper is not None:
        array = np.dstack((array, array_upper))
    if array.shape[-1] != 2:
        raise ValueError('Last dimension should be 2', array.shape)
    return np.apply_along_axis(lambda x: Interval(*x), -1, array)


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

    out = np.array([i.l for i in iarray.flatten()]).reshape(iarray.shape)
    yield out
    for i in range(1, 1 << iarray.size):
        update_iarray_instance(iarray, out, i - 1, i)
        yield out


def to_tensor(iarray):
    s = [2] * iarray.size
    t = np.empty(s + list(iarray.shape))
    for id, m in enumerate(iter_all_instances(iarray)):
        t[np.unravel_index(id, s)] = m
    return t


class LinearSystemSolver:
    @staticmethod
    def brute_force(imatrix, vec):
        up_bound_vec = np.ones(vec.shape) * -np.inf
        low_bound_vec = np.ones(vec.shape) * np.inf
        for m in iter_all_instances(imatrix):
            v = np.linalg.solve(m, vec)
            up_bound_vec = np.maximum(up_bound_vec, v)
            low_bound_vec = np.minimum(low_bound_vec, v)
        return to_iarray(low_bound_vec, up_bound_vec)


if __name__ == '__main__':
    a = np.array([
        [1, 2, 3],
        [3, 0, 7],
        [5, 4, 3],
    ])
    b = np.array([
        [10, 3, 5],
        [10, 1, 9],
        [7, 9, 5],
    ])
    v = np.array([1, 1, 1])

    # a = np.array([
    #     [1, -3],
    #     [0, 5],
    # ])
    # b = np.array([
    #     [2, -2],
    #     [10, 6],
    # ])
    # v = np.array([2, 3])

    # a = np.random.rand(4, 4)
    # b = np.random.rand(4, 4)

    m = to_iarray(a, b)
    t = to_tensor(m)
    tt = TT.from_tensor(t)

    print(t.size)
    print(tt.size)

    for c in tt._cores:
        print(c.shape)

    # print(LinearSystemSolver.brute_force(m, v))
