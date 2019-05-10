# *-* coding: utf-8 -*-


class Interval:
    def __init__(self, left, right):
        if left > right:
            left, right = right, left
        self.l = round(left, 5)
        self.r = round(right, 5)

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

    def __radd__(self, other):
        return self.__add__(other)

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

    def __rsub__(self, other):
        return self.__sub__(other)

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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        if isinstance(other, Interval):
            a, b = other.l, other.r
        else:
            a = b = other
        if a <= 0 <= b:
            raise ZeroDivisionError('Interval containing zero', (a, b))
        i = type(self)(1.0 / b, 1.0 / a)
        self *= i
        return self

    def __truediv__(self, other):
        res = type(self)(self.l, self.r)
        res /= other
        return res

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __iand__(self, other):
        if isinstance(self, Interval):
            a = max(self.l, other.l)
            b = min(self.r, other.r)
        else:
            a = max(self.l, other)
            b = min(self.r, other)
        if a > b:
            raise ValueError('Empty Interval')
        self.l = a
        self.r = b
        return self

    def __and__(self, other):
        res = type(self)(self.l, self.r)
        res &= other
        return res

    def __rand__(self, other):
        return self.__and__(other)

    def __abs__(self):
        if self.l * self.l < 0:
            raise RuntimeError('Abs of interval containing zero')

        if self.r < 0:
            r, l = self.interval
        else:
            l, r = self.interval

        return type(self)(l, r)

    def get(self, n):
        assert n in (0, 1)
        return self.r if n else self.l

    def width(self):
        return self.r - self.l

    def norm(self):
        return abs(self.l) + abs(self.r)
