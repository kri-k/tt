# *-* coding: utf-8 -*-


class Interval:
    def __init__(self, left, right):
        if left > right:
            left, right = right, left
        self.left = round(left, 5)
        self.right = round(right, 5)

    @property
    def interval(self):
        return (self.left, self.right)

    def __repr__(self):
        return 'I({}, {})'.format(self.left, self.right)

    def __str__(self):
        return self.__repr__()

    def __iadd__(self, other):
        if isinstance(other, Interval):
            self.left += other.left
            self.right += other.right
        else:
            self.left += other
            self.right += other
        return self

    def __add__(self, other):
        res = type(self)(self.left, self.right)
        res += other
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        if isinstance(other, Interval):
            self.left -= other.right
            self.right -= other.left
        else:
            self.left -= other
            self.right -= other
        return self

    def __sub__(self, other):
        res = type(self)(self.left, self.right)
        res -= other
        return res

    def __rsub__(self, other):
        return self.__sub__(other)

    def __imul__(self, other):
        if isinstance(other, Interval):
            t = sorted(i * j for i in self.interval for j in other.interval)
            self.left = t[0]
            self.right = t[-1]
        else:
            self.left *= other
            self.right *= other
            if other < 0:
                self.right, self.left = self.interval
        return self

    def __mul__(self, other):
        res = type(self)(self.left, self.right)
        res *= other
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        if isinstance(other, Interval):
            a, b = other.left, other.right
        else:
            a = b = other
        if a <= 0 <= b:
            raise ZeroDivisionError('Interval containing zero', (a, b))
        i = type(self)(1.0 / b, 1.0 / a)
        self *= i
        return self

    def __truediv__(self, other):
        res = type(self)(self.left, self.right)
        res /= other
        return res

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __iand__(self, other):
        if isinstance(self, Interval):
            a = max(self.left, other.left)
            b = min(self.right, other.right)
        else:
            a = max(self.left, other)
            b = min(self.right, other)
        if a > b:
            raise ValueError('Empty Interval')
        self.left = a
        self.right = b
        return self

    def __and__(self, other):
        res = type(self)(self.left, self.right)
        res &= other
        return res

    def __rand__(self, other):
        return self.__and__(other)

    def __abs__(self):
        if self.left * self.left < 0:
            raise RuntimeError('Abs of interval containing zero')

        if self.right < 0:
            right, left = self.interval
        else:
            left, right = self.interval

        return type(self)(left, right)

    def get(self, n):
        assert n in (0, 1)
        return self.right if n else self.left

    def width(self):
        return self.right - self.left

    def norm(self):
        return abs(self.left) + abs(self.right)
