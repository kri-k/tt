# -*- coding: utf-8 -*-
import numpy as np
import pytest

from tt import TT
from tt import utils


TESTED_SHAPES = [
    (1, 8),
    (8, 1),
    (4, 5),
    (2, 2, 2),
    (2, 2, 2, 2),
    (2, 3, 4, 5),
    (2, 4, 3, 2, 5, 3),
]


def check_equal(a, b):
    return np.allclose(a, b, rtol=1e-8, atol=1e-8)


class TestTT:
    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_naive_transformation(self, shape):
        tensor = utils.rand_tensor(*shape)
        tt_cores = TT.from_tensor(tensor)
        restored_tensor = tt_cores.to_tensor()
        assert tt_cores.shape == shape
        assert check_equal(tensor, restored_tensor)

        tensor = utils.ones_tensor(*shape)
        tt_cores = TT.from_tensor(tensor)
        restored_tensor = tt_cores.to_tensor()
        assert tt_cores.shape == shape
        assert check_equal(tensor, restored_tensor)

    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_ones(self, shape):
        tt_cores = TT.ones(*shape)
        check_equal(utils.ones_tensor(*shape), tt_cores.to_tensor())

    @pytest.mark.parametrize('multiplier',
                             [0, 0.01, 0.1, 0.5, 1, 1.8, 5, 10.3])
    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_mul(self, shape, multiplier):
        for k in (multiplier, -multiplier):
            tensor = utils.rand_tensor(*shape)
            tt_cores_1 = TT.from_tensor(tensor)
            tensor *= k
            tt_cores_2 = tt_cores_1 * k
            tt_cores_1 *= k
            assert check_equal(tensor, tt_cores_1.to_tensor())
            assert check_equal(tensor, tt_cores_2.to_tensor())

    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_sum(self, shape):
        t = [utils.rand_tensor(*shape) for _ in range(3)]
        tt = [TT.from_tensor(tensor) for tensor in t]
        assert check_equal(
            TT.sum(TT.sum(tt[0], tt[1]), tt[2]).to_tensor(),
            sum(t))

    @pytest.mark.parametrize('summand',
                             [0, 0.01, 0.1, 0.5, 1, 1.8, 5, 10.3])
    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_add(self, shape, summand):
        for k in (summand, -summand):
            tensor = utils.rand_tensor(*shape)
            tt_cores = TT.from_tensor(tensor) + k
            tensor += k
            assert check_equal(tensor, tt_cores.to_tensor())

    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_sum_elements(self, shape):
        tensor = utils.rand_tensor(*shape)
        assert check_equal(
            np.sum(tensor), TT.from_tensor(tensor).sum_elements())

    @pytest.mark.parametrize(('shape_tt', 'shape_arr'), [
        [(2, 2), (2,)], [(4, 2, 3), (3,)],
        [(2, 2), (2, 2)], [(2, 2), (2, 1)],
        [(5, 2), (2, 2)], [(5, 2), (2, 1)],
        [(2, 5), (5, 2)], [(2, 5), (5, 1)],
        [(3, 2, 5), (5, 2)], [(3, 2, 5), (5, 1)],
    ])
    def test_matmul_ndarray(self, shape_tt, shape_arr):
        t = utils.rand_tensor(*shape_tt)
        m = utils.rand_tensor(*shape_arr)

        tt = TT.from_tensor(t)
        tt @= m

        target_shape = shape_tt[:-1]
        if shape_arr[-1] > 1 and len(shape_arr) > 1:
            target_shape = target_shape + (shape_arr[-1],)
        assert tt.shape == target_shape

        t = t @ m
        if shape_arr[-1] == 1:
            t = t[..., 0]
        assert check_equal(tt.to_tensor(), t)

    @pytest.mark.parametrize(('shape_tt_1', 'shape_tt_2'), [
        [(2, 2), (2, 2)], [(2, 2), (2, 2, 2)], [(2, 2, 2), (2, 2)],
        [(2, 3, 4), (4, 3, 2)], [(3, 4, 2, 5), (5, 4, 2, 3)],
    ])
    def test_matmul_tt(self, shape_tt_1, shape_tt_2):
        t_1 = utils.rand_tensor(*shape_tt_1)
        t_2 = utils.rand_tensor(*shape_tt_2)

        tt_1 = TT.from_tensor(t_1)
        tt_2 = TT.from_tensor(t_2)
        tt_1 @= tt_2

        target_shape = shape_tt_1[:-1] + shape_tt_2[1:]
        assert tt_1.shape == target_shape

        t_1 = np.tensordot(t_1, t_2, axes=([-1], [0]))
        assert check_equal(tt_1.to_tensor(), t_1)

    @pytest.mark.parametrize('shape', TESTED_SHAPES)
    def test_stack(self, shape):
        a = utils.rand_tensor(*shape)
        b = utils.rand_tensor(*shape)
        t = TT.stack(
            TT.from_tensor(a), TT.from_tensor(b)
        ).to_tensor()
        assert check_equal(t, np.array([a, b]))
