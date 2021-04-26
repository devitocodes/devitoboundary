import pytest
import os

import numpy as np
import pandas as pd
from devitoboundary.stencils.evaluation import (get_data_inc_reciprocals,
                                                split_types, add_distance_column,
                                                get_component_weights,
                                                find_boundary_points, evaluate_stencils,
                                                get_variants, apply_grid_offset)
from devitoboundary.stencils.stencil_utils import generic_function
from devitoboundary.stencils.stencils import BoundaryConditions, get_stencils_lambda
from devitoboundary.symbolics.symbols import x_b
from devito import Eq, Grid, Function, Dimension


class TestDistances:
    """
    Tests to verify the distances used in stencil evaluation.
    """
    def test_find_boundary_points(self):
        """Test that boundary points are correctly identified"""
        # Make some fake data
        data = np.full((5, 5, 5), -1)
        data[:, :, 2] = 0.5
        x, y, z = find_boundary_points(data)
        assert np.all(z == 2)
        assert x.size == 25
        assert y.size == 25

    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('spacing', [0.1, 1, 10])
    def test_reciprocal_calculation(self, axis, spacing):
        """
        A test to check that reciprocal eta calculated are consistent.
        """
        xyz = ('x', 'y', 'z')
        left = np.full((10, 10, 10), -2*spacing, dtype=float)
        right = np.full((10, 10, 10), -2*spacing, dtype=float)

        left_index = [None, None, None]
        left_index[axis] = 4
        right_index = [None, None, None]
        right_index[axis] = 5

        left[left_index[0], left_index[1], left_index[2]] = 0.3*spacing

        right[right_index[0], right_index[1], right_index[2]] = -0.7*spacing

        # Should produce the same results
        data_l = get_data_inc_reciprocals(left, spacing, xyz[axis])
        data_r = get_data_inc_reciprocals(right, spacing, xyz[axis])

        assert(np.all(np.isclose(data_l, data_r, equal_nan=True)))

    @pytest.mark.parametrize('offset', [-0.5, 0.5])
    def test_grid_shift(self, offset):
        """
        A test to check that grid offset is applied correctly to eta values
        """
        n_pts = 11
        x = 2*np.arange(2*n_pts)
        
        if offset == 0.5:
            eta_r = np.append(np.linspace(0, 0.9, n_pts), np.full(n_pts, np.NaN))
            eta_l = np.append(np.full(n_pts, np.NaN), np.linspace(0, -0.9, n_pts))
        elif offset == -0.5:
            eta_r = np.append(np.full(n_pts, np.NaN), np.linspace(0, 0.9, n_pts))
            eta_l = np.append(np.linspace(0, -0.9, n_pts), np.full(n_pts, np.NaN))
        else:
            raise ValueError("Invalid offset")

        frame = {'x': x, 'eta_l': eta_l, 'eta_r': eta_r}

        points = pd.DataFrame(frame)
        print("Before")
        print(points)

        apply_grid_offset(points, 'x', offset)

        print("After")
        print(points)

        raise NotImplementedError


    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_type_splitting(self, axis):
        """
        A test to check that splitting of points into various categories
        functions as intended.
        """
        xyz = ('x', 'y', 'z')
        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [slice(None), slice(None), slice(None)]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        data = get_data_inc_reciprocals(distances, 1, xyz[axis])
        add_distance_column(data)

        first, last, double, paired_left, paired_right = split_types(data,
                                                                     xyz[axis],
                                                                     10)

        assert(np.all(first.index.get_level_values(xyz[axis]).to_numpy() == 1))
        assert(np.all(last.index.get_level_values(xyz[axis]).to_numpy() == 6))
        assert(np.all(double.index.get_level_values(xyz[axis]).to_numpy() == 2))
        assert(np.all(paired_left.index.get_level_values(xyz[axis]).to_numpy() == 3))
        assert(np.all(paired_right.index.get_level_values(xyz[axis]).to_numpy() == 5))


class TestStencils:
    """
    A class containing tests to check stencil evaluation.
    """
    # TODO: Need to check evaluate_stencils
    # TODO: Need to check fill_weights
    @pytest.mark.parametrize('point_type', ['first', 'last'])
    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('spacing', [0.1, 1., 10.])
    def test_evaluate_stencils_offset(self, point_type, order, spacing):
        """
        Check that offsetting the grid and boundary by the same amount results
        in identical stencils for both cases. This is checked on both sides of
        the boundary.
        """
        spec = {2*i: 0 for i in range(1+order//2)}
        bcs = BoundaryConditions(spec, order)
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils_lambda = get_stencils_lambda(2, 0, bcs, cache=cache)

        distances = np.full((10, 10, 10), -order*spacing, dtype=float)
        distances[4, :, :] = np.linspace(0.1*spacing, 0.4*spacing, 10)
        if point_type == 'first':
            data = get_data_inc_reciprocals(distances, spacing, 'x')[::2]
            add_distance_column(data)
            data.dist = -order//2
        else:
            data = get_data_inc_reciprocals(distances, spacing, 'x')[1::2]
            add_distance_column(data)
            data.dist = order//2

        offset_data = data.copy()
        offset_data.eta_l += 0.5
        offset_data.eta_r += 0.5

        left_variants = np.tile(-2*np.arange(order//2), (100, 1)) + order - 1
        right_variants = np.tile(2*np.arange(order//2), (100, 1)) + 2

        normal_stencils = evaluate_stencils(data, point_type, order//2, left_variants,
                                            right_variants, order, stencils_lambda,
                                            0.)
        offset_stencils = evaluate_stencils(offset_data, point_type, order//2, left_variants,
                                            right_variants, order, stencils_lambda,
                                            0.5)
        assert np.all(normal_stencils == offset_stencils)

    @pytest.mark.parametrize('point_type', ['first', 'last'])
    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('spacing', [0.1, 1., 10.])
    def test_get_variants_offset(self, point_type, order, spacing):
        """
        Check that offsetting the grid and boundary by the same amount results
        in identical stencils for both cases. This is checked on both sides of
        the boundary.
        """
        spec = {2*i: 0 for i in range(1+order//2)}
        bcs = BoundaryConditions(spec, order)
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils_lambda = get_stencils_lambda(2, 0, bcs, cache=cache)

        distances = np.full((10, 10, 10), -order*spacing, dtype=float)
        distances[4, :, :] = np.linspace(0.1*spacing, 0.4*spacing, 10)
        if point_type == 'first':
            data = get_data_inc_reciprocals(distances, spacing, 'x')[::2]
            add_distance_column(data)
            data.dist = -order//2
        else:
            data = get_data_inc_reciprocals(distances, spacing, 'x')[1::2]
            add_distance_column(data)
            data.dist = order//2

        grid = Grid(shape=(10, 10, 10), extent=(9*spacing, 9*spacing, 9*spacing))
        s_dim = Dimension(name='s')
        ncoeffs = order + 1

        w_shape = grid.shape + (ncoeffs,)
        w_dims = grid.dimensions + (s_dim,)

        w_normal = Function(name='w_n', dimensions=w_dims, shape=w_shape)
        w_offset = Function(name='w_o', dimensions=w_dims, shape=w_shape)

        offset_data = data.copy()
        offset_data.eta_l += 0.5
        offset_data.eta_r += 0.5

        get_variants(data, order, point_type, 'x', stencils_lambda, w_normal, 0., 0.)
        get_variants(offset_data, order, point_type, 'x', stencils_lambda, w_offset, 0.5, 0.)

        if point_type == 'first':
            assert np.all(np.isclose(w_normal.data[4], w_offset.data[4]))
        else:
            assert np.all(np.isclose(w_normal.data[5], w_offset.data[5]))

    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('deriv', [1, 2])
    def test_get_component_weights(self, axis, deriv):
        """
        Check that get_component_weights returns stencils evaluated to their correct
        values.
        """
        def evaluate_variant(stencils_lambda, left_var, right_var,
                             left_eta, right_eta):
            """Evaluate the specified stencil"""
            stencil = stencils_lambda[left_var, right_var]
            eval_stencil = np.array([stencil[i](left_eta, right_eta)
                                     for i in range(5)])
            return eval_stencil

        def check_row(data, index, stencils_lambda):
            """Check values in the specified row are as intended"""
            indices = [slice(None), slice(None), slice(None)]
            indices[axis] = index
            stencils = data[indices[0], indices[1], indices[2]]

            if index == 0:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 1, 0, 1.6)
            elif index == 1:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 3, 0, 0.6)
            elif index == 2:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 3, -0.4, 0.6)
            elif index == 3:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 0, -0.4, 0)
            elif index == 4:
                true_stencil = evaluate_variant(stencils_lambda,
                                                2, 1, -1.4, 1.6)
            elif index == 5:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 3, 0, 0.6)
            elif index == 6:
                true_stencil = evaluate_variant(stencils_lambda,
                                                4, 0, -0.4, 0)
            elif index == 7:
                true_stencil = evaluate_variant(stencils_lambda,
                                                2, 0, -1.4, 0)
            elif index >= 8:
                true_stencil = evaluate_variant(stencils_lambda,
                                                0, 0, 0, 0)

            misfit = stencils - true_stencil
            assert np.amax(np.absolute(misfit)) < 1e-6

        order = 4
        spec = {2*i: 0 for i in range(1+order//2)}
        bcs = BoundaryConditions(spec, order)

        grid = Grid(shape=(10, 10, 10), extent=(9., 9., 9.))
        function = Function(name='function', grid=grid, space_order=order)

        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [slice(None), slice(None), slice(None)]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        stencils_lambda = get_stencils_lambda(deriv, 0, bcs, cache=cache)

        w = get_component_weights(distances, axis, function, deriv, stencils_lambda, 0)

        for i in range(10):
            check_row(w.data, i, stencils_lambda)
