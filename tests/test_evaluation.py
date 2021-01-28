import pytest
import os

import numpy as np
from devitoboundary.stencils.evaluation import get_data_inc_reciprocals, \
    split_types, add_distance_column, get_component_weights
from devitoboundary.stencils.stencil_utils import generic_function
from devitoboundary.stencils.stencils import StencilGen
from devitoboundary.symbolics.symbols import x_b
from devito import Eq, Grid, Function


class TestDistances:
    """
    A class containing tests to verify the distances used in stencil evaluation.
    """

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
    # FIXME: Need to check evaluate_stencils
    # FIXME: Need to check fill_weights
    # FIXME: Wants to check several grid spacings
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('deriv', [1, 2])
    def test_stencil_evaluation(self, axis, deriv):
        """
        A test to check that stencils are evaluated to their correct values.
        """
        def evaluate_variant(stencil_generator, left_var, right_var,
                             left_eta, right_eta):
            """Evaluate the specified stencil"""
            stencil = stencil_generator.stencils_lambda[left_var, right_var]
            eval_stencil = np.array([stencil[i](left_eta, right_eta)
                                     for i in range(5)])
            return eval_stencil

        def check_row(data, index, stencil_generator):
            """Check values in the specified row are as intended"""
            indices = [slice(None), slice(None), slice(None)]
            indices[axis] = index
            stencils = data[indices[0], indices[1], indices[2]]

            if index == 0:
                true_stencil = evaluate_variant(stencil_generator,
                                                0, 1, 0, 1.6)
            elif index == 1:
                true_stencil = evaluate_variant(stencil_generator,
                                                0, 3, 0, 0.6)
            elif index == 2:
                true_stencil = evaluate_variant(stencil_generator,
                                                4, 3, -0.4, 0.6)
            elif index == 3:
                true_stencil = evaluate_variant(stencil_generator,
                                                4, 0, -0.4, 0)
            elif index == 4:
                true_stencil = evaluate_variant(stencil_generator,
                                                2, 1, -1.4, 1.6)
            elif index == 5:
                true_stencil = evaluate_variant(stencil_generator,
                                                0, 3, 0, 0.6)
            elif index == 6:
                true_stencil = evaluate_variant(stencil_generator,
                                                4, 0, -0.4, 0)
            elif index == 7:
                true_stencil = evaluate_variant(stencil_generator,
                                                2, 0, -1.4, 0)
            elif index >= 8:
                true_stencil = evaluate_variant(stencil_generator,
                                                0, 0, 0, 0)

            misfit = stencils - true_stencil
            assert np.amax(np.absolute(misfit)) < 1e-6

        bcs = [Eq(generic_function(x_b, 2*i), 0)
               for i in range(3)]

        grid = Grid(shape=(10, 10, 10), extent=(9., 9., 9.))
        function = Function(name='function', grid=grid, space_order=4)

        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [slice(None), slice(None), slice(None)]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        stencil_file = os.path.dirname(__file__) + '/../devitoboundary/stencil_cache.dat'

        sten_gen = StencilGen(function.space_order, bcs,
                              stencil_file=stencil_file)
        sten_gen.all_variants(deriv, 0.)

        w = get_component_weights(distances, axis, function, deriv, sten_gen)

        for i in range(10):
            check_row(w.data, i, sten_gen)
