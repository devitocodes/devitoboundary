import pytest

import numpy as np
from devitoboundary.stencils.evaluation import get_data_inc_reciprocals, \
    split_types


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

    @pytest.mark.parametrize('axis', [0])
    def test_type_splitting(self, axis):
        """
        A test to check that splitting of points into various categories
        functions as intended.
        """
        xyz = ('x', 'y', 'z')
        distances = np.full((10, 10, 10), -2, dtype=float)
        ind = [None, None, None]
        ind[axis] = np.array([1, 2, 5])
        distances[ind[0], ind[1], ind[2]] = 0.6

        data = get_data_inc_reciprocals(distances, 1, xyz[axis])

        first, last, double, paired_left, paired_right = split_types(data,
                                                                     xyz[axis],
                                                                     10)
        print(first)
        print(last)
        print(double)
        print(paired_left)
        print(paired_right)
        raise NotImplementedError("Beep boop")
