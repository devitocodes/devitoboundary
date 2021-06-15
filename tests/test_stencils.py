import pytest

import numpy as np
import sympy as sp
import pickle
import os

from devitoboundary.stencils.stencils import (taylor, BoundaryConditions, get_ext_coeffs,
                                              StencilSet)
from devitoboundary.symbolics.symbols import x_a, x_t, x_b, E


class TestBCs:
    """Tests for the BoundaryConditions object"""
    @pytest.mark.parametrize('order, expected',
                             [(2, '(x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (4, '(x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (6, '(x - x_b)**6*a[6] + (x - x_b)**5*a[5]'
                                  ' + (x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]'),
                              (8, '(x - x_b)**8*a[8] + (x - x_b)**7*a[7]'
                                  ' + (x - x_b)**6*a[6] + (x - x_b)**5*a[5]'
                                  ' + (x - x_b)**4*a[4] + (x - x_b)**3*a[3]'
                                  ' + (x - x_b)**2*a[2] + (x - x_b)*a[1] + a[0]')])
    def test_taylor(self, order, expected):
        """Test generated Taylor series are as expected"""
        x = sp.symbols('x')
        series = taylor(x, order)
        assert str(series) == expected

    @pytest.mark.parametrize('spec, order, expected',
                             [({0: 0, 2: 0, 4: 0}, 4, '(x - x_b)**3*a[3] + (x - x_b)*a[1]'),
                              ({1: 0, 3: 0, 5: 0}, 2, '(x - x_b)**2*a[2] + a[0]'),
                              ({0: 0, 2: 0, 4: 0, 6: 0}, 6, '(x - x_b)**5*a[5] + (x - x_b)**3*a[3]'
                                                            ' + (x - x_b)*a[1]'),
                              ({1: 0, 3: 0, 5: 0}, 4, '(x - x_b)**4*a[4] + (x - x_b)**2*a[2] + a[0]'),
                              ({1: 10}, 1, '10.0*x - 10.0*x_b + a[0]')])
    def test_get_taylor(self, spec, order, expected):
        """Test Taylor series accounting for bcs are as expected"""
        bcs = BoundaryConditions(spec, order)

        series = bcs.get_taylor(order=None)

        assert str(series) == expected


class TestExtrapolations:
    """Tests for the generated extrapolations"""
    @pytest.mark.parametrize('spec, order, coeff, expected',
                             [({0: 0, 2: 0, 4: 0}, 4, 0, '(zeta-j)*j*(j-2*zeta)/((1+2*zeta)*(1+zeta))'),
                              ({0: 0, 2: 0, 4: 0}, 4, 1, '(zeta-j)*(1+j)*(1+2*zeta-j)/(zeta*(1+2*zeta))'),
                              ({1: 0, 3: 0}, 4, 0, 'j*(j-2*zeta)/(1+2*zeta)'),
                              ({1: 0, 3: 0}, 4, 1, '(1+j)*(1+2*zeta-j)/(1+2*zeta)'),
                              ({0: 0, 2: 0, 4: 0, 6: 0}, 6, 1, 'j*(j-zeta)*(j+2)*(j-2*zeta)*(j-2*zeta-2)/((1+zeta)*(2*zeta + 1)*(2*zeta+3))'),
                              ({1: 0, 3: 0, 5: 0}, 6, 1, '-j*(j+2)*(j-2*zeta-2)*(j-2*zeta)/((2*zeta+1)*(2*zeta+3))')])
    def test_get_ext_coeffs(self, spec, order, coeff, expected):
        """
        Test to check that extrapolation coefficnts are generated as expected, and
        match known results.
        """
        zeta, j = sp.symbols('zeta, j')

        # Substitutions needed
        test_subs = [(x_a[i], 1-order//2+i) for i in range(order//2)]
        test_subs += [(x_t, j), (x_b, zeta)]

        bcs = BoundaryConditions(spec, order)

        coeffs = get_ext_coeffs(bcs)[order//2]

        expected_coeff = sp.sympify(expected, locals={'zeta': zeta, 'j': j})

        generated_coeff = coeffs[E[coeff]].subs(test_subs)

        assert sp.simplify(generated_coeff - expected_coeff) == 0

    def test_coefficient_orders(self):
        """
        Test to check that the lower-order coefficients generated are consistent
        """
        for i in range(2, 5):
            spec = {2*j: 0 for j in range(i)}
            bcs_ref = BoundaryConditions(spec, 2*i-2)
            bcs_main = BoundaryConditions(spec, 2*i)

            coeffs_ref = get_ext_coeffs(bcs_ref)[i-1]
            coeffs_main = get_ext_coeffs(bcs_main)[i-1]

            assert coeffs_ref == coeffs_main

    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('type', ['even', 'odd'])
    def test_polynomial_recovery(self, order, type):
        """
        Test that polynomials of matching order are correctly recovered
        """
        if type == 'even':
            spec = {2*i: 0 for i in range(order)}
        else:
            spec = {2*i+1: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        coeffs = get_ext_coeffs(bcs)[order//2]

        poly = bcs.get_taylor(order=order-1)

        extrapolation = sum([coeffs[E[i]]*poly.subs(bcs.x, x_a[i]) for i in range(order//2)])
        exterior = poly.subs(bcs.x, x_t)

        assert sp.simplify(extrapolation - exterior) == 0

    @pytest.mark.parametrize('order', [2, 4])
    def test_caching_write(self, order):
        """Test that caching writes correctly"""
        spec = {2*i: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        # Reset the test cache
        with open('tests/test_extrapolation_cache_w.dat', 'wb') as f:
            pickle.dump({}, f)

        # Write an extrapolation
        write_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_w.dat')

        # Write an extrapolation of order+2
        high_spec = {2*i: 0 for i in range(order+2)}
        high_bcs = BoundaryConditions(high_spec, order+2)

        high_write_extrapolation = get_ext_coeffs(high_bcs, cache='tests/test_extrapolation_cache_w.dat')

        # Read both extrapolations again and check
        cached_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_w.dat')
        high_cached_extrapolation = get_ext_coeffs(high_bcs, cache='tests/test_extrapolation_cache_w.dat')

        assert write_extrapolation == cached_extrapolation
        assert high_write_extrapolation == high_cached_extrapolation

    @pytest.mark.parametrize('order', [2, 4])
    def test_caching_read(self, order):
        """Test that caching reads correctly"""

        spec = {2*i: 0 for i in range(order)}
        bcs = BoundaryConditions(spec, order)

        cached_extrapolation = get_ext_coeffs(bcs, cache='tests/test_extrapolation_cache_r.dat')
        generated_extrapolation = get_ext_coeffs(bcs)
        for npts in cached_extrapolation:
            for key in cached_extrapolation[npts]:
                diff = cached_extrapolation[npts][key] - generated_extrapolation[npts][key]
                assert sp.simplify(diff) == 0


class TestStencilSet:
    """Tests for the StencilSet class and its functions"""
    @pytest.mark.parametrize('order', [4, 6])
    @pytest.mark.parametrize('setup', [{'offset': 0, 'deriv': 2, 'bcs': 'even'},
                                       {'offset': 0.5, 'deriv': 1, 'bcs': 'even'},
                                       {'offset': -0.5, 'deriv': 1, 'bcs': 'odd'}])
    def test_get_keys(self, order, setup):
        """
        Test that the keys returned are correct
        """
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        offset = setup['offset']
        deriv = setup['deriv']

        if setup['bcs'] == 'even':
            bcs = BoundaryConditions({2*i: 0 for i in range(1+order//2)}, order)
            key_len = order+1
            min_left = 0.5-order//2
            max_right = order//2-0.5
        else:
            bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)
            key_len = order+3
            min_left = -0.5-order//2
            max_right = order//2 + 0.5

        stencils = StencilSet(deriv, offset, bcs, cache=cache)

        keys = stencils._get_keys()

        if offset == 0:
            assert keys.shape[0] == key_len**2
        else:
            assert keys.shape[0] == key_len*(key_len+1)

        left = keys[:, 0]
        right = keys[:, 1]

        max_left = 0
        min_right = 0
        if offset == -0.5:
            min_right = -0.5
        elif offset == 0.5:
            max_left = 0.5

        assert np.amax(left[~np.isnan(left)]) == max_left
        assert np.amin(left[~np.isnan(left)]) == min_left

        assert np.amax(right[~np.isnan(right)]) == max_right
        assert np.amin(right[~np.isnan(right)]) == min_right

    @pytest.mark.parametrize('setup', [{'ord': 4, 'l': -1.5, 'r': 0.5, 'el': [-2], 'er': [1]},
                                       {'ord': 4, 'l': -2.5, 'r': 0., 'el': [], 'er': [1]},
                                       {'ord': 4, 'l': -0.5, 'r': 2., 'el': [-2, -1], 'er': []},
                                       {'ord': 6, 'l': -1.5, 'r': 0.5, 'el': [-3, -2], 'er': [1, 2, 3]},
                                       {'ord': 6, 'l': -2.5, 'r': 0., 'el': [-3], 'er': [1, 2, 3]},
                                       {'ord': 6, 'l': -0.5, 'r': 2., 'el': [-3, -2, -1], 'er': [3]}])
    def test_get_outside(self, setup):
        """Check that oints outside boundary are identified correctly"""
        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'
        order = setup['ord']
        key = (setup['l'], setup['r'])

        bcs = BoundaryConditions({2*i + 1: 0 for i in range(1+order//2)}, order)

        # Quick test so only on one stencil type
        stencils = StencilSet(1, -0.5, bcs, cache=cache)

        points_l, points_r = stencils._get_outside(key)

        assert np.all(points_l == setup['el'])
        assert np.all(points_r == setup['er'])

    @pytest.mark.parametrize('setup', [{'key': (-0.5, np.NaN), 'el': [0, 1], 'er': [1, 2], 'f': False},
                                       {'key': (-1, np.NaN), 'el': [0, 1], 'er': [1, 2], 'f': False},
                                       {'key': (-1.5, np.NaN), 'el': [-1, 0], 'er': [1, 2], 'f': False},
                                       {'key': (np.NaN, 0.5), 'el': [-2, -1], 'er': [-1, 0], 'f': False},
                                       {'key': (np.NaN, 1), 'el': [-2, -1], 'er': [-1, 0], 'f': False},
                                       {'key': (np.NaN, 1.5), 'el': [-2, -1], 'er': [0, 1], 'f': False},
                                       {'key': (-0.5, 1.5), 'el': [0, 1], 'er': [0, 1], 'f': False},
                                       {'key': (-1, 1), 'el': [0, 1], 'er': [-1, 0], 'f': False},
                                       {'key': (-1.5, 0), 'el': [-1, 0], 'er': [-1], 'f': False},
                                       {'key': (-1.5, 0.5), 'el': [-1, 0], 'er': [-1, 0], 'f': False},
                                       {'key': (-1.5, 1), 'el': [-1, 0], 'er': [-1, 0], 'f': False},
                                       {'key': (-0.5, 1.5), 'el': [0, 1], 'er': [0, 1], 'f': False},
                                       {'key': (-0.5, 0.5), 'el': [0], 'er': [0], 'f': False},
                                       {'key': (-0.5, 0), 'el': [0], 'er': [0], 'f': False},
                                       {'key': (0, 0), 'el': [0], 'er': [0], 'f': True}])
    def test_get_extrapolation_points(self, setup):
        """
        Check that the points to be used for extrapolation are identified
        correctly.
        """
        key = setup['key']
        order = 4

        cache = os.path.dirname(__file__) + '/../devitoboundary/extrapolation_cache.dat'

        bcs = BoundaryConditions({2*i: 0 for i in range(1+order//2)}, order)

        stencils = StencilSet(2, 0, bcs, cache=cache)

        out_l, out_r = stencils._get_outside(key)

        ext_l, ext_r, l_floor, r_floor = stencils._get_extrapolation_points(key, out_l, out_r)

        # Should be false in this case
        if setup['f']:
            assert l_floor or r_floor
        else:
            assert not l_floor or r_floor

        assert np.all(ext_l == setup['el'])
        assert np.all(ext_r == setup['er'])

    @pytest.mark.parametrize('setup', [])
    def test_stencil_convergence(self, setup):
        """
        Check that grid refinement increases accuracy in line with discretization order.
        """
        # TODO: Finish this
        pass
