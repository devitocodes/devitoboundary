import pytest

import numpy as np
import sympy as sp
from devitoboundary.stencils.stencils import taylor, BoundaryConditions, get_ext_coeffs
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
