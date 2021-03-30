import pytest

import numpy as np
import sympy as sp
from devitoboundary.stencils.stencils import taylor, BoundaryConditions


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
