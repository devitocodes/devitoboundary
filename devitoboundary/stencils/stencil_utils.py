import numpy as np
import sympy as sp
from devitoboundary.symbolics.symbols import a, n, n_max


def standard_stencil(deriv, space_order, offset=0.):
    """
    Generate a stencil expression with standard weightings. Offset can be
    applied to this stencil to evaluate at non-node positions.

    Parameters
    ----------
    deriv : int
        The derivative order for the stencil
    space_order : int
        The space order of the discretization
    offset : float
        The offset at which the derivative is to be evaluated. In grid
        increments. Default is 0.

    Returns
    -------
    stencil_expr : sympy.Add
        The stencil expression
    """
    # Want to start by calculating standard stencil expansions
    min_index = offset - space_order/2
    x_list = [i + min_index for i in range(space_order+1)]

    base_coeffs = sp.finite_diff_weights(deriv, x_list, 0)[-1][-1]

    # FIXME: Will need modifiying for arrays
    return np.array(base_coeffs)


def generic_function(val, deriv=0):
    """
    Returns specified derivative of a polynomial series. To be used in the place
    of functions for specification of boundary conditions.

    Parameters
    ----------
    val : Sympy symbol
        The variable of the function: x_b should be used.
    deriv : int
        The order of the derivative. Default is zero.
    """
    x_poly = sp.symbols('x_poly')
    polynomial = sp.Sum(a[n]*x_poly**n,
                        (n, 0, n_max))
    return sp.diff(polynomial, x_poly, deriv).subs(x_poly, val)
