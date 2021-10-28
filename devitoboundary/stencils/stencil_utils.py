import numpy as np
import sympy as sp
from devitoboundary.symbolics.symbols import a, n, n_max


def standard_stencil(deriv, space_order, offset=0., as_float=True, as_dict=False):
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
    as_float : bool
        Convert stencil to np.float32. Default is True.
    as_dict : bool
        Return as dictionary rather than array

    Returns
    -------
    stencil_expr : sympy.Add
        The stencil expression
    """
    # Want to start by calculating standard stencil expansions
    min_index = -offset - space_order/2
    x_list = [i + min_index for i in range(space_order+1)]

    base_coeffs = sp.finite_diff_weights(deriv, x_list, 0)[-1][-1]

    if as_float:
        coeffs = np.array(base_coeffs, dtype=np.float32)
    else:
        coeffs = np.array(base_coeffs, dtype=object)

    if as_dict:
        mask = coeffs != 0
        coeffs = coeffs[mask]
        indices = (np.array(x_list)[mask] + offset).astype(int)
        return dict(zip(indices, coeffs))
    else:
        return coeffs


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


def get_grid_offset(function, axis):
    """
    For a function, return the grid offset for a specified axis.

    Parameters
    ----------
    function : devito Function
        The function to get the offset of
    axis : int
        The axis for which offset should be recovered
    """
    if function.is_Staggered:
        stagger = function.staggered
        if isinstance(stagger, tuple):
            if function.space_dimensions[axis] in stagger:
                return 0.5
            elif -function.space_dimensions[axis] in stagger:
                return -0.5
        else:
            if function.space_dimensions[axis] == stagger:
                return 0.5
            elif -function.space_dimensions[axis] == stagger:
                return -0.5
    return 0.
