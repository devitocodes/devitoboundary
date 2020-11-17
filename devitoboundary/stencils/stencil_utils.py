import sympy as sp
from devitoboundary.symbolics.symbols import f


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

    return sum([base_coeffs[i]*f[i-int(space_order/2)]
                for i in range(len(base_coeffs))])
