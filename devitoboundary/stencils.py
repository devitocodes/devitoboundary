"""
A module for stencil generation given a set of boundary conditions, and a method
order.
"""
import sympy as sp


class Ext_Poly:
    """
    An extrapolation polynomial constructed from a set of boundary conditions.
    This extrapolation is as a function of x.
    """

    def __init__(self, s_o, n_pts):

        self._x = sp.IndexedBase('x')  # Arbitrary values of x
        self._u_x = sp.IndexedBase('u_x')  # Respective values of the function

        self._a = sp.IndexedBase('a')

        self._n, self._n_max = sp.symbols('n, n_max')  # Maximum polynomial order

        self._s_o = s_o
        self._n_pts = n_pts

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o

    @property
    def n_pts(self):
        """The number of points available to the extrapolation"""
        return self._n_pts

    @property
    def x(self):
        """The indexed base for arbitrary values of x"""
        return self._x

    @property
    def u_x(self):
        """Corresponding function values for arbitrary values of x"""
        return self._u_x

    @property
    def a(self):
        """The polynomial coefficients for the extrapolation"""
        return self._a

    def u(self, val, deriv=0):
        """Returns specified derivative of a polynomial of a given order"""
        x_poly = sp.symbols('x_poly')
        polynomial = sp.Sum(self._a[self._n]*x_poly**self._n, (self._n, 0, self._n_max))
        return sp.diff(polynomial, x_poly, deriv).subs(x_poly, val)

    def add_bcs(self, bc_list):
        """Add list of boundary condtions using u"""
        self._bcs = bc_list

    def coeff_gen(self):
        """Generate the polynomial coefficients for the specification"""
        n_bcs = len(self._bcs)
        n_p_used = min(max(self._s_o - n_bcs + 1, 1), self._n_pts)  # Number of points used for the polynomial

        poly_order = n_bcs + n_p_used - 1

        # Generate additional equations for each point used
        eq_list = [sp.Eq(self.u(self._x[i]), self._u_x[i]) for i in range(n_p_used)]

        short_bcs = self._bcs.copy()
        main_bcs = [None for i in range(len(short_bcs))]
        for i in range(len(self._bcs)):
            main_bcs[i] = sp.Eq(self._bcs[i].lhs.subs(self._n_max, poly_order).doit(), self._bcs[i].rhs)
        poly_order -= main_bcs.count(sp.Eq(0, 0))  # Truncate illegible bcs
        equations = self._bcs + eq_list

        for i in range(len(equations)):
            equations[i] = sp.Eq(equations[i].lhs.subs(self._n_max, poly_order).doit(), equations[i].rhs)

        solve_variables = tuple(self._a[i] for i in range(poly_order+1))

        return sp.solve(equations, solve_variables)


class Ext_Variations:
    """
    All possible polynomial variants required given the order of the spatial
    discretization and a list of boundary conditions. There will be
    a single polynomial generated for the independent case, and
    one for each unified case as available points are depleted.
    """


class Stencil_Gen:
    """
    Modified stencils for an immersed boundary at which a set of boundary conditions
    are to be imposed.
    """

    def __init__(self, bc_list, s_o):
        self._bc_list = bc_list
        self._s_o = s_o

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o
