"""
A module for stencil generation given a set of boundary conditions, and a method
order.
"""
import sympy as sp


class Stencil_Gen:
    """
    Modified stencils for an immersed boundary at which a set of boundary conditions
    are to be imposed.
    """

    def __init__(self, s_o):
        self._s_o = s_o
        self._x = sp.IndexedBase('x')  # Arbitrary values of x
        self._u_x = sp.IndexedBase('u_x')  # Respective values of the function

        self._a = sp.IndexedBase('a')
        self._n, self._n_max = sp.symbols('n, n_max')  # Maximum polynomial order

        self._x_b, self._x_r, self._x_l = sp.symbols('x_b, x_r, x_l')
        self._x_c = sp.symbols('x_c')  # Continuous x

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o

    @property
    def x_b(self):
        """The generic boundary position"""
        return self._x_b

    def u(self, val, deriv=0):
        """Returns specified derivative of a polynomial of a given order"""
        x_poly = sp.symbols('x_poly')
        polynomial = sp.Sum(self._a[self._n]*x_poly**self._n, (self._n, 0, self._n_max))
        return sp.diff(polynomial, x_poly, deriv).subs(x_poly, val)

    def add_bcs(self, bc_list):
        """Add list of boundary condtions using u"""
        self._bcs = bc_list

    def _coeff_gen(self, n_pts, bcs=None):
        """Generate the polynomial coefficients for the specification"""
        if bcs is None:
            bcs = self._bcs
        n_bcs = len(bcs)
        n_p_used = min(max(self._s_o - n_bcs + 1, 1), n_pts)  # Number of points used for the polynomial

        poly_order = n_bcs + n_p_used - 1

        # Generate additional equations for each point used
        eq_list = [sp.Eq(self.u(self._x[i]), self._u_x[i]) for i in range(n_p_used)]

        short_bcs = bcs.copy()
        main_bcs = [None for i in range(len(short_bcs))]
        for i in range(len(bcs)):
            main_bcs[i] = sp.Eq(bcs[i].lhs.subs(self._n_max, poly_order).doit(), bcs[i].rhs)
        poly_order -= main_bcs.count(sp.Eq(0, 0))  # Truncate illegible bcs
        equations = bcs + eq_list

        for i in range(len(equations)):
            equations[i] = sp.Eq(equations[i].lhs.subs(self._n_max, poly_order).doit(), equations[i].rhs)

        solve_variables = tuple(self._a[i] for i in range(poly_order+1))

        return sp.solve(equations, solve_variables)

    def _stencils(self):
        """
        Generate all possible polynomial variants required given the order of the
        spatial discretization and a list of boundary conditions. There will be
        a single polynomial generated for the independent case, and
        one for each unified case as available points are depleted.
        """
        n_bcs = len(self._bcs)

        # Initialise list for storing polynomials
        ds_poly = []
        # Set up ds_bc_list
        ds_bc_list = []
        for i in range(n_bcs):
            ds_bc_list.append(self._bcs[i].subs(self._x_b, self._x_l))
            ds_bc_list.append(self._bcs[i].subs(self._x_b, self._x_r))

        for i in range(1, self._s_o - n_bcs + 1):
            ds_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1 - i,
                                             bcs=ds_bc_list)
            print(ds_poly_coeffs)
            ds_poly_i = 0
            for j in range(len(ds_poly_coeffs)):
                ds_poly_i += ds_poly_coeffs[self._a[j]]*self._x_c**j
            ds_poly.append(ds_poly_i)

        ss_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1)
        ss_poly = 0
        for i in range(len(ss_poly_coeffs)):
            ss_poly += ss_poly_coeffs[self._a[i]]*self._x_c**i
        return ss_poly, ds_poly
