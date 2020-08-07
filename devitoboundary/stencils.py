"""
A module for stencil generation given a set of boundary conditions, and a method
order.
"""
import numpy as np
import sympy as sp
from devito import Eq
# TODO: Allow for stencil lists for each derivative rather than having a single
#       global list.


class Stencil_Gen:
    """
    Stencil_Gen(space_order)

    Modified stencils for an immersed boundary at which a set of boundary conditions
    are to be imposed.

    Parameters
    ----------
    space_order : int
        The order of the desired spatial discretization.

    Attributes
    ----------
    stencil_list : list
        A nested list of possible stencil variants. Indexed by [left variant],
        [right variant]. A variant is the number of half grid increments by
        which the stencil is truncated by the boundary.
    space_order : int
        The order of the stencils.
    x_b : Sympy symbol
        The generic boundary position to be used for specifying boundary
        conditions.

    Methods
    -------
    u(val, deriv=0)
        The generic function, to be used for specifying bounday conditions.
    add_bcs(bc_list)
        Add a list of boundary conditions constructed using u and x_b. Must be
        called before all_variants() and subs().
    all_variants(deriv)
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative. Must be called before subs().
    subs(eta_l=None, eta_r=None)
        Obtain a numpy array of the stencil coefficients given values of eta_l
        and eta_r. This is the offset between the central stencil point and
        the boundary on the respective side. As such eta_l should always be
        negative and eta_r positive.
    """

    def __init__(self, s_o):
        self._s_o = s_o
        self._x = sp.IndexedBase('x')  # Arbitrary values of x
        self._u_x = sp.IndexedBase('u_x')  # Respective values of the function

        self._a = sp.IndexedBase('a')
        self._n, self._n_max = sp.symbols('n, n_max')  # Maximum polynomial order

        self._x_b, self._x_r, self._x_l = sp.symbols('x_b, x_r, x_l')
        self._x_c = sp.symbols('x_c')  # Continuous x

        self._stencil_list = None
        self._i_poly_variants = None
        self._u_poly_variants = None

    @property
    def stencil_list(self):
        """The list of all possible stencils"""
        return self._stencil_list

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o

    @property
    def x_b(self):
        """The generic boundary position"""
        return self._x_b

    def u(self, val, deriv=0):
        """
        Returns specified derivative of the extrapolations polynomial. To be used
        for specification of boundary conditions.

        Parameters
        ----------
        val : Sympy symbol
            The variable of the function. Should typically be x_b.
        deriv : int
            The order of the derivative. Default is zero.
        """
        x_poly = sp.symbols('x_poly')
        polynomial = sp.Sum(self._a[self._n]*x_poly**self._n, (self._n, 0, self._n_max))
        return sp.diff(polynomial, x_poly, deriv).subs(x_poly, val)

    def add_bcs(self, bc_list):
        """
        Add a list of boundary conditions. These conditions should be formed as
        devito Eq objects equating some u(x_b) to a given value.

        Parameters
        ----------
        bc_list : list
            The list of boundary conditions.
        """
        self._bcs = bc_list

    def _coeff_gen(self, n_pts, bcs=None):
        """Generate the polynomial coefficients for the specification"""
        if bcs is None:
            bcs = self._bcs
        n_bcs = len(bcs)
        n_p_used = min(max(self._s_o - n_bcs + 1, 1), n_pts)  # Number of points used for the polynomial

        poly_order = n_bcs + n_p_used - 1

        # Generate additional equations for each point used
        eq_list = [Eq(self.u(self._x[i]), self._u_x[i]) for i in range(n_p_used)]

        short_bcs = bcs.copy()
        main_bcs = [None for i in range(len(short_bcs))]
        for i in range(len(bcs)):
            main_bcs[i] = Eq(bcs[i].lhs.subs(self._n_max, poly_order).doit(), bcs[i].rhs)
        poly_order -= main_bcs.count(Eq(0, 0))  # Truncate illegible bcs
        equations = bcs + eq_list

        for i in range(len(equations)):
            equations[i] = Eq(equations[i].lhs.subs(self._n_max, poly_order).doit(), equations[i].rhs)

        solve_variables = tuple(self._a[i] for i in range(poly_order+1))

        return sp.solve(equations, solve_variables)

    def _poly_variants(self):
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
            ds_poly_i = 0
            for j in range(len(ds_poly_coeffs)):
                ds_poly_i += ds_poly_coeffs[self._a[j]]*self._x_c**j
            ds_poly.append(ds_poly_i)

        ss_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1)
        ss_poly = 0
        for i in range(len(ss_poly_coeffs)):
            ss_poly += ss_poly_coeffs[self._a[i]]*self._x_c**i
        self._i_poly_variants = ss_poly
        self._u_poly_variants = ds_poly

    def all_variants(self, deriv):
        """
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.

        Parameters
        ----------
        deriv : int
            The derivative for which stencils should be calculated
        """
        self._f = sp.IndexedBase('f')
        self._h_x = sp.symbols('h_x')
        self._eta_l, self._eta_r = sp.symbols('eta_l, eta_r')

        # Want to check the json here for the relevent nested list

        # Want to start by calculating standard stencil expansions
        base_coeffs = sp.finite_diff_weights(deriv,
                                             range(-int(self._s_o/2), int(self._s_o/2)+1),
                                             0)[-1][-1]
        base_stencil = 0
        for i in range(len(base_coeffs)):
            base_stencil += base_coeffs[i]*self._f[i-int(self._s_o/2)]

        # Get the polynomial variants
        self._poly_variants()

        # Set up empty nested list of size MxM
        self._stencil_list = [[None for i in range(self._s_o+1)] for j in range(self._s_o+1)]

        # Number of boundary conditions
        n_bcs = len(self._bcs)

        for le in range(self._s_o+1):
            # Left interval
            for ri in range(self._s_o+1):
                # Right interval
                # Set stencil [le, ri]
                if (le != 0 or ri != 0):
                    stencil_entry = base_stencil

                    # Points unusable on right
                    right_u = min(ri, int(ri/2+1))
                    # Points outside on right
                    right_o = int(np.ceil(ri/2))
                    # Points unusable on left
                    left_u = min(le, int(le/2+1))
                    # Points outside on left
                    left_o = int(np.ceil(le/2))
                    # Available points for right poly
                    a_p_right = self._s_o + 1 - right_u - left_o
                    # Available points for left poly
                    a_p_left = self._s_o + 1 - left_u - right_o
                    # Use a unified polynomial if less than s_o - n_bcs + 1 points available

                    # Points to use for right poly
                    u_p_right = min(self._s_o - n_bcs + 1, a_p_right)
                    # Points to use for left poly
                    u_p_left = min(self._s_o - n_bcs + 1, a_p_left)

                    if a_p_right >= self._s_o - n_bcs + 1 and a_p_left >= self._s_o - n_bcs + 1:
                        # Use separate polynomials
                        # Right side polynomial
                        if ri != 0:
                            r_poly = self._i_poly_variants

                            for n in range(u_p_right+1):
                                # Substitute in correct values of x and u_x
                                r_poly = r_poly.subs([(self._u_x[n], self._f[int(self._s_o/2)-right_u-n]),
                                                     (self._x[n], (int(self._s_o/2)-right_u-n)*self._h_x)])

                            # Also need to replace x_b with eta_r*h_x
                            r_poly = r_poly.subs(self._x_b, self._eta_r*self._h_x)

                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   r_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                        else:
                            r_poly = None

                        # Left side polynomial
                        if le != 0:
                            l_poly = self._i_poly_variants

                            for n in range(u_p_left+1):
                                # Substitute in correct values of x and u_x
                                l_poly = l_poly.subs([(self._u_x[n], self._f[n+left_u-int(self._s_o/2)]),
                                                     (self._x[n], (n+left_u-int(self._s_o/2))*self._h_x)])

                            # Also need to replace x_b with eta_l*h_x
                            l_poly = l_poly.subs(self._x_b, self._eta_l*self._h_x)

                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   l_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                        else:
                            l_poly = None

                    elif self._s_o >= 4:
                        # Available points for unified polynomial construction
                        a_p_uni = self._s_o + 1 - right_u - left_u
                        # Special case when points available for unified polynomial are zero (or smaller)
                        if a_p_uni <= 0:
                            # Grab the unified polynomial for one point
                            u_poly = self._u_poly_variants[0]
                            # Substitute u_x[0] with single available f
                            # Substitute x[0] with position*h_x
                            u_poly = u_poly.subs([(self._u_x[0], self._f[0]),
                                                  (self._x[0], 0)])

                            # If r is even, then set eta_r to 0.5*h_x
                            if ri % 2 == 0:
                                u_poly = u_poly.subs(self._x_r, 0.5*self._h_x)
                            else:
                                u_poly = u_poly.subs(self._x_r, self._eta_r*self._h_x)
                            # If l is even, then set eta_l to -0.5+*h_x
                            if le % 2 == 0:
                                u_poly = u_poly.subs(self._x_l, -0.5*self._h_x)
                            else:
                                u_poly = u_poly.subs(self._x_l, self._eta_l*self._h_x)

                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   u_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   u_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))

                        else:
                            # Grab the polynomial for that number of points
                            u_poly = self._u_poly_variants[a_p_uni - 1]
                            for n in range(1+self._s_o-right_u - left_u):
                                u_poly = u_poly.subs([(self._u_x[n], self._f[left_u+n-int(self._s_o/2)]),
                                                      (self._x[n], (left_u+n-int(self._s_o/2))*self._h_x)])
                            u_poly = u_poly.subs(self._x_r, self._eta_r*self._h_x)
                            u_poly = u_poly.subs(self._x_l, self._eta_l*self._h_x)
                            for n in range(right_o):
                                stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                                   u_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))
                            for n in range(left_o):
                                stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                                   u_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                    else:
                        # Order 2 edge case (use separate polynomials)
                        # For order 2, the double sided polynomial is never needed.
                        # Right side polynomial
                        r_poly = self._i_poly_variants

                        # Substitute in correct values of x and u_x
                        r_poly = r_poly.subs([(self._u_x[0], self._f[0]),
                                              (self._x[0], 0)])

                        # If ri is 2, then set eta_r to 0.5*h_x
                        if ri == 2:
                            r_poly = r_poly.subs(self._x_b, 0.5*self._h_x)
                        else:
                            r_poly = r_poly.subs(self._x_b, self._eta_r*self._h_x)

                        for n in range(right_o):
                            stencil_entry = stencil_entry.subs(self._f[int(self._s_o/2)-n],
                                                               r_poly.subs(self._x_c, (int(self._s_o/2)-n)*self._h_x))

                        # Left side polynomial
                        l_poly = self._i_poly_variants

                        # Substitute in correct values of x and u_x
                        l_poly = l_poly.subs([(self._u_x[0], self._f[0]),
                                              (self._x[0], 0)])

                        # If le is 2, then set eta_r to 0.5*h_x
                        if le == 2:
                            l_poly = l_poly.subs(self._x_b, -0.5*self._h_x)
                        else:
                            l_poly = l_poly.subs(self._x_b, self._eta_l*self._h_x)

                        for n in range(left_o):
                            stencil_entry = stencil_entry.subs(self._f[n-int(self._s_o/2)],
                                                               l_poly.subs(self._x_c, (n-int(self._s_o/2))*self._h_x))
                else:
                    stencil_entry = base_stencil
                self._stencil_list[le][ri] = sp.simplify(stencil_entry)

    def subs(self, eta_l=None, eta_r=None):
        """
        Obtain a numpy array of the stencil coefficients given values of eta_l
        and eta_r. This is the offset between the central stencil point and
        the boundary on the respective side. As such eta_l should always be
        negative and eta_r positive.

        Parameters
        ----------
        eta_l : float
            The offset between the center of the stencil and the left side
            boundary, measured in grid increments. Should always be negative.
            Default is None.
        eta_r : float
            The offset between the center of the stencil and the right side
            boundary, measured in grid increments. Should always be positive.
            Default is None

        Returns
        -------
        coeffs : ndarray
            An array of the stencil coefficients for these eta values.
        """

        if eta_l is None or abs(eta_l) > self._s_o/2:
            dist_l = 0
            sub_l = 0  # Placeholder to stop subs() encountering None
        else:
            if eta_l >= 0:
                raise ValueError("eta_l must be negative. Current value is %.2f" % eta_l)
            # Turn eta into the relevent le index in the stencil list
            dist_l = self._s_o - np.ceil(abs(eta_l)*2).astype(np.int) + 1
            sub_l = eta_l

        if eta_r is None or eta_r > self._s_o/2:
            dist_r = 0
            sub_r = 0  # Placeholder to stop subs() encountering None
        else:
            if eta_r <= 0:
                raise ValueError("eta_r must be positive. Current value is %.2f" % eta_r)
            # Turn eta into the relevent ri index in the stencil list
            dist_r = self._s_o - np.ceil(eta_r*2).astype(np.int) + 1
            sub_r = eta_r

        stencil = self._stencil_list[dist_l][dist_r].subs([(self._eta_l, sub_l),
                                                           (self._eta_r, sub_r)])
        coeffs = np.empty(self._s_o+1)
        for i in range(self._s_o+1):
            coeffs[i] = float(stencil.coeff(self._f[i-int(self._s_o/2)], 1))

        return coeffs
