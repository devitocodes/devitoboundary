"""
A module for stencil generation given a set of boundary conditions, and a method
order.
"""
import numpy as np
import sympy as sp
import pickle
import warnings

from devito import Eq
from devitoboundary.symbolics.symbols import (x_a, u_x_a, n_max, a, x_b, x_l,
                                              x_r, x_c, f, h_x, eta_l, eta_r)
from devitoboundary.stencils.stencil_utils import (standard_stencil,
                                                   generic_function)


__all__ = ['StencilGen']


class StencilGen:
    """
    Stencil_Gen(space_order, bcs, stencil_file=None)

    Modified stencils for an immersed boundary at which a set of boundary conditions
    are to be imposed.

    Parameters
    ----------
    space_order : int
        The order of the desired spatial discretization.
    bcs : list of Sympy Eq
        The list of boundary conditions.
    stencil_file : str
        The filepath of the stencil cache.

    Attributes
    ----------
    stencil_list : list
        A nested list of possible stencil variants. Indexed by [left variant],
        [right variant]. A variant is the number of half grid increments by
        which the stencil is truncated by the boundary.
    space_order : int
        The order of the stencils.

    Methods
    -------
    all_variants(deriv)
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.
    """

    def __init__(self, s_o, bcs, stencil_file=None):
        self._s_o = s_o

        self._bcs = bcs
        self._stencil_list = None
        self._i_poly_variants = None
        self._u_poly_variants = None

        if stencil_file is None:
            self._stencil_dict = {}
        else:
            with open(stencil_file, 'rb') as f:
                stencils = pickle.load(f)
                if not isinstance(stencils, dict):
                    raise TypeError("Specified file does not contain a dictionary")
                self._stencil_dict = stencils
        self._stencil_file = stencil_file

    @property
    def bc_list(self):
        """The list of boundary conditions"""
        return self._bcs

    @property
    def stencil_list(self):
        """The list of all possible stencils"""
        return self._stencil_list

    @property
    def space_order(self):
        """The formal order of the stencils"""
        return self._s_o

    def _coeff_gen(self, n_pts, bcs=None):
        """
        Generate extrapolation polynomial coefficients given the number of
        interior points available.
        """

        def point_count(n_bcs, n_pts):
            """
            The number of points used by the polynomial, given number of bcs
            and points available.
            """
            # Points to be used can be no larger than number available
            # Points required is equal to space_order - number of bcs
            # At least one point must be used
            return min(max(self._s_o - n_bcs + 1, 1), n_pts)

        def extr_poly_order(n_bcs, n_p_used):
            """
            The order of the polynomial required given number of boundary
            conditions and points to be used.
            """
            return n_bcs + n_p_used - 1

        def reduce_order(bcs, poly_order):
            """
            Return a reduction in polynomial order, since boundary conditions
            which evaluate to zero will result in polynomial coefficients which
            are functions of one another otherwise.
            """
            eval_lhs = [bcs[i].lhs.subs(n_max, poly_order).doit() for i in range(len(bcs))]
            # FIXME: Also want to remove these boundary conditions rather than
            # just relying on the reduction in polynomial order to prevent
            # 0 = 1 type equations.
            return eval_lhs.count(0)

        def evaluate_equations(equations, poly_order):
            """
            Evaluate the sums in the equation list to the specified order.
            """
            for i in range(len(equations)):
                equations[i] = Eq(equations[i].lhs.subs(n_max, poly_order).doit(),
                                  equations[i].rhs)

        def solve_for_coeffs(equations, poly_order):
            """
            Return the coefficients of the extrapolation polynomial
            """
            solve_variables = tuple(a[i] for i in range(poly_order+1))
            return sp.solve(equations, solve_variables)

        if bcs is None:
            bcs = self._bcs
        n_bcs = len(bcs)

        n_p_used = point_count(n_bcs, n_pts)

        poly_order = extr_poly_order(n_bcs, n_p_used)
        poly_order -= reduce_order(bcs, poly_order)

        # Generate additional equations for each point used
        eq_list = [Eq(generic_function(x_a[i]), u_x_a[i])
                   for i in range(n_p_used)]

        equations = bcs + eq_list

        evaluate_equations(equations, poly_order)

        return solve_for_coeffs(equations, poly_order)

    def _poly_variants(self):
        """
        Generate all possible polynomial variants required given the order of the
        spatial discretization and a list of boundary conditions. There will be
        a single polynomial generated for the independent case, and
        one for each unified case as available points are depleted.
        """

        def double_sided_bcs(bcs):
            """Turn single sided set of bcs into double sided"""
            double_bcs = []
            for i in range(len(bcs)):
                double_bcs.append(bcs[i].subs(x_b, x_l))
                double_bcs.append(bcs[i].subs(x_b, x_r))
            return double_bcs

        def generate_double_sided():
            """
            Generate double sided polynomials based on boundary conditions
            imposed at both ends of a stencil.
            """
            ds_poly = []

            # Set up double-sided boundary conditions list
            ds_bcs = double_sided_bcs(self._bcs)

            # Unique extrapolation for each number of interior points available
            # Can't have less than one interior point
            # Maximum number of interior points required is one more than the
            # space order minus the number of boundary conditions
            n_bcs = len(self._bcs)
            for i in range(1, self._s_o - n_bcs + 1):
                ds_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1 - i,
                                                 bcs=ds_bcs)

                ds_poly.append(sum([ds_poly_coeffs[a[j]]*x_c**j
                               for j in range(len(ds_poly_coeffs))]))

            return ds_poly

        def generate_single_sided():
            """
            Generate a single-sided polynomial based on boundary conditions
            imposed.
            """
            n_bcs = len(self._bcs)
            ss_poly_coeffs = self._coeff_gen(self._s_o - n_bcs + 1)
            ss_poly = sum([ss_poly_coeffs[a[i]]*x_c**i
                           for i in range(len(ss_poly_coeffs))])

            return ss_poly

        # i -> sides are independent from one another
        self._i_poly_variants = generate_single_sided()
        # u -> sides are unified with one another
        self._u_poly_variants = generate_double_sided()

    def all_variants(self, deriv, stencil_out=None):
        """
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.

        Parameters
        ----------
        deriv : int
            The derivative for which stencils should be calculated
        stencil_out : str
            The filepath to where the stencils should be cached. This will
            default to the filepath set at initialization. If this is not done,
            the filepath supplied here will be used. If both are missing,
            then stencils will not be cached.
        """
        # FIXME: Add an offset to the arguments

        try:
            key = str(self._bcs)+str(self._s_o)+str(deriv)+'ns'
            self._stencil_list = self._stencil_dict[key]
        except KeyError:
            if stencil_out is None and self._stencil_file is None:
                no_warn = "No file specified for caching generated stencils."
                warnings.warn(no_warn)
            if stencil_out is not None and self._stencil_file is not None:
                dupe_warn = "File already specified for caching stencils." \
                    + " Defaulting to {}"
                warnings.warn(dupe_warn.format(self._stencil_file))

            warnings.warn("Generating new stencils, this may take some time.")
            self._all_variants(deriv)

            if self._stencil_file is not None:
                with open(self._stencil_file, 'wb') as f:
                    pickle.dump(self._stencil_dict, f)
            elif stencil_out is not None:
                with open(stencil_out, 'wb') as f:
                    pickle.dump(self._stencil_dict, f)

    def _all_variants(self, deriv):
        """
        Calculate the stencil coefficients of all possible stencil variants
        required for a given derivative.

        Parameters
        ----------
        deriv : int
            The derivative for which stencils should be calculated
        """

        def get_unusable(variant):
            """Get the number of unusable points on a side given the variant"""
            return min(variant, int(variant/2+1))

        def get_outside(variant):
            """Get the number of exterior points on a side given the variant"""
            return int(np.ceil(variant/2))

        def get_available(outside, unusable):
            """
            Get the number of points available for extrapolation given the
            number of exterior and unusable stencil points. Applicable to
            individual extrapolations, but not unified ones.
            """
            return self._s_o + 1 - unusable - outside

        def get_available_unified(left_unusable, right_unusable):
            """
            Get the number of points available for the extrapolation given the
            number of unusable points on each side. For unified polynomials
            only.
            """
            return self._s_o + 1 - right_unusable - left_unusable

        def get_points_to_use():
            """Get the number of points to use in the polynomial"""
            n_bcs = len(self._bcs)
            return self._s_o - n_bcs + 1

        def sub_x_u(expr, unavailable, points_used, side):
            """
            Replace x_a and u_x_a with grid increments from stencil center
            point and values of f at respective positions.
            """
            # Need to multiply indices etc by -1 for left (negative) side
            if side == 'left':
                # FIXME: Not sure this needs to be +1?
                for i in range(points_used+1):
                    index = -1*(int(self._s_o/2)-unavailable-i)
                    substitutions = [(u_x_a[i], f[index]),
                                     (x_a[i], index*h_x)]
                    expr = expr.subs(substitutions)  # Update with new subs
            elif side == 'right':
                # FIXME: Not sure this needs to be +1?
                for i in range(points_used+1):
                    index = int(self._s_o/2)-unavailable-i
                    substitutions = [(u_x_a[i], f[index]),
                                     (x_a[i], index*h_x)]
                    expr = expr.subs(substitutions)  # Update with new subs

            return expr

        def sub_x_u_center(expr):
            """
            Replace x_a and u_x_a with position and respective value of f for
            the center stencil point only. For polynomials where only a single
            interior point is used for extrapolation.
            """
            expr = expr.subs([(u_x_a[0], f[0]),
                              (x_a[0], 0)])

            return expr

        def sub_x_u_unified(expr, left_unusable, right_unusable):
            """
            Replace x_a and u_x_a with positions relative to center stencil
            point and respective function values. For unified polynomials.
            """
            for i in range(1 + self._s_o - right_unusable - left_unusable):
                index = left_unusable+i-int(self._s_o/2)
                substitutions = [(u_x_a[i], f[index]),
                                 (x_a[i], index*h_x)]
                expr = expr.subs(substitutions)

            return expr

        def sub_x_b(expr, side):
            """
            Replace x_b with the specified eta multiplied by the grid increment.
            """
            if side == 'left':
                eta = eta_l
            elif side == 'right':
                eta = eta_r

            return expr.subs(x_b, eta*h_x)

        def sub_x_lr(expr, left_variant, right_variant, floor=False):
            """
            Replace x_l and x_r with eta_l and eta_r multiplied by grid
            increment. Apply a floor of 0.5 if specified.
            """
            # Even variant number when "floor" engaged corresponds with case
            # where boundary is within 0.5 grid increments of stencil center.
            if right_variant % 2 == 0 and floor:
                expr = expr.subs(x_r, 0.5*h_x)
            else:
                expr = expr.subs(x_r, eta_r*h_x)
            if left_variant % 2 == 0 and floor:
                expr = expr.subs(x_l, -0.5*h_x)
            else:
                expr = expr.subs(x_l, eta_l*h_x)

            return expr

        def sub_x_b_floor(expr, variant, side):
            """
            Replace x_b with the specified eta multiplied by grid increment.
            For cases where eta < 0.5, eta is replaced with 0.5. Used for the
            order 2 edge case, where individual polynomials are used in a double
            sided stencil.
            """
            if side == 'left':
                # If variant = 2, then apply a floor
                if variant == 2:
                    expr = expr.subs(x_b, -0.5*h_x)
                else:
                    expr = expr.subs(x_b, eta_l*h_x)
            elif side == 'right':
                # If variant = 2, then apply a floor
                if variant == 2:
                    expr = expr.subs(x_b, 0.5*h_x)
                else:
                    expr = expr.subs(x_b, eta_r*h_x)

            return expr

        def sub_exterior_points(stencil, poly, exterior_points, side):
            """
            Replace exterior stencil points with polynomial extrapolations.
            """
            if side == 'left':
                # FIXME: Might be better to create a list then substitute?
                for i in range(exterior_points):
                    # Index from left
                    index = -1*(int(self._s_o/2)-i)
                    node_position = index*h_x
                    poly_substitution = poly.subs(x_c, node_position)
                    stencil = stencil.subs(f[index], poly_substitution)
            elif side == 'right':
                for i in range(exterior_points):
                    index = int(self._s_o/2)-i
                    node_position = index*h_x
                    poly_substitution = poly.subs(x_c, node_position)
                    stencil = stencil.subs(f[index], poly_substitution)

            return stencil

        def apply_individual_extrapolation(variant, stencil, side):
            """
            Return a modified version of the stencil, applying the
            extrapolation for the specified side.
            """
            poly = self._i_poly_variants

            unusable = get_unusable(variant)
            outside = get_outside(variant)
            to_use = get_points_to_use()

            # Substitute in correct values of x and u_x
            poly = sub_x_u(poly, unusable, to_use, side)

            # Also need to replace x_b with eta*h_x
            poly = sub_x_b(poly, side)

            # Replace exterior points with extrapolate values
            stencil = sub_exterior_points(stencil, poly, outside, side)

            return stencil

        def modify_individual_stencil(left_variant, right_variant, stencil):
            """
            Modify stencil for the case that individual polynomials are to be
            used.
            """
            # Right side polynomial
            if right_variant != 0:
                stencil = apply_individual_extrapolation(right_variant,
                                                         stencil, 'right')
            # Left side polynomial
            if left_variant != 0:
                stencil = apply_individual_extrapolation(left_variant,
                                                         stencil, 'left')

            return stencil

        def modify_unified_stencil(left_variant, right_variant, stencil):
            """
            Modify stencil for the case that unified polynomials are to be used.
            """
            right_o = get_outside(right_variant)
            left_o = get_outside(left_variant)
            right_u = get_unusable(right_variant)
            left_u = get_unusable(left_variant)
            available = get_available_unified(left_u, right_u)
            # Special case when points available for unified polynomial are zero (or smaller)
            if available <= 0:
                # Grab the unified polynomial for one point
                poly = self._u_poly_variants[0]

                poly = sub_x_u_center(poly)

                poly = sub_x_lr(poly, left_variant, right_variant, floor=True)

            else:
                # Grab the polynomial for that number of points
                poly = self._u_poly_variants[available - 1]

                poly = sub_x_u_unified(poly, left_u, right_u)

                poly = sub_x_lr(poly, left_variant, right_variant, floor=False)

            stencil = sub_exterior_points(stencil,
                                          poly, right_o,
                                          'right')

            stencil = sub_exterior_points(stencil,
                                          poly, left_o,
                                          'left')

            return stencil

        def modify_edge_stencil(left_variant, right_variant, stencil):
            """
            Modify the stencil for the 2nd order edge case where individual
            extrapolations are to be used.
            """
            right_o = get_outside(right_variant)
            left_o = get_outside(left_variant)

            # Right side polynomial
            r_poly = self._i_poly_variants
            # Left side polynomial
            l_poly = self._i_poly_variants

            r_poly = sub_x_u_center(r_poly)
            l_poly = sub_x_u_center(l_poly)

            r_poly = sub_x_b_floor(r_poly, right_variant, 'right')
            l_poly = sub_x_b_floor(l_poly, left_variant, 'left')

            stencil = sub_exterior_points(stencil, r_poly, right_o, 'right')
            stencil = sub_exterior_points(stencil, r_poly, left_o, 'left')

            return stencil

        def add_stencil_entry(left_variant, right_variant, base_stencil, n_bcs):
            """
            Add the stencil entry for the specified variant combination to the
            stencil list.
            """
            stencil_entry = base_stencil
            if (left_variant != 0 or right_variant != 0):
                # Points unusable on right
                right_u = get_unusable(right_variant)
                # Points outside on right
                right_o = get_outside(right_variant)
                # Points unusable on left
                left_u = get_unusable(left_variant)
                # Points outside on left
                left_o = get_outside(left_variant)
                # Available points for right poly
                a_p_right = get_available(left_o, right_u)
                # Available points for left poly
                a_p_left = get_available(right_o, left_u)

                use_separate = (a_p_right >= self._s_o - n_bcs + 1
                                and a_p_left >= self._s_o - n_bcs + 1)

                if use_separate:
                    # Use separate polynomials
                    stencil_entry = modify_individual_stencil(left_variant,
                                                              right_variant,
                                                              stencil_entry)

                elif self._s_o >= 4:
                    stencil_entry = modify_unified_stencil(left_variant,
                                                           right_variant,
                                                           stencil_entry)

                else:
                    # Order 2 edge case (use separate polynomials)
                    # For order 2, the double sided polynomial is never
                    # needed.
                    stencil_entry = modify_edge_stencil(left_variant,
                                                        right_variant,
                                                        stencil_entry)

            # Set stencil entry
            self._stencil_list[left_variant][right_variant] \
                = sp.simplify(stencil_entry)

        # FIXME: Will want an offset added in the future
        base_stencil = standard_stencil(deriv, self._s_o)

        # Get the polynomial variants
        self._poly_variants()

        # Set up empty nested list of size MxM
        self._stencil_list = [[None for i in range(self._s_o+1)]
                              for j in range(self._s_o+1)]

        # Number of boundary conditions
        n_bcs = len(self._bcs)

        # FIXME: Can this loop be performed with DASK?
        for le in range(self._s_o+1):
            # Left interval
            for ri in range(self._s_o+1):
                # Right interval
                add_stencil_entry(le, ri, base_stencil, n_bcs)

        # FIXME: Will want to use the offset when implemented
        key = str(self._bcs)+str(self._s_o)+str(deriv)+'ns'
        self._stencil_dict[key] = self._stencil_list
