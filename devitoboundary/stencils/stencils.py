import sympy as sp
import numpy as np
import pickle

from devito.logger import warning
from devitoboundary.stencils.stencil_utils import standard_stencil
from devitoboundary.symbolics.symbols import (a, x_b, x_a, x_t, E, eta_l, eta_r)

__all__ = ['taylor', 'BoundaryConditions', 'get_ext_coeffs', 'StencilSet']

_feps = np.finfo(np.float32).eps  # Get the eps


def taylor(x, order):
    """Generate a taylor series expansion of a given order"""
    n = sp.symbols('n')
    polynomial = sp.Sum(a[n]*(x-x_b)**n, (n, 0, order)).doit()
    return polynomial


def get_taylor_order(pts_count, bcs):
    """Get the maximum taylor series order given the number of bcs and points"""
    n_skip = pts_count
    for i in range(bcs.order):
        if n_skip == 0:
            if i+1 in [*bcs.bcs]:
                pass
            else:
                return i
        elif i in [*bcs.bcs]:
            pass
        else:
            n_skip -= 1
    return bcs.order


class BoundaryConditions:
    """
    Contains information on a given set of boundary conditions.

    Parameters
    ----------
    bcs : dict
        A dict of {derivative: value}
    order : int
        The order of the finite difference discretization

    Methods
    -------
    get_taylor: Get the taylor series with appropriate coefficient modifications

    Attributes
    ----------
    bcs: The bcs contained by this object

    order: The order of the discretization these bcs are associated with

    x: The variable used for the taylor series

    """
    def __init__(self, bcs, order):
        self._bcs = bcs
        self._order = order

        self._x = sp.symbols('x')

    def __str__(self):
        return "BoundaryConditions(bcs: {}, order: {})".format(self._bcs, self._order)

    def get_taylor(self, order=None):
        """Get the taylor series with appropriate coefficient modifications"""
        if order is None:
            order = self.order
        series = taylor(self._x, order)
        for deriv, val in self._bcs.items():
            series = series.subs(a[deriv], val/np.math.factorial(deriv))
        return series

    @property
    def bcs(self):
        """The bcs contained by this object"""
        return self._bcs

    @property
    def order(self):
        """The order of the discretization these bcs are associated with"""
        return self._order

    @property
    def x(self):
        """The variable used for the taylor series"""
        return self._x


def get_ext_coeffs(bcs, cache=None):
    """
    Get the extrapolation coefficients for a set of boundary conditions

    Parameters
    ----------
    bcs : BoundaryConditions
        The boundary conditions to be applied
    cache : str
        The path to the extrapolation cache. Optional.

    Returns
    -------
    coeff_dict : dict
        The dictionary of extrapolation coefficients
    """
    if cache is None:
        return _get_ext_coeffs(bcs)
    else:
        try:
            with open(cache, 'rb') as f:
                coeff_cache = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Invalid cache location")

        if not isinstance(coeff_cache, dict):
            raise TypeError("Specified file does not contain a dictionary")

        # Unique key for this extrapolation
        key = str(bcs)

        try:
            coeff_dict = coeff_cache[key]
        except KeyError:
            warning("Extrapolation not in cache. Generating new extrapolation")
            coeff_dict = _get_ext_coeffs(bcs)

            # Add new entry to dictionary
            coeff_cache[key] = coeff_dict
            # And update the cache
            with open(cache, 'wb') as f:
                pickle.dump(coeff_cache, f)

        return coeff_dict


def _get_ext_coeffs(bcs):
    """Get the extrapolation coefficients for a set of boundary conditions"""
    valid_bcs = np.array([*bcs.bcs])
    valid_bcs = valid_bcs[valid_bcs <= bcs.order]

    n_pts = 1 + bcs.order - len(valid_bcs)  # Maximum number of interior points
    coeff_dict = {}  # Master coefficient dictionary
    for points_count in range(1, n_pts+1):
        taylor_order = get_taylor_order(points_count, bcs)
        taylor = bcs.get_taylor(order=taylor_order)
        lhs = sum([E[point]*taylor.subs(bcs.x, x_a[point]) for point in range(points_count)])
        rhs = taylor.subs(bcs.x, x_t)
        eqs = [sp.Eq(sp.expand(lhs).coeff(a[i], 1), sp.expand(rhs).coeff(a[i], 1)) for i in range(bcs.order+1)
               if sp.Eq(lhs.coeff(a[i], 1), rhs.coeff(a[i], 1)) is not sp.true]

        # Variables to solve for
        solve_vars = [E[point] for point in range(points_count)]
        coeffs = sp.solve(eqs, solve_vars)

        coeff_dict[points_count] = sp.factor(sp.simplify(coeffs))

    return coeff_dict


def build_main_subs(ext_l, ext_r, l_floor, r_floor):
    """
    Build the substitutions for each side. These substitutions
    will be applied to the extrapolation coefficients to replace
    everything except the target point.
    """
    # Left side main subs
    subs_l = [(x_a[i], ext_l[i]) for i in range(len(ext_l))]
    # Set floor on eta if necessary
    if not l_floor:
        subs_l += [(x_b, eta_l)]
    else:
        subs_l += [(x_b, -0.5)]

    # Right side main subs
    subs_r = [(x_a[i], ext_r[i]) for i in range(len(ext_r))]
    # Set floor on eta if necessary
    if not r_floor:
        subs_r += [(x_b, eta_r)]
    else:
        subs_r += [(x_b, 0.5)]

    return subs_l, subs_r


def get_target_coeffs(coeffs, out):
    """Get the coeffs for a particular target"""
    target_subs = [(x_t, out)]

    target_coeffs = {key: coeffs[key].subs(target_subs) for key in coeffs}
    return target_coeffs


def merge_stencil_dicts(short, extrapolation):
    """Merges the shortened stencil with the extrapolations"""
    return {key: short.get(key, 0) + extrapolation.get(key, 0)
            for key in set(short) | set(extrapolation)}


class StencilSet():
    """
    A set of stencils for a particular specification

    Parameters
    ----------
    deriv : int
        The derivative which the stencils should approximate
    offset : float
        The relative offset at which the stencils should evaluate
    bcs : BoundaryConditions
        The boundary conditions imposed by these stencils
    cache : str
        The path to the extrapolation coefficient cache
    cautious : bool
        If True, then stencil points within half a grid increment of the boundary
        will have their values extrapolated rather than simply being skipped in
        the extrapolation. Default is False
    """

    def __init__(self, deriv, offset, bcs, cache=None, cautious=False):
        # Set various parameters
        self._deriv = deriv
        self._offset = offset
        self._bcs = bcs
        self._cautious = cautious

        # Calculate the extrapolation coefficients
        self._ext_coeffs = get_ext_coeffs(bcs, cache=cache)

        # Determine the maximum number of extrapolation points from
        # keys in the extrapolation coefficient dictionary
        self._max_ext_points = np.amax([*self._ext_coeffs])

        # Base stencil (used in stencil determination)
        self._std_stencil = standard_stencil(self.deriv, self.order,
                                             offset=self.offset, as_dict=True)

        # Generate stencils
        self._stencils = self._get_stencils()

    @property
    def max_ext_points(self):
        """
        The maximum number of extrapolation points stencils
        in this set use.
        """
        return self._max_ext_points

    @property
    def deriv(self):
        """The derivative order these stencils approximate"""
        return self._deriv

    @property
    def offset(self):
        """
        The relative offset at which these stencils evaluate
        the derivative.
        """
        return self._offset

    @property
    def bcs(self):
        """The boundary conditions enforced by these stencils"""
        return self._bcs

    @property
    def order(self):
        """The discretization order of the stencils"""
        return self._bcs.order

    @property
    def stencils(self):
        """The dictionary of all stencil variants"""
        return self._stencils

    @property
    def is_cautious(self):
        """
        StencilSet uses cautious evaluation (points within half a grid spacing
        of the boundary get extrapolated).
        """
        return self._cautious

    @property
    def lambdaify(self):
        """
        The stencils with coefficients in the form of functions with
        arguments eta_l, eta_r.
        """
        try:
            return self._stencils_lambda
        except AttributeError:
            lambdas = {}
            for variant in self.stencils:
                var_dict = {}
                for ind in self.stencils[variant]:
                    var_dict[ind] = sp.lambdify([eta_l, eta_r], self.stencils[variant][ind])
                lambdas[variant] = var_dict
            self._stencils_lambda = lambdas
            return self._stencils_lambda

    def _get_keys(self):
        """Generate keys for the stencil dict"""
        # Generate number of variants per side
        # Note: Need to generate an extra key in cautious case
        count_l = 2*self.max_ext_points + self._cautious
        count_r = 2*self.max_ext_points + self._cautious
        if np.sign(self.offset) == 1:
            count_l += 1
        elif np.sign(self.offset) == -1:
            count_r += 1

        # Innermost valid eta values
        variants_l = -self.max_ext_points + np.arange(count_l)/2 + 0.5 - self._cautious/2
        variants_r = self.max_ext_points - np.arange(count_r)[::-1]/2 - 0.5 + self._cautious/2

        # Append a NaN for where eta is too large to care about
        variants_l = np.append(np.NaN, variants_l)
        variants_r = np.append(variants_r, np.NaN)

        # The combinations of these values
        com_l, com_r = np.meshgrid(variants_l, variants_r)

        # Flattened versions
        com_l_flat = com_l.flatten()
        com_r_flat = com_r.flatten()

        keys = np.array((com_l_flat, com_r_flat)).T
        return keys

    def _get_empty_stencil_dict(self):
        """Generate an empty stencil dict with the necessary keys"""
        keys = self._get_keys()
        key_values = [tuple(row) for row in keys]
        nones = [None for row in keys]
        empty_dict = dict(zip(key_values, nones))

        return empty_dict

    def _get_outside(self, key):
        """
        Get the points outside on the left and right sides.
        """
        eta_l, eta_r = key

        # FIXME: Will want to be modified to work with shifted stencils in due course
        base_indices = list(range(-self.order//2, 1+self.order//2))

        if not np.isnan(eta_l):
            # Create the modifier to add one if cautious is True and eta_l%1 == 0.5
            l_mod = self._cautious and abs(eta_l % 1) < _feps

            # Stencil points outside on the left
            out_l = max(0, self.order//2 + np.ceil(eta_l) + l_mod)

            points_l = base_indices[:int(out_l)]
        else:
            points_l = []

        if not np.isnan(eta_r):
            # Create the modifier to add one if cautious is True and eta_l%1 == 0.5
            r_mod = self._cautious and abs(eta_r % 1) < _feps

            # Stencil points outside on the right
            out_r = max(0, self.order//2 - np.floor(eta_r) + r_mod)

            # Need to handle out_r = -out_r = 0 for backward indexing
            if out_r != 0:
                points_r = base_indices[-int(out_r):]
            else:
                points_r = []
        else:
            points_r = []

        # Trim points which don't have coefficients
        points_l = list(set(points_l) & set(self._std_stencil.keys()))
        points_r = list(set(points_r) & set(self._std_stencil.keys()))

        return points_l, points_r

    def _get_extrapolation_points(self, key, out_l, out_r):
        """
        Get the points used for the extrapolation on left and right sides.
        """
        eta_l, eta_r = key

        # Need to find the first available points to figure out where to start
        # Eta (or max number of extrapolation points) on other side limits other bound
        if not np.isnan(eta_l):
            bound_l = int(np.floor(eta_l)+1)
        else:
            bound_l = -self.max_ext_points

        if not np.isnan(eta_r):
            bound_r = int(np.ceil(eta_r)-1)
        else:
            bound_r = self.max_ext_points

        # Bounds at the other end
        if not np.isnan(eta_r):
            other_l = min(bound_l + self.max_ext_points - 1, int(np.floor(eta_r)))
        else:
            other_l = bound_l + self.max_ext_points - 1

        if not np.isnan(eta_l):
            other_r = max(bound_r - self.max_ext_points + 1, int(np.ceil(eta_l)))
        else:
            other_r = bound_r - self.max_ext_points + 1

        base_indices = list(self._std_stencil.keys())

        # Flip a bool if false (will need to apply eta floor for stability)
        if bound_l <= other_l:
            ext_points_l = list(range(bound_l, other_l+1))
            left_floor = False
        else:
            ext_points_l = np.setdiff1d(base_indices, out_l+out_r)
            left_floor = True

        if bound_r >= other_r:
            ext_points_r = list(range(other_r, bound_r+1))
            right_floor = False
        else:
            ext_points_r = np.setdiff1d(base_indices, out_l+out_r)
            right_floor = True

        return ext_points_l, ext_points_r, left_floor, right_floor

    def _get_ext_coefficients(self, main_subs_l, main_subs_r):
        """
        Get the extrapolation coefficients given the points
        available for extrapolation.
        """
        coeffs_base_l = self._ext_coeffs[len(main_subs_l)-1].copy()
        coeffs_base_r = self._ext_coeffs[len(main_subs_r)-1].copy()

        coeffs_l = {}
        coeffs_r = {}

        # Take each coeff and apply the main substitutions
        for key in coeffs_base_l:
            coeffs_l[key] = coeffs_base_l[key].subs(main_subs_l)

        for key in coeffs_base_r:
            coeffs_r[key] = coeffs_base_r[key].subs(main_subs_r)

        return coeffs_l, coeffs_r

    def _get_extrapolations(self, out_l, out_r, ext_l, ext_r, coeffs_l, coeffs_r):
        """
        Get the extrapolations required as part of the stencil
        modification. Note that these will need adding to the
        stencil coefficients which remain in the interior region.
        """
        # Dicts containing additions to the stencil
        extrapolations_l = {}
        extrapolations_r = {}

        # Loop over target points (left)
        for target in out_l:
            target_coeffs_l = get_target_coeffs(coeffs_l, target)
            # Stencil coefficient for the target point
            # FIXME: Will want to use a shifted stencil in due course sometimes
            target_coefficient = self._std_stencil[target]

            # Loop over extrapolation points
            for i in range(len(ext_l)):
                try:  # Try to add, if there is nothing to add to, make a new entry
                    extrapolations_l[ext_l[i]] += target_coefficient*target_coeffs_l[E[i]]
                except KeyError:
                    extrapolations_l[ext_l[i]] = target_coefficient*target_coeffs_l[E[i]]

        # Loop over target points (right)
        for target in out_r:
            target_coeffs_r = get_target_coeffs(coeffs_r, target)
            # Stencil coefficient for the target point
            # FIXME: Will want to use a shifted stencil in due course sometimes
            target_coefficient = self._std_stencil[target]

            # Loop over extrapolation points
            for i in range(len(ext_r)):
                try:  # Try to add, if there is nothing to add to, make a new entry
                    extrapolations_r[ext_r[i]] += target_coefficient*target_coeffs_r[E[i]]
                except KeyError:
                    extrapolations_r[ext_r[i]] = target_coefficient*target_coeffs_r[E[i]]

        extrapolations = {key: extrapolations_l.get(key, 0) + extrapolations_r.get(key, 0)
                          for key in set(extrapolations_l) | set(extrapolations_r)}

        return extrapolations

    def _get_short_stencil(self, out_l, out_r):
        """Get the shortened version of the stencil"""
        # FIXME: Will want to use a shifted stencil in due course sometimes
        short_stencil = self._std_stencil.copy()
        to_remove = out_l + out_r
        for outside in to_remove:
            del short_stencil[outside]
        return short_stencil

    def _get_stencils(self):
        """
        Generate all possible stencil variants, given the
        specification.
        """
        # Initialise the empty dicts for stencils and their indices
        stencils = self._get_empty_stencil_dict()

        # Initial pass to remove keys where no points require extrapolation
        initial_keys = list(stencils.keys())
        print(initial_keys)
        # FIXME: Will need to modify this for the case where the last point is
        # on the interior, but still needs extrapolation for stability

        for key in initial_keys:
            if self._cautious:
                if (key[0] < min(self._std_stencil.keys()) - _feps or np.isnan(key[0])) and (key[1] > max(self._std_stencil.keys()) + _feps or np.isnan(key[1])):
                    del stencils[key]
            else:
                if (key[0] - 0.5 < min(self._std_stencil.keys()) - _feps or np.isnan(key[0])) and (key[1] + 0.5 > max(self._std_stencil.keys()) + _feps or np.isnan(key[1])):
                    del stencils[key]

        # Loop over each key
        shortened_keys = list(stencils.keys())
        for key in shortened_keys:
            print(key)
            # For left and right sides
            # Need to get points to extrapolate to
            # FIXME; Will need to detect edge cases where stabilization is necessary
            # and adjust the outside points accordingly.
            # Edge case is only for pressure, so will need a switch (cautious=True)
            # Want to modify the values of out_l and out_r if cautious==True
            out_l, out_r = self._get_outside(key)
            print("out_l", out_l)
            print("out_r", out_r)

            # Need to get points to extrapolate from
            ext_l, ext_r, l_floor, r_floor = self._get_extrapolation_points(key, out_l, out_r)

            if len(ext_l) > 0 and len(ext_r) > 0:
                # Get the main substitutions for the extrapolation coefficients
                main_subs_l, main_subs_r = build_main_subs(ext_l, ext_r, l_floor, r_floor)

                # Get the extrapolation coefficients (sans target)
                coeffs_l, coeffs_r = self._get_ext_coefficients(main_subs_l, main_subs_r)

                extrapolations = self._get_extrapolations(out_l, out_r, ext_l,
                                                          ext_r, coeffs_l, coeffs_r)

                short_stencil = self._get_short_stencil(out_l, out_r)

                # Now need to merge these two halves
                stencil = merge_stencil_dicts(short_stencil, extrapolations)

                # Add the indices and coefficients to the dict
                stencils[key] = stencil

            else:
                # If all stencil points lie outside boundary, then this entry is set to be zero
                stencils[key] = {0: 0}

            print('\n')

        return stencils
