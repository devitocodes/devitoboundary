"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from devito import Function, VectorFunction, Dimension, ConditionalDimension, \
    Eq, Operator, switchconfig, Coefficient, Substitutions, Ge, Gt, Le, Lt
# from devito.symbolics import CondEq
from devitoboundary import __file__, StencilGen, DirectionalDistanceFunction

__all__ = ['ImmersedBoundary']


class ImmersedBoundary:
    """
    An immersed boundary object for implementation of surface topography in a 3D
    domain. The surface on which boundary conditions are to be imposed is
    supplied as a polygon file.

    Parameters
    ----------
    infile : str
        Path to the surface polygon file.
    functions : Devito Function, VectorFunction, or tuple thereof
        The function(s) to which the boundary is to be attached.
    toggle_normals : bool
        Swap the interior and exterior regions of the model domain. Default is
        False.
    """

    def __init__(self, infile, functions, toggle_normals=False):
        # Cache file is stencil_cache.dat
        self._cache = os.path.dirname(__file__) + '/stencil_cache.dat'

        # Get the grid and function list
        self._get_functions(functions)

        # Check all functions have the same space order
        for function in self._functions:
            if function.space_order != self._functions[0].space_order:
                ord_err = "All functions must share a space order"
                raise ValueError(ord_err)

        # Create a function map to relate function names to functions
        self.function_map = {}
        for function in self._functions:
            self.function_map[function.name] = function

        # Want to create a Stencil_Gen for each function
        # Store these in a dictionary
        self._stencils = {}
        for function in self._functions:  # Loop over functions
            # Check that functions have symbolic coefficients set
            if function.coefficients != 'symbolic':
                sym_err = "Function {} does not have symbolic coefficients set"
                raise ValueError(sym_err.format(function.name))

            self._stencils[function.name] = StencilGen(function.space_order,
                                                       stencil_file=self._cache)

        # Calculate distances
        self._dist = DirectionalDistanceFunction(functions, infile,
                                                 toggle_normals)
        # Handy shortcuts
        self._sdf = self._dist.sdf
        self._directional = self._dist.directional

    @property
    def grid(self):
        """The grid to which the boundary is attached"""
        return self._grid

    @property
    def functions(self):
        """The functions to which the boundary is attached"""
        return self._functions

    def _get_functions(self, functions):
        """
        Put supplied functions into a tuple and extract the grid, checking that
        all supplied functions are defined on the same grid. Sets the variables
        self._grid and self._functions
        """
        # Check variable type
        is_tuple = isinstance(functions, tuple)
        is_function = issubclass(type(functions), Function)
        is_vfunction = issubclass(type(functions), VectorFunction)

        if is_tuple:
            # Multiple Functions supplied
            if issubclass(type(functions[0]), VectorFunction):
                # First function is a VectorFunction
                check_grid = functions[0][0].grid
            else:
                check_grid = functions[0].grid

            for function in functions:
                # Check if the current function is a Vectorfunction
                if issubclass(type(function), VectorFunction):
                    # Need first component to get grid
                    f_grid = function[0].grid
                else:
                    f_grid = function.grid

                if f_grid is not check_grid:
                    grid_err = "Functions do not share a grid."
                    raise ValueError(grid_err)

            self._grid = check_grid  # Set boundary grid
            self._functions = functions  # Set boundary functions

        elif is_function:
            # Single Function
            self._grid = functions.grid
            # Put single functions in a tuple for consistency
            self._functions = (functions,)

        elif is_vfunction:
            # Single VectorFunction
            self._grid = functions[0].grid
            # Put single functions in a tuple for consistency
            self._functions = (functions,)

    def x_b(self, function):
        """
        The boundary position used for specifying boundary conditions.
        Shortcut for self.stencils[function.name].x_b

        Parameters
        ----------
        function : Devito function
            The function to which x_b refers
        """
        return self._stencils[function.name].x_b

    def u(self, function, x, deriv=0):
        """
        The generic function for specifying boundary conditions.
        Shortcut for self.stencils[function.name].u(x, deriv)

        Parameters
        ----------
        function : Devito function
            The function which u represents
        x : sympy symbol
            The variable used for the boundary condtion (should
            always be x_b)
        deriv : int
            The order of the derivative of the function. Default
            is zero (no derivative taken)
        """
        return self._stencils[function.name].u(x, deriv)

    def add_bcs(self, function, bc_list):
        """
        Attach boundary conditions to be imposed on the specified function on
        the boundary surface.

        Parameters
        ----------
        function : Devito function
            The function to attach the boundary conditions to.
        bc_list : list of Devito Eq objects
            The set of boundary conditions, specified in terms of 'u' and
            'x_b'
        """
        self._stencils[function.name].add_bcs(bc_list)

    def has_bcs(self, f_name):
        """
        Checks that a function attatched to the boundary has boundary
        conditions.

        Parameters
        ----------
        f_name : string
            The name of function to be checked

        Returns
        -------
        bc_bool : bool
            True if the specified function has boundary conditions
            attatched.
        """
        if self._stencils[f_name].bc_list is None:
            return False
        return True

    def _calculate_stencils(self, f_name, deriv):
        """
        Calculate or retrieve the set of stencils required for the specified
        derivative and function.
        """
        if not self.has_bcs(f_name):
            raise RuntimeError("Function has no boundary conditions set")
        self._stencils[f_name].all_variants(deriv, stencil_out=self._cache)
        # Calling this function multiple times for different derivatives
        # will overwrite stencils each time

    def subs(self, spec):
        """
        Return a Substitutions object for all stencil modifications associated
        with the boundary, given the derivatives specified.

        Parameters
        ----------
        spec : tuple
            Desired derivatives supplied as strings e.g. ('f.d2', 'g.d1') for
            second derivative of f and first derivative of g.
        """
        print('Started substitutions')
        # Recurring values for tidiness
        m_size = int(self._functions[0].space_order/2)

        x, y, z = self._grid.dimensions

        # List to store weight functions for each function
        weights = []

        # Additional dimension for storing weights
        s_dim = Dimension(name='s')
        # FIXME: should this be based off the current spec, or do I flag
        # inconsistent orders?
        ncoeffs = self._functions[0].space_order + 1

        wshape = self._grid.shape + (ncoeffs,)
        wdims = self._grid.dimensions + (s_dim,)

        for specification in spec:
            # Loop over each specification
            f_name, deriv = specification.split(".d")
            deriv = int(deriv)

            # Loop over every item in the dictionary
            print('Started stencil calculation')
            self._calculate_stencils(f_name, deriv)
            print('Finished stencil calculation')

            # Set up weight functions for this function
            w_x = Function(name=f_name+"_w_x",
                           dimensions=wdims,
                           shape=wshape)
            w_y = Function(name=f_name+"_w_y",
                           dimensions=wdims,
                           shape=wshape)
            w_z = Function(name=f_name+"_w_z",
                           dimensions=wdims,
                           shape=wshape)

            # Initialise these functions with standard stencils
            """
            print('Started base coefficients')
            std_coeffs = sp.finite_diff_weights(deriv,
                                                range(-m_size, m_size+1),
                                                0)[-1][-1]
            std_coeffs = np.array(std_coeffs)
            print('Finished base coefficients')

            print('Started base coefficient filling')
            w_x.data[:, :, :] = std_coeffs[:]
            w_y.data[:, :, :] = std_coeffs[:]
            w_z.data[:, :, :] = std_coeffs[:]
            print('Initialised function')
            """

            print("Calculating stencil weights")
            # Loop over left and right values
            for l in range(self._functions[0].space_order + 1):
                for r in range(self._functions[0].space_order + 1):
                    # Initialise empty list for eqs
                    eqs = []
                    eqs += self._get_eqs(f_name, 'x', l, r, w_x)
                    eqs += self._get_eqs(f_name, 'y', l, r, w_y)
                    eqs += self._get_eqs(f_name, 'z', l, r, w_z)
                    # Operator is run in batches as operators with large
                    # numbers of equations take some time to initialise
                    op_weights = Operator(eqs, name='Weights')
                    # op_weights.apply()
                    switchconfig(log_level='ERROR')(op_weights.apply)()
                    # DIY Progress Bar
                    print('■', end='', flush=True)
            for i in range(5):
                plt.imshow(w_x.data[:, 100, :, i])
                plt.colorbar()
                plt.show()
            print("\nWeight calculation complete.")

            weights.append(Coefficient(deriv,
                           self.function_map[f_name],
                           self._grid.dimensions[0],
                           w_x))
            weights.append(Coefficient(deriv,
                           self.function_map[f_name],
                           self._grid.dimensions[1],
                           w_y))
            weights.append(Coefficient(deriv,
                           self.function_map[f_name],
                           self._grid.dimensions[2],
                           w_z))

        return Substitutions(*tuple(weights))

    def _get_eqs(self, f_name, dim, left, right, weights):
        """
        Return a list of space_order + 1 Eq objects evaluating each weight in
        terms of eta_l and eta_r for that dimension for a given left and right
        index.

        Parameters
        ----------
        f_name : str
            The name of the function
        dim : str
            The dimension for which the stencils should be calculated
        left : int
            The left index in the stencil list
        right : int
            The right index in the stencil list
        weights : Devito Function
            The weight function for which equations should be made
        """

        s_o = self._stencils[f_name].space_order
        x, y, z, s = weights.dimensions
        h_x, h_y, h_z = self._grid.spacing

        f = self._stencils[f_name]._f
        eta_l = self._stencils[f_name]._eta_l
        eta_r = self._stencils[f_name]._eta_r
        stencil = self._stencils[f_name].stencil_list[left][right]

        # The indices of the left and right eta in the distance function
        if dim == 'x':
            l_key = 0
            r_key = 1
            spacing = h_x
        elif dim == 'y':
            l_key = 2
            r_key = 3
            spacing = h_y
        elif dim == 'z':
            l_key = 4
            r_key = 5
            spacing = h_z

        # Create a mask for where the left-right stencil variant is valid
        if right == 0:
            right_cond = Ge(self._directional[r_key]/spacing, int(s_o/2))
        else:
            rcond_lo = Ge(self._directional[r_key]/spacing, int(s_o/2)-right/2)
            rcond_hi = Lt(self._directional[r_key]/spacing, int(s_o/2)-(right-1)/2)
            right_cond = sp.And(rcond_lo, rcond_hi)

        if left == 0:
            left_cond = Le(self._directional[l_key]/spacing, -int(s_o/2))
        else:
            lcond_lo = Gt(self._directional[l_key]/spacing, (left-1)/2 - int(s_o/2))
            lcond_hi = Le(self._directional[l_key]/spacing, left/2 - int(s_o/2))
            left_cond = sp.And(lcond_lo, lcond_hi)

        cond = sp.And(left_cond, right_cond)

        mask = ConditionalDimension(name='mask', parent=z, condition=cond)

        # Create a master list of substitutions
        # This will be used to extract single weights
        subs_master = [(f[i-int(s_o/2)], 0) for i in range(s_o+1)]

        # Also need the two substitutions for eta
        subs_eta = [(eta_l, self._directional[l_key]/spacing),
                    (eta_r, self._directional[r_key]/spacing)]

        eqs = []  # Initialise empty list for eqs

        # Create M+1 equations here
        for i in range(s_o+1):
            # Substitution which will isolate a single coefficient of f
            subs_coeff = subs_master.copy()
            subs_coeff[i] = (f[i-int(s_o/2)], 1)

            eqs.append(Eq(weights[x, y, z, i],
                          stencil.subs(subs_coeff + subs_eta),
                          implicit_dims=mask))

        return eqs
