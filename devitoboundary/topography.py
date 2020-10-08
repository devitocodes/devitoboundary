"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""
import os

import numpy as np
import sympy as sp
import warnings

from devito import Function, VectorFunction, Dimension
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

        # Want to create a Stencil_Gen for each function
        # Store these in a dictionary
        self._stencils = {}
        for function in functions:  # Loop over functions
            # Check that functions have symbolic coefficients set
            if function.coefficients != 'symbolic':
                sym_err = "Function {} does not have symbolic coefficients set"
                raise ValueError(sym_err.format(function.name))

            self._stencils[function.name] = StencilGen(function.space_order,
                                                       stencil_file=self._cache)

        # Get the grid and function list
        self._get_functions(functions)

        # Calculate distances
        self._dist = DirectionalDistanceFunction(functions, infile,
                                                 toggle_normals)

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
        return self.stencils[function.name].x_b

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
        return self.stencils[function.name].u(x, deriv)

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
        self.stencils[function.name].add_bcs(bc_list)

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
        if self.stencils[f_name].bc_list is None:
            return False
        return True

    def _calculate_stencils(self, f_name, deriv):
        """
        Calculate or retrieve the set of stencils required for the specified
        derivative and function.
        """
        if not self.has_bcs(f_name):
            raise RuntimeError("Function has no boundary conditions set")
        self.stencils[f_name].all_variants(deriv, stencil_out=self._cache)
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
        # Recurring value for tidiness
        m_size = int(self._functions[0].space_order/2)

        # List to store weight functions for each function
        weights = []

        # Additional dimension for storing weights
        s_dim = Dimension(name='s')
        ncoeffs = self._functions[0].space_order+1

        wshape = self._grid.shape + (ncoeffs,)
        wdims = self._grid.dimensions + (s_dim,)

        for specification in spec:
            # Loop over each specification
            f_name, deriv = specification.split(".d")
            deriv = int(deriv)

            # Loop over every item in the dictionary
            self._calculate_stencils(f_name, deriv)

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

        """
            weights.append(Coefficient(spec[function],
                           function,
                           self._grid.dimensions[0],
                           w_x))
            weights.append(Coefficient(spec[function],
                           function,
                           self._grid.dimensions[1],
                           w_y))
            weights.append(Coefficient(spec[function],
                           function,
                           self._grid.dimensions[2],
                           w_z))

        return Substitutions(*tuple(weights))
        """
