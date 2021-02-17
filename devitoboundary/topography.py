"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from devitoboundary.distance import AxialDistanceFunction
from devitoboundary.stencils.evaluation import get_weights

__all__ = ['ImmersedBoundary']


def get_grid_offsets(function, axis):
    """
    For a function, get the grid offset and set the grid offset accordingly.

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
            if function.dimensions[axis] in stagger:
                return 0.5
        else:
            if function.dimensions[axis] == stagger:
                return 0.5
    return 0


def add_offset_columns(functions):
    """
    Add extra columns to contain the grid and evaluation offsets, and initialise
    to zero.
    """
    # Currently hardcoded for 3D
    xyz = ['x', 'y', 'z']
    for axis in range(3):
        for i, row in functions.iterrows():
            functions.loc[i, 'grid_offset_'+xyz[axis]] \
                = get_grid_offsets(row['function'], axis)
        # Calculate minimum and maximum grid offset
        min_offset = np.amin(functions['grid_offset_'+xyz[axis]])
        max_offset = np.amax(functions['grid_offset_'+xyz[axis]])

        for i, row in functions.iterrows():
            if functions.loc[i, 'grid_offset_'+xyz[axis]] == min_offset:
                functions.loc[i, 'eval_offset_'+xyz[axis]] \
                    = max_offset - min_offset
            elif functions.loc[i, 'grid_offset_'+xyz[axis]] == max_offset:
                functions.loc[i, 'eval_offset_'+xyz[axis]] \
                    = min_offset - max_offset
            else:
                raise ValueError("Multiple degrees of staggering present in"
                                 + " specified function")


def name_functions(functions):
    """
    Add an extra column to the dataframe containing the names of each function.

    Parameters
    ----------
    functions : pandas DataFrame
        The dataframe of functions
    """
    functions['name'] = None
    for i, row in functions.iterrows():
        functions.loc[i, 'name'] = row['function'].name


class ImmersedBoundary:
    """
    An object to encapsulate an immersed boundary implemented via modified
    stencil coefficients.

    Parameters
    ----------
    name : str
        The name of the boundary surface.
    surface : str
        The path to the geometry file
    functions : pandas DataFrame
        A dataframe of the functions to which the immersed boundary surface is
        to be applied. Should contain the columns 'function' and 'bcs'.

    Methods
    -------
    subs(derivs)
    """

    def __init__(self, name, surface, functions):
        self._name = name
        self._surface = surface
        # Check functions contain columns with specified names
        if 'function' not in functions.columns:
            raise ValueError("No function column specified")
        if 'bcs' not in functions.columns:
            raise ValueError("No boundary conditions column specified")
        self._functions = functions

    def _get_function_weights(self, group):
        """
        Take a group, get the function offset, generate the axial distance function.
        Then loop over the specified derivatives and get the stencil weights.

        Parameters
        ----------
        group : pandas DataFrame
            The group in the dataframe corresponding with a particular function.
        """
        # First entry used to generate axial distance function
        first = group.iloc[0]
        function = first.function
        function_mask = self._functions.function == first.function

        bcs = self._functions.loc[function_mask, 'bcs'].values[0]

        xyz = ['x', 'y', 'z']
        grid_offset = tuple([first['grid_offset_'+xyz[i]] for i in range(3)])
        eval_offset = tuple([first['eval_offset_'+xyz[i]] for i in range(3)])

        # Create the axial distance function
        ax = AxialDistanceFunction(first.function, self._surface,
                                   offset=grid_offset)

        # Empty list for weights
        weights = []

        for i, row in group.iterrows():
            derivative = row.derivative
            # Where to put these weights?
            weights.append(get_weights(ax.axial, function, derivative, bcs, offsets=eval_offset))

        weights = pd.Series(weights)
        weights.index = group.index
        return weights

    def subs(self, derivs):
        """
        Return a devito Substitutions for each specified combination of function
        and derivative. Note that the evaluation offset of the stencils returned
        is based on the staggering of the functions specified.

        Parameters
        ----------
        derivs : pandas DataFrame
            The desired combinations of function and derivative. These should be
            paired in two columns of a dataframe, called 'function' and
            'derivative' respectively.
        """
        # Check that dataframe contains columns with specified names
        if 'function' not in derivs.columns:
            raise ValueError("No function column specified")
        if 'derivative' not in derivs.columns:
            raise ValueError("No derivative column specified")
        # Need to check all functions specified are in the attatched functions
        if not np.all(derivs.function.isin(self._functions.function)):
            raise ValueError("Specified functions are not attatched to boundary")

        # Add columns for grid and evaluation offset
        add_offset_columns(derivs)

        # Add names column to allow for grouping
        name_functions(derivs)

        grouped = derivs.groupby('name')

        weights = pd.Series([], dtype=object)

        for name, group in grouped:
            # Loop over items in each group and call a function
            func_weights = self._get_function_weights(group)
            weights = weights.append(func_weights)

        derivs = derivs.join(weights.rename("substitution"))

        return derivs[['function', 'derivative', 'substitution']]
