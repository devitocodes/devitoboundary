"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""
import numpy as np

from devito import Substitutions
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
            elif -function.dimensions[axis] in stagger:
                return -0.5
        else:
            if function.dimensions[axis] == stagger:
                return 0.5
            elif -function.dimensions[axis] == stagger:
                return -0.5
    return 0.


def add_offset_column(functions):
    """
    Adds an extra column to contain grid offset, and initialises to zero.
    """
    # Currently hardcoded for 3D
    functions['grid_offset'] = None

    for i, row in functions.iterrows():
        grid_offsets = []
        for axis in range(3):
            grid_offsets.append(get_grid_offsets(row['function'], axis))
        functions.at[i, 'grid_offset'] = tuple(grid_offsets)


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

        grid_offset = first['grid_offset']

        # Create the axial distance function
        ax = AxialDistanceFunction(first.function, self._surface,
                                   offset=grid_offset)

        # Empty tuple for weights
        weights = ()

        for i, row in group.iterrows():
            derivative = row.derivative
            eval_offset = row.eval_offset
            weights += get_weights(ax.axial, function, derivative, bcs, offsets=eval_offset)

        return weights

    def subs(self, derivs):
        """
        Return a devito Substitutions for each specified combination of function
        and derivative. Note that the evaluation offset of the stencils returned
        is based on the staggering of the functions specified.

        Parameters
        ----------
        derivs : pandas DataFrame
            The desired combinations of function, derivative, and the offset
            at which the derivative should be taken. These should be in three
            columns of a dataframe, called 'function', 'derivative', and
            'eval_offset' respectively. Note that the offset should be relative
            to the location of the function nodes (-0.5 for backward staggered,
            0.5 for forward, and 0. for no stagger). Offset should be provided
            as a tuple of (x, y, z).
        """
        # Check that dataframe contains columns with specified names
        if 'function' not in derivs.columns:
            raise ValueError("No function column specified")
        if 'derivative' not in derivs.columns:
            raise ValueError("No derivative column specified")
        if 'eval_offset' not in derivs.columns:
            raise ValueError("No evaluation offset column specified")
        # Need to check all functions specified are in the attatched functions
        if not np.all(derivs.function.isin(self._functions.function)):
            raise ValueError("Specified functions are not attatched to boundary")

        # Add column for grid offset
        add_offset_column(derivs)

        # Add names column to allow for grouping
        name_functions(derivs)

        grouped = derivs.groupby('name')

        weights = ()

        for name, group in grouped:
            # Loop over items in each group and call a function
            func_weights = self._get_function_weights(group)
            weights += func_weights

        return Substitutions(*weights)
