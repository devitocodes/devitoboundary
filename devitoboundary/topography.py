"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from sympy import finite_diff_weights, Max
from devito import Function, Dimension, Substitutions, Coefficient, TimeFunction, Eq, Operator, Grid
from devito.tools import as_tuple
from devitoboundary import PolySurface, StencilGen
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['GenericSurface', 'ImmersedBoundarySurface']


class GenericSurface():
    """
    A generic object attatched to one or more Devito Functions, used to contain
    data relevant for implementing 2.5D internal boundaries within a given 3D
    domain. A faceted surface is constructed from the topography data. This
    surface can be queried for axial distances to the boundary, simplifying
    implementation of immersed boundaries in finite difference models.

    Parameters
    ----------
    boundary_data : array_like
        Array of topography points grouped as [[x, y, z], [x, y, z], ...]
    functions : tuple of Devito Function or TimeFunction objects
        The function(s) to which the boundary is to be attached. Note that these
        must all have the same space order.
    """

    def __init__(self, boundary_data, functions):
        self._boundary_data = np.array(boundary_data)
        # Check that boundary data has correct formatting
        assert len(self._boundary_data.shape) == 2, "Boundary data incorrectly formatted"
        assert self._boundary_data.shape[1] == 3, "GenericSurface is for 3D boundaries only"
        # Assert functions supplied as tuple
        assert isinstance(functions, tuple), "Functions must be supplied as tuple"
        # Check that all the functions share a grid
        # Check that all the functions have the same space order
        for function in functions:
            assert function.grid is functions[0].grid, "All functions must share the same grid"
            assert function.space_order == functions[0].space_order, "All functions must have the same space order"
        # Want to check that this grid is 3D
        self._grid = functions[0].grid
        assert len(self._grid.dimensions) == 3, "GenericSurface is for 3D grids only"
        self._functions = functions

        self._surface = PolySurface(self._boundary_data, self._grid)

    @property
    def boundary_data(self):
        """The topography data for the boundary"""
        return self._boundary_data

    @property
    def grid(self):
        """The grid to which the boundary is attached"""
        return self._grid

    @property
    def functions(self):
        """The functions to which the boundary is attached"""
        return self._functions

    def plot_boundary(self, invert_z=True, save=False, save_path=None):
        """
        Plot the boundary surface as a triangular mesh.
        """

        fig = plt.figure()
        plot_axes = fig.add_subplot(111, projection='3d')
        plot_axes.plot_trisurf(self._boundary_data[:, 0],
                               self._boundary_data[:, 1],
                               self._boundary_data[:, 2],
                               color='aquamarine')

        plot_axes.set_xlabel("x")
        plot_axes.set_ylabel("y")
        plot_axes.set_zlabel("z")
        plot_axes.set_zlim(0, self._grid.extent[2], False)
        if invert_z:
            plot_axes.invert_zaxis()
        if save:
            if save_path is not None:
                plt.savefig(save_path)
            else:
                raise OSError("Invalid filepath.")
        plt.show()

    def query(self, q_points, index_input=False):
        """
        Query a set of points to find axial distances to the boundary surface.
        Distances are returned in grid increments.

        Parameters
        ----------
        q_points : array_like
            Array of points to query grouped as [[x, y, z], [x, y, z], ...]

        Returns
        -------
        z_dist : ndarray
            Distance to the surface in the z direction for the respective points
            in q_points. Values of NaN indicate that the surface does not
            occlude the point in this direction.
        y_pos_dist : ndarray
            Distances to the surface in the positive y direction. Same behaviours
            as z_dist
        y_neg_dist : ndarray
            Distances to the surface in the negative y direction. Same behaviours
            as z_dist
        x_pos_dist : ndarray
            Distances to the surface in the positive x direction. Same behaviours
            as z_dist
        x_neg_dist : ndarray
            Distances to the surface in the negative x direction. Same behaviours
            as z_dist
        """
        return self._surface.query(q_points, index_input)

    def fd_node_sides(self):
        """
        Check all nodes in the grid and determine if they are outside or inside the
        boundary surface.

        Returns
        -------
        positive_mask : ndarray
            A boolean mask matching the size of the grid. True where the respective
            node lies on the positive (outside) of the boundary surface.
        """
        return self._surface.fd_node_sides()


class ImmersedBoundarySurface(GenericSurface):
    """
    An immersed boundary object for implementation of surface topography in a 3D
    domain. The boundary surface is reconstructed from an appropriately-formatted
    cloud of topography measurements.

    Parameters
    ----------
    boundary_data : array_like
        Array of topography points grouped as [[x, y, z], [x, y, z], ...]
    functions : tuple of Devito Function or TimeFunction objects
        The function(s) to which the boundary is to be attached.
    stencil_file : str
        The file where a cache of stencils is stored. If none provided, then
        any stencils will be calculated from scratch. Default is None.
    """

    def __init__(self, boundary_data, functions, stencil_file=None):
        super().__init__(boundary_data, functions)
        # Want to create an appropriately named Stencil_Gen for each function
        # Store these in a dictionary
        self.stencils = {}
        for function in functions:
            if function.coefficients != 'symbolic':
                raise ValueError("Function {} does not have symbolic coefficients set".format(function.name))
            self.stencils[function.name] = StencilGen(function.space_order,
                                                      stencil_file=stencil_file)

        self._node_id()
        self._distance_calculation()

    def _node_id(self):
        """
        Identifies the nodes within the area of effect of the boundary, where
        stencils will require modification. Axial distances are calculated during
        this process.
        """

        print('Node ID started')
        self._positive_mask = self.fd_node_sides()

        m_size = int(self._functions[0].space_order/2)

        # Edge detection
        # Want to add M/2 layers of padding on every edge
        # FIXME: would be more efficient to use a subdomainset here
        pg_shape = np.array(self._grid.shape) + 2*m_size
        pg_extent = (self._grid.extent[0] + 2*m_size*self._grid.spacing[0],
                     self._grid.extent[1] + 2*m_size*self._grid.spacing[1],
                     self._grid.extent[2] + 2*m_size*self._grid.spacing[2])
        padded_grid = Grid(shape=pg_shape, extent=pg_extent)
        edge_detect = TimeFunction(name='edge_detect', grid=padded_grid,
                                   space_order=self._functions[0].space_order)

        edge_detect.data[:] = np.pad(self._positive_mask, (m_size,), 'edge')

        # detect_eq = Eq(edge_detect.forward, edge_detect.div)
        detect_eq = Eq(edge_detect.forward, Max(abs(edge_detect.dx), abs(edge_detect.dy), abs(edge_detect.dz)))

        detect_op = Operator([detect_eq], name='DetectBoundary')
        detect_op.apply(time_M=1)
        # 1e-9 deals with floating point errors
        edge_mask = (edge_detect.data[1, m_size:-m_size, m_size:-m_size, m_size:-m_size] > 1e-9)
        self._boundary_node_mask = np.logical_and(self._positive_mask, edge_mask)

    def _distance_calculation(self):
        """
        Calculates the axial distances between the identified boundary nodes and the
        boundary surface.
        """
        # Node x, y, and z indices
        node_xind, node_yind, node_zind = np.where(self._boundary_node_mask)
        # vstack these
        self._boundary_nodes = np.vstack((node_xind, node_yind, node_zind)).T
        
        print(self._boundary_nodes)
        print(self._boundary_nodes.shape[0])
        # Query boundary nodes for distances
        axial_distances = self.query(self._boundary_nodes, index_input=True)
        # Set distances as variables
        self._z_dist = axial_distances[0]
        self._yp_dist = axial_distances[1]
        self._yn_dist = axial_distances[2]
        self._xp_dist = axial_distances[3]
        self._xn_dist = axial_distances[4]

        # FIXME: Occasionally z_dist contains nan (which shouldn't be the case)

    def plot_nodes(self, show_boundary=True, show_nodes=True, save=False, save_path=None):
        """
        Plots the boundary surface and the nodes identified as needing modification
        to their weights.
        """

        fig = plt.figure()
        plot_axes = fig.add_subplot(111, projection='3d')
        if show_boundary:
            plot_axes.plot_trisurf(self._boundary_data[:, 0],
                                   self._boundary_data[:, 1],
                                   self._boundary_data[:, 2],
                                   color='aquamarine')

        if show_nodes:
            plot_axes.scatter(self._boundary_nodes[:, 0]*self._grid.spacing[0],
                              self._boundary_nodes[:, 1]*self._grid.spacing[1],
                              self._boundary_nodes[:, 2]*self._grid.spacing[2],
                              marker='^', color='orangered')
        plot_axes.set_xlabel("x")
        plot_axes.set_ylabel("y")
        plot_axes.set_zlabel("z")
        plot_axes.set_zlim(-1*self._pmls*self._spacing[2], self._extent[2] - self._pmls*self._spacing[2], False)
        plot_axes.invert_zaxis()
        if save:
            if save_path is not None:
                plt.savefig(save_path)
            else:
                raise OSError("Invalid filepath.")
        plt.show()

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
        Attatch boundary conditions to be imposed on the specified function on
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

    def has_bcs(self, function):
        """
        Checks that a function attatched to the boundary has boundary
        conditions.

        Parameters
        ----------
        function : Devito function
            The function to be checked

        Returns
        -------
        bc_bool : bool
            True if the specified function has boundary conditions
            attatched.
        """
        if self.stencils[function.name].bc_list is None:
            return False
        return True

    def _calculate_stencils(self, function, deriv, stencil_out=None):
        """
        Calculate or retrieve the set of stencils required for the specified
        derivative and function.
        """
        if not self.has_bcs(function):
            raise RuntimeError("Function has no boundary conditions set")
        self.stencils[function.name].all_variants(deriv, stencil_out=stencil_out)
        # May get odd behaviour if calling this function for multiple different
        # derivatives of the same function repeatedly.

    def subs(self, spec, stencil_out=None):
        """
        Return a Substitutions object for all stencil modifications associated
        with the boundary, given the derivatives specified.

        Parameters
        ----------
        spec : dict
            Dictionary containing pairs of functions and their derivatives.
            E.g. {u : 2, v : 1} for second derivative of u and first of v.
        stencil_out : str
            Filepath to cache stencils if no file was specified for caching
            at initialisation. Default is None (no caching)
        """
        m_size = int(self._functions[0].space_order/2)

        # List to store weight functions for each function
        weights = []

        s_dim = Dimension(name='s')
        ncoeffs = self._functions[0].space_order+1

        wshape = self._grid.shape + (ncoeffs,)
        wdims = self._grid.dimensions + (s_dim,)

        # Can't have two derivatives of the same function due to matching keys
        # Unpack the dictionary
        for function in spec:
            # Loop over every item in the dictionary
            self._calculate_stencils(function, spec[function], stencil_out=stencil_out)

            # Set up weight functions for this function
            w_x = Function(name=function.name+"_w_x",
                           dimensions=wdims,
                           shape=wshape)
            w_y = Function(name=function.name+"_w_y",
                           dimensions=wdims,
                           shape=wshape)
            w_z = Function(name=function.name+"_w_z",
                           dimensions=wdims,
                           shape=wshape)

            # Initialise function data with standard FD weights
            exterior_mask = np.logical_not(self._positive_mask)

            # Construct standard stencils
            std_coeffs = finite_diff_weights(spec[function], range(-m_size, m_size+1), 0)[-1][-1]
            std_coeffs = np.array(std_coeffs)

            w_x.data[:, :, :] = std_coeffs[:]
            w_y.data[:, :, :] = std_coeffs[:]
            w_z.data[:, :, :] = std_coeffs[:]

            # Loop over set of points
            # Call self.stencils[function.name].subs() for each dimension for each modified point
            for i in range(self._boundary_nodes.shape[0]):
                pos_x = self._boundary_nodes[i, 0]
                pos_y = self._boundary_nodes[i, 1]
                pos_z = self._boundary_nodes[i, 2]
                if not np.isnan(self._z_dist[i]):
                    w_z.data[pos_x, pos_y, pos_z] \
                        = self.stencils[function.name].subs(eta_r=self._z_dist[i]) 
                else:
                    warnings.warn("Encountered missing z distance during stencil generation.")

                if not np.isnan(self._yp_dist[i]) or not np.isnan(self._yn_dist[i]):
                    if np.isnan(self._yp_dist[i]):
                        eta_r = None
                    else:
                        eta_r = self._yp_dist[i]
                    if np.isnan(self._yn_dist[i]):
                        eta_l = None
                    else:
                        eta_l = self._yn_dist[i]
                    
                    w_y.data[pos_x, pos_y, pos_z] \
                        = self.stencils[function.name].subs(eta_r=eta_r, eta_l=eta_l)

                if not np.isnan(self._xp_dist[i]) or not np.isnan(self._xn_dist[i]):
                    if np.isnan(self._xp_dist[i]):
                        eta_r = None
                    else:
                        eta_r = self._xp_dist[i]
                    if np.isnan(self._xn_dist[i]):
                        eta_l = None
                    else:
                        eta_l = self._xn_dist[i]
                    
                    w_x.data[pos_x, pos_y, pos_z] \
                        = self.stencils[function.name].subs(eta_r=eta_r, eta_l=eta_l)

            # Zero weights in the exterior
            # weights[function.name+"_x"]
            w_x.data[exterior_mask] = 0
            w_y.data[exterior_mask] = 0
            w_z.data[exterior_mask] = 0

            # derivative, dimension, function, weights
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


