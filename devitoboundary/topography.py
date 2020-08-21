"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        positive_mask = self.fd_node_sides()

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

        edge_detect.data[:] = np.pad(positive_mask, (m_size,), 'edge')

        # detect_eq = Eq(edge_detect.forward, edge_detect.div)
        detect_eq = Eq(edge_detect.forward, Max(abs(edge_detect.dx), abs(edge_detect.dy), abs(edge_detect.dz)))

        detect_op = Operator([detect_eq], name='DetectBoundary')
        detect_op.apply(time_M=1)

        edge_mask = np.sign(edge_detect.data[1, m_size:-m_size, m_size:-m_size, m_size:-m_size])

        self._boundary_node_mask = np.logical_and(positive_mask, edge_mask)

    def _distance_calculation(self):
        """
        Calculates the axial distances between the identified boundary nodes and the
        boundary surface.
        """
        # Node x, y, and z indices
        node_xind, node_yind, node_zind = np.where(self._boundary_node_mask)
        # vstack these
        boundary_nodes = np.vstack((node_xind, node_yind, node_zind)).T
        print(boundary_nodes)
        # Query boundary nodes for distances
        axial_distances = self.query(boundary_nodes, index_input=True)
        # Set distances as variables
        self._z_dist = axial_distances[0]
        self._yp_dist = axial_distances[1]
        self._yn_dist = axial_distances[2]
        self._xp_dist = axial_distances[3]
        self._xn_dist = axial_distances[4]

    def plot_nodes(self, show_boundary=True, show_nodes=True, save=False, save_path=None):
        """
        Plots the boundary surface and the nodes identified as needing modification
        to their weights.
        """

        fig = plt.figure()
        plot_axes = fig.add_subplot(111, projection='3d')
        if show_boundary:
            plot_axes.plot_trisurf(self._boundary_data['x'],
                                   self._boundary_data['y'],
                                   self._boundary_data['z'] - self._pmls*self._spacing[2],
                                   color='aquamarine')
        if show_nodes:
            plot_axes.scatter(self._modified_nodes['x'],
                              self._modified_nodes['y'],
                              self._modified_nodes['z'] - self._pmls*self._spacing[2],
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
        if not has_bcs(function):
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
        # subs({u : 1, u : 2, v : 1})
        # Unpack the dictionary
        for function in spec:
            # Do the thing on every pair in the dictionary
            print(spec[function])

    def _generate_coefficients(self, node, deriv_order):
        """
        A coefficient generator for immersed boundaries. Uses floating point
        values to calculate stencil weights.
        """

        m_size = int(self._method_order/2)  # For tidiness

        # Construct standard stencils
        std_coeffs = finite_diff_weights(deriv_order, range(-m_size, m_size+1), 0)[-1][-1]
        std_coeffs = np.array(std_coeffs, dtype=np.float64)

        # Minor modifications in x direction
        coeffs_x = std_coeffs.copy()
        if node['x_eta_r'] % 1 == 0:
            coeffs_x[int(node['x_eta_r'])-m_size-1:] = 0
        if node['x_eta_l'] % 1 == 0:  # This % may be problematic
            coeffs_x[:1+m_size-int(node['x_eta_l'])] = 0

        # Minor modifications in y direction
        coeffs_y = std_coeffs.copy()
        if node['y_eta_r'] % 1 == 0:
            coeffs_y[int(node['y_eta_r'])-m_size-1:] = 0
        if node['y_eta_l'] % 1 == 0:
            coeffs_y[:1+m_size-int(node['y_eta_l'])] = 0

        # Minor modifications in z direction
        coeffs_z = std_coeffs.copy()
        if node['z_eta'] % 1 == 0:
            coeffs_y[:1+m_size-int(node['z_eta'])] = 0

        # One side outside in x direction
        if (not np.isnan(node['x_eta_r']) and np.isnan(node['x_eta_l'])) \
                or (np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
            if (not np.isnan(node['x_eta_r']) and np.isnan(node['x_eta_l'])):
                xi_x = node['x_eta_r'] % 1
                rows_x = int(m_size-node['x_eta_r'])+1  # Rows of extrapolation matrix
            else:  # Stencils will only be flipped at end
                xi_x = abs(node['x_eta_l']) % 1
                rows_x = int(m_size+node['x_eta_l'])+1  # Rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_x < 0.5:
                splay_x = True
            else:
                splay_x = False
            ex_matrix_x = np.zeros((rows_x, m_size))
            for i in range(rows_x):  # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_x) - xi_x)**(2*j+1))
                    rhs[j] = ((i+1) - xi_x)**(2*j+1)
                ex_matrix_x[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            add_coeffs = np.zeros(m_size)
            for i in range(m_size):
                add_coeffs[i] = np.dot(ex_matrix_x[:, i], coeffs_x[-rows_x:])
            coeffs_x[-rows_x-m_size-splay_x:-rows_x-splay_x] += add_coeffs
            coeffs_x[-rows_x:] = 0

            if (np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
                coeffs_x[:] = coeffs_x[::-1]  # Flip for boundaries on left

        # One side outside in y direction
        if (not np.isnan(node['y_eta_r']) and np.isnan(node['y_eta_l'])) \
                or (np.isnan(node['y_eta_r']) and not np.isnan(node['y_eta_l'])):
            if (not np.isnan(node['y_eta_r']) and np.isnan(node['y_eta_l'])):
                xi_y = node['y_eta_r'] % 1
                rows_y = int(m_size-node['y_eta_r'])+1  # Rows of extrapolation matrix
            else:  # Stencils will only be flipped at end
                xi_y = abs(node['y_eta_l']) % 1
                rows_y = int(m_size+node['y_eta_l'])+1  # Rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_y < 0.5:
                splay_y = True
            else:
                splay_y = False
            ex_matrix_y = np.zeros((rows_y, m_size))
            for i in range(rows_y):  # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_y) - xi_y)**(2*j+1))
                    rhs[j] = ((i+1) - xi_y)**(2*j+1)
                ex_matrix_y[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            add_coeffs = np.zeros(m_size)
            for i in range(m_size):
                add_coeffs[i] = np.dot(ex_matrix_y[:, i], coeffs_y[-rows_y:])
            coeffs_y[-rows_y-m_size-splay_y:-rows_y-splay_y] += add_coeffs
            coeffs_y[-rows_y:] = 0

            if (np.isnan(node['y_eta_r']) and not np.isnan(node['y_eta_l'])):
                coeffs_y[:] = coeffs_y[::-1]  # Flip for boundaries on left

        # Only one side can be outside in z direction
        if not np.isnan(node['z_eta']):
            xi_z = abs(node['z_eta']) % 1
            rows_z = int(m_size+node['z_eta'])+1  # rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_z < 0.5:
                splay_z = True
            else:
                splay_z = False
            ex_matrix_z = np.zeros((rows_z, m_size))
            for i in range(rows_z):  # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_z) - xi_z)**(2*j+1))
                    rhs[j] = ((i+1) - xi_z)**(2*j+1)
                ex_matrix_z[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            add_coeffs = np.zeros(m_size)
            for i in range(m_size):
                add_coeffs[i] = np.dot(ex_matrix_z[:, i], coeffs_z[-rows_z:])
            if len(coeffs_z[-rows_z-m_size-splay_z:-rows_z-splay_z]) != len(add_coeffs):
                print(node['x'], node['y'], node['z'])
                print(coeffs_z[-rows_z-m_size-splay_z:-rows_z-splay_z])
                print(add_coeffs)
            coeffs_z[-rows_z-m_size-splay_z:-rows_z-splay_z] += add_coeffs
            coeffs_z[-rows_z:] = 0
            coeffs_z[:] = coeffs_z[::-1]  # Flip for boundaries on left

        # Both sides outside in x direction <- This never happens as the locus can be projected outside the boundary
        if (not np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
            raise NotImplementedError("Stencil overlaps boundary on both sides at (%.1f, %.1f, %.1f)"
                                      % (node['x'], node['y'], node['z']))

        # Both sides outside in y direction
        if (not np.isnan(node['y_eta_r']) and not np.isnan(node['y_eta_l'])):
            raise NotImplementedError("Stencil overlaps boundary on both sides at (%.1f, %.1f, %.1f)"
                                      % (node['x'], node['y'], node['z']))

        return pd.Series({'x_coeffs': coeffs_x, 'y_coeffs': coeffs_y, 'z_coeffs': coeffs_z})

    def _construct_stencils(self, deriv_order):
        """
        Constructs stencils for a set of identified nodes.
        """

        self._modified_nodes[['x_coeffs', 'y_coeffs', 'z_coeffs']] \
            = self._modified_nodes.apply(self._generate_coefficients, axis=1, args=(deriv_order,))
        # print(self._modified_nodes[self._modified_nodes['z'] == 100][['x', 'y', 'z', 'x_eta_l', 'x_eta_r', 'y_eta_l', 'y_eta_r']])

    def _crap_weight_filler(self, deriv_order):  # Temporary
        """
        Temporary weight filler for exterior and interior points. Will be
        made redundant once necessary features are implemented in Devito.
        """

        # Fill with standard weights

        vertex_1, vertex_2, vertex_3, \
            plane_grad, plane_const = self._construct_plane(self._mesh)

        m_size = int(self._method_order/2)  # For tidiness

        # Construct standard stencils
        std_coeffs = finite_diff_weights(deriv_order, range(-m_size, m_size+1), 0)[-1][-1]
        std_coeffs = np.array(std_coeffs)

        self._w_x.data[:, :, :] = std_coeffs[:]
        self._w_y.data[:, :, :] = std_coeffs[:]
        self._w_z.data[:, :, :] = std_coeffs[:]

        # Fill zero weights
        above_nodes = self._above_nodes(plane_grad, plane_const, np.dstack((vertex_1, vertex_2, vertex_3)))
        self._w_x.data[np.round_(above_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(above_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(above_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = 0

        self._w_y.data[np.round_(above_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(above_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(above_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = 0

        self._w_z.data[np.round_(above_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(above_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(above_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = 0

    def _weight_function(self, function, deriv_order):
        """
        Creates three Devito functions containing weights needed for
        immersed boundary method. Each function contains weights for one
        dimension.
        """

        # Create weight function
        s_dim = Dimension(name='s')
        ncoeffs = self._method_order+1

        wshape = list(self._shape)
        wshape.append(ncoeffs)
        wshape = as_tuple(wshape)

        wdims = list(self._dimensions)
        wdims.append(s_dim)
        wdims = as_tuple(wdims)

        self._w_x = Function(name='w_x', dimensions=wdims, shape=wshape)
        self._w_y = Function(name='w_y', dimensions=wdims, shape=wshape)
        self._w_z = Function(name='w_z', dimensions=wdims, shape=wshape)

        # ########### Temporary ########### #
        self._crap_weight_filler(deriv_order)
        # ########### Temporary ########### #

        self._w_x.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = np.vstack(self._modified_nodes['x_coeffs'].values)

        self._w_y.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = np.vstack(self._modified_nodes['y_coeffs'].values)

        self._w_z.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int') + self._pmls,
                       np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int') + self._pmls,
                       :] = np.vstack(self._modified_nodes['z_coeffs'].values)

        self.subs = Substitutions(Coefficient(deriv_order, function,
                                              function.grid.dimensions[0],
                                              self._w_x),
                                  Coefficient(deriv_order, function,
                                              function.grid.dimensions[1],
                                              self._w_y),
                                  Coefficient(deriv_order, function,
                                              function.grid.dimensions[2],
                                              self._w_z))
