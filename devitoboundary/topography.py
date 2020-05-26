"""
A module for implementation of topography in Devito via the immersed
boundary method.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from sympy import finite_diff_weights
from devito import Function, Dimension, Substitutions, Coefficient
from devito.tools import as_tuple
from devitoboundary import PolySurface
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
        The function(s) to which the boundary is to be attached.
    """

    def __init__(self, boundary_data, functions):
        self._boundary_data = np.array(boundary_data)
        # Check that boundary data has correct formatting
        assert len(self._boundary_data.shape) == 2, "Boundary data incorrectly formatted"
        assert self._boundary_data.shape[1] == 3, "GenericSurface is for 3D boundaries only"
        # Assert functions supplied as tuple
        assert isinstance(functions, tuple), "Functions must be supplied as tuple"
        # Check that all the functions share a grid
        for function in functions:
            assert function.grid is functions[0].grid, "All functions must share the same grid"
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
    behaviours: tuple
        The behaviours which each function wants to display at the boundary
        (e.g. 'antisymmetric_mirror').
    """

    def __init__(self, function, boundary_data, deriv_order, pmls=0):
        # Boundary3D.__init__(self, function, boundary_data, pmls)

        # self._generate_triangles()
        self._node_id()
        # self._construct_stencils(deriv_order)
        # self._weight_function(function, deriv_order)  # Temporarily disabled

    def _generate_triangles(self):
        """
        Generates a triangle mesh from 3D topography data using Delaunay
        triangulation. The surface of the mesh generated will be used
        to represent the boundary.
        """
        # Triangulation is 2D using x and y values
        self._mesh = Delaunay(self._boundary_data.iloc[:, 0:2]).simplices

    def _construct_plane(self, vertices):
        """
        Finds the equation of the plane defined by the triangle, along
        with its boundaries.
        """
        # Find equation of plane
        vertex_1 = self._boundary_data.iloc[vertices[:, 0]].to_numpy()
        vertex_2 = self._boundary_data.iloc[vertices[:, 1]].to_numpy()
        vertex_3 = self._boundary_data.iloc[vertices[:, 2]].to_numpy()
        vector_1 = vertex_2 - vertex_1
        vector_2 = vertex_3 - vertex_2
        plane_grad = np.cross(vector_1, vector_2)
        plane_const = -np.sum(plane_grad*vertex_1, axis=1)

        print("The first bit is done")
        # Return plane equation, vertices, and vectors of boundaries
        return vertex_1, vertex_2, vertex_3, \
            plane_grad, plane_const

    def _above_bool(self, block, gradient, constant, verts):
        """
        Returns True if a point is above the boundary (-ve z direction).
        Returns False otherwise.
        """

        loc_a = (block[0]*gradient[0]
                 + block[1]*gradient[1]
                 + block[2]*gradient[2]
                 + constant < 0)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = my + c instead (for handling edges of type x = c)
            xy_flip = ((verts[0, v_c[0]] - verts[0, v_c[1]]) == 0)
            clean_dx = (verts[0, v_c[0]] - verts[0, v_c[1]])  # Zero-free denominator
            clean_dy = (verts[1, v_c[0]] - verts[1, v_c[1]])
            if xy_flip:
                clean_dx = 1  # Prevents undefined behaviour
            else:
                clean_dy = 1  # Could be turned into try-except?

            # For edges of type y = mx + c
            m_x = (verts[1, v_c[0]] - verts[1, v_c[1]])/clean_dx
            c_x = verts[1, v_c[0]] - m_x*verts[0, v_c[0]]
            # For edges of type x = c
            m_y = (verts[0, v_c[0]] - verts[0, v_c[1]])/clean_dy
            c_y = verts[0, v_c[0]] - m_y*verts[1, v_c[0]]

            v_above = verts[1, v_3] > m_x*verts[0, v_3] + c_x
            # True if inside of triangle is in +ve y direction from edge or in
            # -ve x direction if edge is vertical.
            if xy_flip:
                v_above = (verts[0, v_3]
                           <= m_y*verts[1, v_3]
                           + c_y)

            cond_1 = np.logical_and(block[1] >= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   v_above))

            cond_2 = np.logical_and(block[0] <= m_y*block[1] + c_y,
                                    np.logical_and(xy_flip,
                                                   v_above))

            cond_3 = np.logical_and(block[1] <= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(block[0] >= m_y*block[1] + c_y,
                                    np.logical_and(xy_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_a = np.logical_and(loc_a, tri_bounds)

        return loc_a

    def _z_bool(self, block, gradient, constant, verts):
        """
        Returns True if a point is located within the z locus.
        Returns False otherwise.
        """

        # Points are below boundary (+ve z direction from boundary)
        loc_z = (block[0]*gradient[0]
                 + block[1]*gradient[1]
                 + block[2]*gradient[2]
                 + constant > 0)

        # Points are above base of locus (-ve z direction from locus base)
        loc_z_base = (block[0]*gradient[0]
                      + block[1]*gradient[1]
                      + (block[2] - (self._spacing[2]
                                     * self._method_order/2))*gradient[2]
                      + constant <= 0)

        loc_z = np.logical_and(loc_z, loc_z_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = my + c instead (for handling edges of type x = c)
            xy_flip = ((verts[0, v_c[0]] - verts[0, v_c[1]]) == 0)
            clean_dx = (verts[0, v_c[0]] - verts[0, v_c[1]])  # Zero-free denominator
            clean_dy = (verts[1, v_c[0]] - verts[1, v_c[1]])
            if xy_flip:
                clean_dx = 1  # Prevents undefined behaviour
            else:
                clean_dy = 1  # Could be turned into try-except?

            # For edges of type y = mx + c
            m_x = (verts[1, v_c[0]] - verts[1, v_c[1]])/clean_dx
            c_x = verts[1, v_c[0]] - m_x*verts[0, v_c[0]]
            # For edges of type x = c
            m_y = (verts[0, v_c[0]] - verts[0, v_c[1]])/clean_dy
            c_y = verts[0, v_c[0]] - m_y*verts[1, v_c[0]]

            v_above = verts[1, v_3] > m_x*verts[0, v_3] + c_x
            # True if inside of triangle is in +ve y direction from edge or in
            # -ve x direction if edge is vertical.
            if xy_flip:
                v_above = (verts[0, v_3]
                           <= m_y*verts[1, v_3]
                           + c_y)

            cond_1 = np.logical_and(block[1] >= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   v_above))

            cond_2 = np.logical_and(block[0] <= m_y*block[1] + c_y,
                                    np.logical_and(xy_flip,
                                                   v_above))

            cond_3 = np.logical_and(block[1] <= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(block[0] >= m_y*block[1] + c_y,
                                    np.logical_and(xy_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_z = np.logical_and(loc_z, tri_bounds)

        return loc_z

    def _y_bool(self, block, gradient, constant, verts, grad_pos, grad_neg):
        """
        Returns True if a point is located within the y locus.
        Returns False otherwise.
        """

        # Points are below boundary (+ve z direction from boundary)
        # Sometimes returns false for all if plane is flatish (sort later)
        loc_y = (block[0]*gradient[0]
                 + block[1]*gradient[1]
                 + block[2]*gradient[2]
                 + constant > 0)

        # Points are above base of locus (-ve z direction from locus base)
        loc_y_base = (block[0]*gradient[0]
                      + (block[1]
                         + (grad_neg*self._spacing[1]
                            * self._method_order/2)
                         - (grad_pos*self._spacing[1]
                            * self._method_order/2))*gradient[1]
                      + block[2]*gradient[2]
                      + constant <= 0)

        loc_y = np.logical_and(loc_y, loc_y_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = mz + c instead (for handling edges of type x = c)
            xz_flip = ((verts[0, v_c[0]] - verts[0, v_c[1]]) == 0)
            clean_dx = (verts[0, v_c[0]] - verts[0, v_c[1]])  # Zero-free denominator
            clean_dz = (verts[2, v_c[0]] - verts[2, v_c[1]])
            if xz_flip:
                clean_dx = 1  # Prevents undefined behaviour
            if verts[2, v_c[0]] - verts[2, v_c[1]] == 0:
                clean_dz = 1  # Could be turned into try-except?

            # For edges of type z = mx + c
            m_x = (verts[2, v_c[0]] - verts[2, v_c[1]])/clean_dx
            c_x = verts[2, v_c[0]] - m_x*verts[0, v_c[0]]
            # For edges of type x = c
            m_z = (verts[0, v_c[0]] - verts[0, v_c[1]])/clean_dz
            c_z = verts[0, v_c[0]] - m_z*verts[2, v_c[0]]

            v_above = verts[2, v_3] > m_x*verts[0, v_3] + c_x
            # True if inside of triangle is in +ve z direction from edge or in
            # -ve x direction if edge is vertical.
            if xz_flip:
                v_above = (verts[0, v_3]
                           <= m_z*verts[2, v_3]
                           + c_z)

            cond_1 = np.logical_and(block[2] >= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xz_flip),
                                                   v_above))

            cond_2 = np.logical_and(block[0] <= m_z*block[2] + c_z,
                                    np.logical_and(xz_flip,
                                                   v_above))

            cond_3 = np.logical_and(block[2] <= m_x*block[0] + c_x,
                                    np.logical_and(np.logical_not(xz_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(block[0] >= m_z*block[2] + c_z,
                                    np.logical_and(xz_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_y = np.logical_and(loc_y, tri_bounds)
        return loc_y

    def _x_bool(self, block, gradient, constant, verts, grad_pos, grad_neg):
        """
        Returns True if a point is located within the x locus.
        Returns False otherwise.
        """

        # Points are below boundary (+ve z direction from boundary)
        # Sometimes returns false for all if plane is flatish (sort later)
        loc_x = (block[0]*gradient[0]
                 + block[1]*gradient[1]
                 + block[2]*gradient[2]
                 + constant > 0)

        # Points are above base of locus (-ve z direction from locus base)
        loc_x_base = ((block[0]
                       + (grad_neg*self._spacing[0]
                          * self._method_order/2)
                       - (grad_pos*self._spacing[0]
                          * self._method_order/2))*gradient[0]
                      + block[1]*gradient[1]
                      + block[2]*gradient[2]
                      + constant <= 0)

        loc_x = np.logical_and(loc_x, loc_x_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use y = mz + c instead (for handling edges of type y = c)
            yz_flip = ((verts[1, v_c[0]] - verts[1, v_c[1]]) == 0)
            clean_dy = (verts[1, v_c[0]] - verts[1, v_c[1]])  # Zero-free denominator
            clean_dz = (verts[2, v_c[0]] - verts[2, v_c[1]])
            if yz_flip:
                clean_dy = 1  # Prevents undefined behaviour
            if verts[2, v_c[0]] - verts[2, v_c[1]] == 0:
                clean_dz = 1  # Could be turned into try-eycept?

            # For edges of type z = my + c
            m_y = (verts[2, v_c[0]] - verts[2, v_c[1]])/clean_dy
            c_y = verts[2, v_c[0]] - m_y*verts[1, v_c[0]]
            # For edges of type y = c
            m_z = (verts[1, v_c[0]] - verts[1, v_c[1]])/clean_dz
            c_z = verts[1, v_c[0]] - m_z*verts[2, v_c[0]]

            v_above = verts[2, v_3] > m_y*verts[1, v_3] + c_y
            # True if inside of triangle is in +ve z direction from edge or in
            # -ve y direction if edge is vertical.
            if yz_flip:
                v_above = (verts[1, v_3]
                           <= m_z*verts[2, v_3]
                           + c_z)

            cond_1 = np.logical_and(block[2] >= m_y*block[1] + c_y,
                                    np.logical_and(np.logical_not(yz_flip),
                                                   v_above))

            cond_2 = np.logical_and(block[1] <= m_z*block[2] + c_z,
                                    np.logical_and(yz_flip,
                                                   v_above))

            cond_3 = np.logical_and(block[2] <= m_y*block[1] + c_y,
                                    np.logical_and(np.logical_not(yz_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(block[1] >= m_z*block[2] + c_z,
                                    np.logical_and(yz_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_x = np.logical_and(loc_x, tri_bounds)
        return loc_x

    def _above_nodes(self, gradients, constants, vertices):
        """
        Returns all nodes within z locus, along with their corresponding eta
        values.
        """

        x_min = np.amin(vertices[:, 0, :], 1)
        x_max = np.amax(vertices[:, 0, :], 1)
        x_min_index = (np.ceil(x_min/self._spacing[0])).astype(int) + self._pmls
        x_max_index = (np.floor(x_max/self._spacing[0])).astype(int) + self._pmls
        valid_locus = (x_max_index >= x_min_index)  # Locus actually contains gridlines
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        y_min = np.amin(vertices[:, 1, :], 1)
        y_max = np.amax(vertices[:, 1, :], 1)
        y_min_index = (np.ceil(y_min/self._spacing[1])).astype(int) + self._pmls
        y_max_index = (np.floor(y_max/self._spacing[1])).astype(int) + self._pmls
        valid_locus = y_max_index >= y_min_index
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        y_min = y_min[valid_locus]
        y_max = y_max[valid_locus]
        y_min_index = y_min_index[valid_locus]
        y_max_index = y_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        z_max = np.amax(vertices[:, 2, :], 1)
        z_min_index = self._pmls
        z_max_index = (np.floor(z_max/self._spacing[2])).astype(int) + self._pmls

        a_locus_data = pd.DataFrame(columns=['x', 'y',
                                             'z'])

        for i in range(len(x_min)):
            # Create meshgrid
            locus_mesh = np.meshgrid(np.linspace((x_min_index[i] - self._pmls)*self._spacing[0],
                                                 (x_max_index[i] - self._pmls)*self._spacing[0],
                                                 x_max_index[i] - x_min_index[i] + 1),
                                     np.linspace((y_min_index[i] - self._pmls)*self._spacing[1],
                                                 (y_max_index[i] - self._pmls)*self._spacing[1],
                                                 y_max_index[i] - y_min_index[i] + 1),
                                     np.linspace((z_min_index - self._pmls)*self._spacing[2],
                                                 (z_max_index[i] - self._pmls)*self._spacing[2],
                                                 z_max_index[i] - z_min_index + 1))
            # Apply revamped _z_bool() for masking
            mask = self._above_bool(locus_mesh, gradients[i], constants[i], vertices[i])
            add_nodes = pd.DataFrame({'x': locus_mesh[0][mask],
                                      'y': locus_mesh[1][mask],
                                      'z': locus_mesh[2][mask]})
            a_locus_data = a_locus_data.append(add_nodes, ignore_index=True, sort=False)
        return a_locus_data

    def _z_nodes(self, gradients, constants, vertices):
        """
        Returns all nodes within z locus, along with their corresponding eta
        values.
        """

        x_min = np.amin(vertices[:, 0, :], 1)
        x_max = np.amax(vertices[:, 0, :], 1)
        x_min_index = (np.ceil(x_min/self._spacing[0])).astype(int) + self._pmls
        x_max_index = (np.floor(x_max/self._spacing[0])).astype(int) + self._pmls
        valid_locus = (x_max_index >= x_min_index)  # Locus actually contains gridlines
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        y_min = np.amin(vertices[:, 1, :], 1)
        y_max = np.amax(vertices[:, 1, :], 1)
        y_min_index = (np.ceil(y_min/self._spacing[1])).astype(int) + self._pmls
        y_max_index = (np.floor(y_max/self._spacing[1])).astype(int) + self._pmls
        valid_locus = y_max_index >= y_min_index
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        y_min = y_min[valid_locus]
        y_max = y_max[valid_locus]
        y_min_index = y_min_index[valid_locus]
        y_max_index = y_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        z_min = np.minimum(np.amin(vertices[:, 2, :], 1), self._extent[2] - self._pmls*self._spacing[2])
        z_max = np.minimum(np.amax(vertices[:, 2, :], 1) + 0.5*self._method_order*self._spacing[2], self._extent[2] - self._pmls*self._spacing[2])
        z_min_index = (np.ceil(z_min/self._spacing[2])).astype(int) + self._pmls
        z_max_index = (np.floor(z_max/self._spacing[2])).astype(int) + self._pmls

        z_locus_data = pd.DataFrame(columns=['x', 'y',
                                             'z', 'z_eta'])

        for i in range(len(x_min)):
            # Create meshgrid
            locus_mesh = np.meshgrid(np.linspace((x_min_index[i] - self._pmls)*self._spacing[0],
                                                 (x_max_index[i] - self._pmls)*self._spacing[0],
                                                 x_max_index[i] - x_min_index[i] + 1),
                                     np.linspace((y_min_index[i] - self._pmls)*self._spacing[1],
                                                 (y_max_index[i] - self._pmls)*self._spacing[1],
                                                 y_max_index[i] - y_min_index[i] + 1),
                                     np.linspace((z_min_index[i] - self._pmls)*self._spacing[2],
                                                 (z_max_index[i] - self._pmls)*self._spacing[2],
                                                 z_max_index[i] - z_min_index[i] + 1))
            # Apply revamped _z_bool() for masking
            mask = self._z_bool(locus_mesh, gradients[i], constants[i], vertices[i])
            eta = (-(gradients[i, 0]*locus_mesh[0][mask]
                     + gradients[i, 1]*locus_mesh[1][mask]
                     + constants[i])/gradients[i, 2] - locus_mesh[2][mask])/self._spacing[2]
            add_nodes = pd.DataFrame({'x': locus_mesh[0][mask],
                                      'y': locus_mesh[1][mask],
                                      'z': locus_mesh[2][mask],
                                      'z_eta': eta})
            z_locus_data = z_locus_data.append(add_nodes, ignore_index=True, sort=False)
        self._modified_nodes = pd.concat([self._modified_nodes, z_locus_data.iloc[:, :3]],
                                         sort=False).drop_duplicates().reset_index(drop=True)
        return z_locus_data

    def _y_nodes(self, gradients, constants, vertices):
        """
        Returns all nodes within y locus, along with their corresponding eta
        values.
        """

        valid_locus = np.logical_not(np.logical_and(vertices[:, 2, 0] == vertices[:, 2, 1],
                                                    vertices[:, 2, 0] == vertices[:, 2, 2]))  # Remove flat polygons
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        x_min = np.amin(vertices[:, 0, :], 1)
        x_max = np.amax(vertices[:, 0, :], 1)
        x_min_index = (np.ceil(x_min/self._spacing[0])).astype(int) + self._pmls
        x_max_index = (np.floor(x_max/self._spacing[0])).astype(int) + self._pmls
        valid_locus = (x_max_index >= x_min_index)  # Locus actually contains gridlines
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        z_min = np.minimum(np.amin(vertices[:, 2, :], 1), self._extent[2] - self._pmls*self._spacing[2])
        z_max = np.minimum(np.amax(vertices[:, 2, :], 1), self._extent[2] - self._pmls*self._spacing[2])
        z_min_index = (np.ceil(z_min/self._spacing[2])).astype(int) + self._pmls
        z_max_index = (np.floor(z_max/self._spacing[2])).astype(int) + self._pmls
        valid_locus = z_max_index >= z_min_index
        x_min = x_min[valid_locus]
        x_max = x_max[valid_locus]
        x_min_index = x_min_index[valid_locus]
        x_max_index = x_max_index[valid_locus]
        z_min = z_min[valid_locus]
        z_max = z_max[valid_locus]
        z_min_index = z_min_index[valid_locus]
        z_max_index = z_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        grad_positive = gradients[:, 1] > 0
        grad_negative = gradients[:, 1] < 0

        y_min = np.amin(vertices[:, 1, :], 1) - grad_negative*0.5*self._method_order*self._spacing[1]  # These may want to be more cleverly calculated for very rough surfaces
        y_max = np.amax(vertices[:, 1, :], 1) + grad_positive*0.5*self._method_order*self._spacing[1]  # Wants to be flipped depending on gradient
        y_min_index = (np.ceil(y_min/self._spacing[1])).astype(int) + self._pmls
        y_max_index = (np.floor(y_max/self._spacing[1])).astype(int) + self._pmls

        y_locus_data = pd.DataFrame(columns=['x', 'y',
                                             'z', 'y_eta_l',
                                             'y_eta_r'])

        for i in range(len(x_min)):
            # Create meshgrid
            locus_mesh = np.meshgrid(np.linspace((x_min_index[i] - self._pmls)*self._spacing[0],
                                                 (x_max_index[i] - self._pmls)*self._spacing[0],
                                                 x_max_index[i] - x_min_index[i] + 1),
                                     np.linspace((y_min_index[i] - self._pmls)*self._spacing[1],
                                                 (y_max_index[i] - self._pmls)*self._spacing[1],
                                                 y_max_index[i] - y_min_index[i] + 1),
                                     np.linspace((z_min_index[i] - self._pmls)*self._spacing[2],
                                                 (z_max_index[i] - self._pmls)*self._spacing[2],
                                                 z_max_index[i] - z_min_index[i] + 1))
            # Apply revamped _z_bool() for masking
            mask = self._y_bool(locus_mesh, gradients[i], constants[i],
                                vertices[i], grad_positive[i], grad_negative[i])
            eta_r = (-(gradients[i, 0]*locus_mesh[0][mask]
                       + gradients[i, 2]*locus_mesh[2][mask]
                       + constants[i])/gradients[i, 1] - locus_mesh[1][mask])/self._spacing[1]
            # Split eta into -ve (left) and +ve (right) values
            eta_l = eta_r.copy()
            eta_r[eta_r < 0] = np.nan
            eta_l[eta_l > 0] = np.nan
            add_nodes = pd.DataFrame({'x': locus_mesh[0][mask],
                                      'y': locus_mesh[1][mask],
                                      'z': locus_mesh[2][mask],
                                      'y_eta_r': eta_r,
                                      'y_eta_l': eta_l})
            y_locus_data = pd.concat([y_locus_data, add_nodes],
                                     sort=False).drop_duplicates().reset_index(drop=True)
        self._modified_nodes = pd.concat([self._modified_nodes, y_locus_data.iloc[:, :3]],
                                         sort=False).drop_duplicates().reset_index(drop=True)
        return y_locus_data

    def _x_nodes(self, gradients, constants, vertices):
        """
        Returns all nodes within x locus, along with their corresponding eta
        values.
        """

        valid_locus = np.logical_not(np.logical_and(vertices[:, 2, 0] == vertices[:, 2, 1],
                                                    vertices[:, 2, 0] == vertices[:, 2, 2]))  # Remove flat polygons
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        y_min = np.amin(vertices[:, 1, :], 1)
        y_max = np.amax(vertices[:, 1, :], 1)
        y_min_index = (np.ceil(y_min/self._spacing[1])).astype(int) + self._pmls
        y_max_index = (np.floor(y_max/self._spacing[1])).astype(int) + self._pmls
        valid_locus = (y_max_index >= y_min_index)  # Locus actually contains gridlines
        y_min = y_min[valid_locus]
        y_max = y_max[valid_locus]
        y_min_index = y_min_index[valid_locus]
        y_max_index = y_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        z_min = np.minimum(np.amin(vertices[:, 2, :], 1), self._extent[2] - self._pmls*self._spacing[2])
        z_max = np.minimum(np.amax(vertices[:, 2, :], 1), self._extent[2] - self._pmls*self._spacing[2])
        z_min_index = (np.ceil(z_min/self._spacing[2])).astype(int) + self._pmls
        z_max_index = (np.floor(z_max/self._spacing[2])).astype(int) + self._pmls
        valid_locus = z_max_index >= z_min_index
        y_min = y_min[valid_locus]
        y_max = y_max[valid_locus]
        y_min_index = y_min_index[valid_locus]
        y_max_index = y_max_index[valid_locus]
        z_min = z_min[valid_locus]
        z_max = z_max[valid_locus]
        z_min_index = z_min_index[valid_locus]
        z_max_index = z_max_index[valid_locus]
        gradients = gradients[valid_locus]
        constants = constants[valid_locus]
        vertices = vertices[valid_locus]

        grad_positive = gradients[:, 0] > 0
        grad_negative = gradients[:, 0] < 0

        x_min = np.amin(vertices[:, 0, :], 1) - grad_negative*0.5*self._method_order*self._spacing[0]  # These may want to be more cleverly calculated for very rough surfaces
        x_max = np.amax(vertices[:, 0, :], 1) + grad_positive*0.5*self._method_order*self._spacing[0]  # Wants to be flipped depending on gradient
        x_min_index = (np.ceil(x_min/self._spacing[0])).astype(int) + self._pmls
        x_max_index = (np.floor(x_max/self._spacing[0])).astype(int) + self._pmls

        x_locus_data = pd.DataFrame(columns=['x', 'y',
                                             'z', 'x_eta_l',
                                             'x_eta_r'])

        for i in range(len(x_min)):
            # Create meshgrid
            locus_mesh = np.meshgrid(np.linspace((x_min_index[i] - self._pmls)*self._spacing[0],
                                                 (x_max_index[i] - self._pmls)*self._spacing[0],
                                                 x_max_index[i] - x_min_index[i] + 1),
                                     np.linspace((y_min_index[i] - self._pmls)*self._spacing[1],
                                                 (y_max_index[i] - self._pmls)*self._spacing[1],
                                                 y_max_index[i] - y_min_index[i] + 1),
                                     np.linspace((z_min_index[i] - self._pmls)*self._spacing[2],
                                                 (z_max_index[i] - self._pmls)*self._spacing[2],
                                                 z_max_index[i] - z_min_index[i] + 1))
            # Apply revamped _z_bool() for masking
            mask = self._x_bool(locus_mesh, gradients[i], constants[i],
                                vertices[i], grad_positive[i], grad_negative[i])
            eta_r = (-(gradients[i, 1]*locus_mesh[1][mask]
                       + gradients[i, 2]*locus_mesh[2][mask]
                       + constants[i])/gradients[i, 0] - locus_mesh[0][mask])/self._spacing[0]
            # Split eta into -ve (left) and +ve (right) values
            eta_l = eta_r.copy()
            eta_r[eta_r < 0] = np.nan
            eta_l[eta_l > 0] = np.nan
            add_nodes = pd.DataFrame({'x': locus_mesh[0][mask],
                                      'y': locus_mesh[1][mask],
                                      'z': locus_mesh[2][mask],
                                      'x_eta_r': eta_r,
                                      'x_eta_l': eta_l})
            x_locus_data = pd.concat([x_locus_data, add_nodes],
                                     sort=False).drop_duplicates().reset_index(drop=True)
        self._modified_nodes = pd.concat([self._modified_nodes, x_locus_data.iloc[:, :3]],
                                         sort=False).drop_duplicates().reset_index(drop=True)
        return x_locus_data

    def _node_id(self):
        """
        Generates a list of nodes where stencils will require modification
        in either x or y directions.
        """

        self._modified_nodes = pd.DataFrame(columns=['x', 'y', 'z'])

        vertex_1, vertex_2, vertex_3, \
            plane_grad, plane_const = self._construct_plane(self._mesh)

        z_nodes = self._z_nodes(plane_grad, plane_const, np.dstack((vertex_1, vertex_2, vertex_3)))
        y_nodes = self._y_nodes(plane_grad, plane_const, np.dstack((vertex_1, vertex_2, vertex_3)))
        x_nodes = self._x_nodes(plane_grad, plane_const, np.dstack((vertex_1, vertex_2, vertex_3)))
        self._modified_nodes = self._modified_nodes.merge(z_nodes, how='outer', on=['x', 'y', 'z'])
        self._modified_nodes = self._modified_nodes.merge(y_nodes, how='outer', on=['x', 'y', 'z'])
        self._modified_nodes = self._modified_nodes.merge(x_nodes, how='outer', on=['x', 'y', 'z'])

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
