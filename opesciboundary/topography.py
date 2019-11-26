"""
A module for implementation of topography in Devito via use of
the immersed boundary method.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from sympy import finite_diff_weights
from devito import Function, Dimension
from devito.tools import as_tuple

__all__ = ['Boundary']


class Boundary():
    """
    An object to contain data relevant for implementing the
    immersed boundary method on a given domain.
    """

    def __init__(self, grid, boundary_data,
                 method_order=4): # FIXME: Add deriv_order

        self._method_order = method_order

        # Derive useful properties from grid
        self._shape = np.asarray(grid.shape)
        self._extent = np.asarray(grid.extent)
        self._spacing = grid.spacing
        self._dimensions = grid.dimensions

        self._read_in(boundary_data)

        self._generate_triangles()

        self._node_id()

        self._weight_function(2)

    @property
    def method_order(self):
        """
        Order of the FD discretisation.
        """
        return self._method_order


    def _read_in(self, boundary_data):
        """
        A function to read in topography data, and output as a pandas
        dataframe.
        """

        self._boundary_data = boundary_data


    # def add_topography(self, boundary_data):
    # Function for adding more boundary data


    def _generate_triangles(self):
        """
        Generates a triangle mesh from 3D topography data using Delaunay
        triangulation. The surface of the mesh generated will be used
        to represent the boundary.
        """

        # Note that triangulation is 2D using x and y values
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



        # Return plane equation, vertices, and vectors of boundaries
        return vertex_1, vertex_2, vertex_3, \
        plane_grad, plane_const


    @staticmethod
    def _above_bool(node, gradient, constant, verts):
        """
        Returns True if a point is located above the boundary. Returns
        False otherwise.
        """
        # Points are above
        loc_above = (node['x']*gradient[:, 0]
                     + node['y']*gradient[:, 1]
                     + node['z']*gradient[:, 2]
                     + constant[:] >= 0)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = my + c instead (for handling edges of type x = c)
            xy_flip = ((verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) == 0)
            clean_dx = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) # Zero-free denominator
            clean_dx[xy_flip] = 1 # Prevents undefined behaviour
            clean_dy = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1])
            clean_dy[np.logical_not(xy_flip)] = 1

            # For edges of type y = mx + c
            m_x = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1])/clean_dx
            c_x = verts[v_c[0]][:, 1] - m_x*verts[v_c[0]][:, 0]
            # For edges of type x = c
            m_y = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0])/clean_dy
            c_y = verts[v_c[0]][:, 0] - m_y*verts[v_c[0]][:, 1]

            v_above = verts[v_3][:, 1] > m_x*verts[v_3][:, 0] + c_x
            # True if inside of triangle is in +ve y direction from edge or in
            # -ve x direction if edge is vertical.
            v_above[xy_flip] = (verts[v_3][xy_flip, 0]
                                <= m_y[xy_flip]*verts[v_3][xy_flip, 1]
                                + c_y[xy_flip])

            cond_1 = np.logical_and(node['y'] >= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   v_above))

            cond_2 = np.logical_and(node['x'] <= m_y*node['y'] + c_y,
                                    np.logical_and(xy_flip,
                                                   v_above))

            cond_3 = np.logical_and(node['y'] <= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(node['x'] >= m_y*node['y'] + c_y,
                                    np.logical_and(xy_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_above = np.logical_and(loc_above, tri_bounds)

        if True in loc_above:
            return True
        return False


    def _z_bool(self, node, gradient, constant, verts, z_eta):
        """
        Returns True if a point is located within the z locus.
        Returns False otherwise. Also modifies the list given as
        an argument to contain respective eta values.
        """
        # Points are below boundary
        # FIXME: Probably redundant?
        loc_z = (node['x']*gradient[:, 0]
                 + node['y']*gradient[:, 1]
                 + node['z']*gradient[:, 2]
                 + constant[:] < 0)
        # Points are above base of locus
        loc_z_base = (node['x']*gradient[:, 0]
                      + node['y']*gradient[:, 1]
                      + (node['z'] + (self._spacing[2]
                                      *self._method_order/2))*gradient[:, 2]
                      + constant[:] >= 0)

        loc_z = np.logical_and(loc_z, loc_z_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = my + c instead (for handling edges of type x = c)
            xy_flip = ((verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) == 0)
            clean_dx = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) # Zero-free denominator
            clean_dx[xy_flip] = 1 # Prevents undefined behaviour
            clean_dy = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1])
            clean_dy[np.logical_not(xy_flip)] = 1

            # For edges of type y = mx + c
            m_x = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1])/clean_dx
            c_x = verts[v_c[0]][:, 1] - m_x*verts[v_c[0]][:, 0]
            # For edges of type x = c
            m_y = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0])/clean_dy
            c_y = verts[v_c[0]][:, 0] - m_y*verts[v_c[0]][:, 1]

            v_above = verts[v_3][:, 1] > m_x*verts[v_3][:, 0] + c_x
            # True if inside of triangle is in +ve y direction from edge or in
            # -ve x direction if edge is vertical.
            v_above[xy_flip] = (verts[v_3][xy_flip, 0]
                                <= m_y[xy_flip]*verts[v_3][xy_flip, 1]
                                + c_y[xy_flip])

            cond_1 = np.logical_and(node['y'] >= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   v_above))

            cond_2 = np.logical_and(node['x'] <= m_y*node['y'] + c_y,
                                    np.logical_and(xy_flip,
                                                   v_above))

            cond_3 = np.logical_and(node['y'] <= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xy_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(node['x'] >= m_y*node['y'] + c_y,
                                    np.logical_and(xy_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_z = np.logical_and(loc_z, tri_bounds)

        if True in loc_z:
            # Find intersections with line in y direction
            intersections = (-constant[loc_z]
                             - node['y']*gradient[loc_z, 1]
                             - node['x']*gradient[loc_z, 0])/gradient[loc_z, 2]
            # Find smallest dy required in positive and negative directions

            z_eta.append(((intersections[intersections > node['z']]
                           - node['z'])/self._spacing[2])[0])

            return True
        return False


    def _y_bool(self, node, gradient, constant, y_bools, verts, y_eta):
        """
        Returns True if a point is located within the y locus.
        Returns False otherwise. Also modifies the list given as
        an argument to contain respective eta values.
        """
        # Points are on negative side (inner side) of boundary
        loc_y = (node['x']*gradient[:, 0]
                 + node['y']*gradient[:, 1]
                 + node['z']*gradient[:, 2]
                 + constant[:] < 0)

        # Points are above base of locus
        loc_y_base = (node['x']*gradient[:, 0]
                      + (node['y']
                         + (y_bools[0]*self._spacing[1]
                            *self._method_order/2)
                         - (y_bools[1]*self._spacing[1]
                            *self._method_order/2))*gradient[:, 1]
                      + node['z']*gradient[:, 2]
                      + constant[:] >= 0)

        loc_y = np.logical_and(loc_y, loc_y_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use x = mz + c instead (for handling edges of type z = c)
            xz_flip = ((verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) == 0)
            clean_dx = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0]) # Zero-free denominator
            clean_dx[xz_flip] = 1 # Prevents undefined behaviour
            clean_dz = (verts[v_c[0]][:, 2] - verts[v_c[1]][:, 2])
            clean_dz[np.logical_not(xz_flip)] = 1

            # For edges of type z = mx + c
            m_x = (verts[v_c[0]][:, 2] - verts[v_c[1]][:, 2])/clean_dx
            c_x = verts[v_c[0]][:, 2] - m_x*verts[v_c[0]][:, 0]
            # For edges of type x = c
            m_z = (verts[v_c[0]][:, 0] - verts[v_c[1]][:, 0])/clean_dz
            c_z = verts[v_c[0]][:, 0] - m_z*verts[v_c[0]][:, 2]

            v_above = verts[v_3][:, 2] > m_x*verts[v_3][:, 0] + c_x
            # True if inside of triangle is in +ve y direction from edge or in
            # -ve x direction if edge is vertical.
            v_above[xz_flip] = (verts[v_3][xz_flip, 0]
                                <= m_z[xz_flip]*verts[v_3][xz_flip, 2]
                                + c_z[xz_flip])

            cond_1 = np.logical_and(node['z'] >= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xz_flip),
                                                   v_above))

            cond_2 = np.logical_and(node['x'] <= m_z*node['z'] + c_z,
                                    np.logical_and(xz_flip,
                                                   v_above))

            cond_3 = np.logical_and(node['z'] <= m_x*node['x'] + c_x,
                                    np.logical_and(np.logical_not(xz_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(node['x'] >= m_z*node['z'] + c_z,
                                    np.logical_and(xz_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_y = np.logical_and(loc_y, tri_bounds)

        if True in loc_y:
            # Find intersections with line in y direction
            intersections = (-constant[loc_y]
                             - node['z']*gradient[loc_y, 2]
                             - node['x']*gradient[loc_y, 0])/gradient[loc_y, 1]
            # Find closest intersection in positive and negative directions
            try:
                y_eta[1].append((np.min(intersections[intersections > node['y']])
                                 - node['y'])/self._spacing[1])
            except ValueError:
                y_eta[1].append(np.nan)
            try:
                y_eta[0].append((np.max(intersections[intersections < node['y']])
                                 - node['y'])/self._spacing[1])
            except ValueError:
                y_eta[0].append(np.nan)
            return True
        return False


    def _x_bool(self, node, gradient, constant, x_bools, verts, x_eta):
        """
        Returns True if a point is located within the x locus.
        Returns False otherwise. Also modifies the list given as
        an argument to contain respective eta values.
        """
        # Points are on negative side (inner side) of boundary
        loc_x = (node['x']*gradient[:, 0]
                 + node['y']*gradient[:, 1]
                 + node['z']*gradient[:, 2]
                 + constant[:] < 0)

        # Points are above base of locus
        loc_x_base = ((node['x']
                       + (x_bools[0]*self._spacing[0]
                          *self._method_order/2)
                       - (x_bools[1]*self._spacing[0]
                          *self._method_order/2))*gradient[:, 0]
                      + node['y']*gradient[:, 1]
                      + node['z']*gradient[:, 2]
                      + constant[:] >= 0)

        loc_x = np.logical_and(loc_x, loc_x_base)

        for v_c in combinations(range(3), 2):
            v_3 = np.setdiff1d(range(3), v_c)[0]

            # Points are within boundaries of triangle

            # Use y = mz + c instead (for handling edges of type z = c)
            yz_flip = ((verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1]) == 0)
            clean_dy = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1]) # Zero-free denominator
            clean_dy[yz_flip] = 1 # Prevents undefined behaviour
            clean_dz = (verts[v_c[0]][:, 2] - verts[v_c[1]][:, 2])
            clean_dz[np.logical_not(yz_flip)] = 1

            # For edges of type z = my + c
            m_y = (verts[v_c[0]][:, 2] - verts[v_c[1]][:, 2])/clean_dy
            c_y = verts[v_c[0]][:, 2] - m_y*verts[v_c[0]][:, 1]
            # For edges of type y = c
            m_z = (verts[v_c[0]][:, 1] - verts[v_c[1]][:, 1])/clean_dz
            c_z = verts[v_c[0]][:, 1] - m_z*verts[v_c[0]][:, 2]

            v_above = verts[v_3][:, 2] > m_y*verts[v_3][:, 1] + c_y
            # True if inside of triangle is in +ve z direction from edge or in
            # -ve y direction if edge is vertical.
            v_above[yz_flip] = (verts[v_3][yz_flip, 1]
                                <= m_z[yz_flip]*verts[v_3][yz_flip, 2]
                                + c_z[yz_flip])

            cond_1 = np.logical_and(node['z'] >= m_y*node['y'] + c_y,
                                    np.logical_and(np.logical_not(yz_flip),
                                                   v_above))

            cond_2 = np.logical_and(node['y'] <= m_z*node['z'] + c_z,
                                    np.logical_and(yz_flip,
                                                   v_above))

            cond_3 = np.logical_and(node['z'] <= m_y*node['y'] + c_y,
                                    np.logical_and(np.logical_not(yz_flip),
                                                   np.logical_not(v_above)))

            cond_4 = np.logical_and(node['y'] >= m_z*node['z'] + c_z,
                                    np.logical_and(yz_flip,
                                                   np.logical_not(v_above)))

            # Horrible Russian doll to do 'or' of all the above
            tri_bounds = np.logical_or(np.logical_or(cond_1, cond_2),
                                       np.logical_or(cond_3, cond_4))

            loc_x = np.logical_and(loc_x, tri_bounds)

        if True in loc_x:
            # Find intersections with line in x direction
            intersections = (-constant[loc_x]
                             - node['z']*gradient[loc_x, 2]
                             - node['y']*gradient[loc_x, 1])/gradient[loc_x, 0]
            # Find closest intersection in positive and negative directions
            try:
                x_eta[1].append((np.min(intersections[intersections > node['x']])
                                 - node['x'])/self._spacing[0])
            except ValueError:
                x_eta[1].append(np.nan)
            try:
                x_eta[0].append((np.max(intersections[intersections < node['x']])
                                 - node['x'])/self._spacing[0])
            except ValueError:
                x_eta[0].append(np.nan)
            return True
        return False


    def _construct_loci(self, vertices):
        """
        Constructs a locus on the inner side of the plane. The thickness
        of the locus is equal to M/2 grid spacings where M is the order
        of the FD method.
        """

        vertex_1, vertex_2, vertex_3, \
        plane_grad, plane_const = self._construct_plane(vertices)

        # Create block of possible points (will carve these out)
        z_max = self._boundary_data['z'].max()
        z_min = max(0, self._boundary_data['z'].min()
                    - (self._spacing[2]*self._method_order/2))
        z_max -= (z_max%self._spacing[2])
        z_min -= (z_max%self._spacing[2])
        z_node_count = int((z_max - z_min)/self._spacing[2]) + 1

        block_x, block_y, block_z = np.meshgrid(np.linspace(0, self._extent[0],
                                                            self._shape[0]),
                                                np.linspace(0, self._extent[1],
                                                            self._shape[1]),
                                                np.linspace(z_min, z_max,
                                                            z_node_count))

        block = pd.DataFrame({'x':block_x.flatten(),
                              'y':block_y.flatten(),
                              'z':block_z.flatten()}, columns=['x', 'y',
                                                               'z', 'x_eta_l',
                                                               'x_eta_r', 'y_eta_l',
                                                               'y_eta_r', 'z_eta'])

        # Delete all points above boundary
        block = block.drop(block[block.apply(self._above_bool, axis=1,
                                             args=(plane_grad, plane_const,
                                                   (vertex_1, vertex_2, vertex_3)))].index)

        # Used to pass eta values outside function
        z_eta = []
        y_eta = [[], []]
        x_eta = [[], []]

        z_locus = block.apply(self._z_bool, axis=1,
                              args=(plane_grad, plane_const,
                                    (vertex_1, vertex_2, vertex_3),
                                    z_eta))

        y_grad_plus = plane_grad[:, 1] > 0
        y_grad_minus = plane_grad[:, 1] < 0

        y_locus = block.apply(self._y_bool, axis=1,
                              args=(plane_grad, plane_const,
                                    (y_grad_plus, y_grad_minus),
                                    (vertex_1, vertex_2, vertex_3),
                                    y_eta))

        x_grad_plus = plane_grad[:, 0] > 0
        x_grad_minus = plane_grad[:, 0] < 0

        x_locus = block.apply(self._x_bool, axis=1,
                              args=(plane_grad, plane_const,
                                    (x_grad_plus, x_grad_minus),
                                    (vertex_1, vertex_2, vertex_3),
                                    x_eta))


        block.loc[z_locus, 'z_eta'] = z_eta
        block.loc[y_locus, 'y_eta_l'] = y_eta[0]
        block.loc[y_locus, 'y_eta_r'] = y_eta[1]
        block.loc[x_locus, 'x_eta_l'] = x_eta[0]
        block.loc[x_locus, 'x_eta_r'] = x_eta[1]



        # Append all points in z locus to modified nodes DataFrame
        self._modified_nodes = self._modified_nodes.append(block[z_locus],
                                                           ignore_index=True,
                                                           sort=False)

        # Append all points in y locus to modified nodes DataFrame
        self._modified_nodes = pd.concat([self._modified_nodes,
                                          block[y_locus]],
                                         sort=False).drop_duplicates().reset_index(drop=True)

        # Append all points in x locus to modified nodes DataFrame
        self._modified_nodes = pd.concat([self._modified_nodes,
                                          block[x_locus]],
                                         sort=False).drop_duplicates().reset_index(drop=True)


    def _node_id(self):
        """
        Generates a list of nodes where stencils will require modification
        in either x or y directions.
        """

        self._modified_nodes = pd.DataFrame(columns=['x', 'y',
                                                     'z', 'x_eta_l',
                                                     'x_eta_r', 'y_eta_l',
                                                     'y_eta_r', 'z_eta'])

        self._construct_loci(self._mesh)

    def plot_nodes(self): # FIXME: Add option to save figure
        """
        Plots the boundary surface and the nodes identified as needing modification
        to their weights.
        """

        fig = plt.figure()
        plot_axes = fig.add_subplot(111, projection='3d')
        plot_axes.plot_trisurf(self._boundary_data['x'],
                               self._boundary_data['y'],
                               self._boundary_data['z'])
        plot_axes.scatter(self._modified_nodes['x'],
                          self._modified_nodes['y'],
                          self._modified_nodes['z'],
                          marker='^', color='g')
        plt.show()


    def _generate_coefficients(self, node, deriv_order):
        """
        A coefficient generator for immersed boundaries. Uses floating point
        values to calculate stencil weights.
        """

        m_size = int(self._method_order/2) # For tidiness

        # Construct standard stencils
        std_coeffs = finite_diff_weights(deriv_order, range(-m_size, m_size+1), 0)[-1][-1]
        std_coeffs = np.array(std_coeffs)

        # Minor modifications in x direction
        coeffs_x = std_coeffs.copy()
        if node['x_eta_r']%1 == 0:
            coeffs_x[int(node['x_eta_r'])-m_size-1:] = 0
        if node['x_eta_l']%1 == 0:
            coeffs_x[:1+m_size-int(node['x_eta_l'])] = 0

        # Minor modifications in y direction
        coeffs_y = std_coeffs.copy()
        if node['y_eta_r']%1 == 0:
            coeffs_y[int(node['y_eta_r'])-m_size-1:] = 0
        if node['y_eta_l']%1 == 0:
            coeffs_y[:1+m_size-int(node['x_eta_y'])] = 0

        # Minor modifications in z direction
        coeffs_z = std_coeffs.copy()
        if node['z_eta']%1 == 0:
            coeffs_z[int(node['z_eta'])-m_size-1:] = 0

        # One side outside in x direction
        if (not np.isnan(node['x_eta_r']) and np.isnan(node['x_eta_l'])) \
            or (np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
            if (not np.isnan(node['x_eta_r']) and np.isnan(node['x_eta_l'])):
                xi_x = node['x_eta_r']%1
                rows_x = int(m_size-node['x_eta_r'])+1 # Rows of extrapolation matrix
            else: # Stencils will only be flipped at end
                xi_x = abs(node['x_eta_l'])%1
                rows_x = int(m_size+node['x_eta_l'])+1 # Rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_x < 0.5:
                splay_x = True
            else:
                splay_x = False
            ex_matrix_x = np.zeros((rows_x, m_size))
            for i in range(rows_x): # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_x) - xi_x)**(2*j+1))
                    rhs[j] = ((i+1) - xi_x)**(2*j+1)
                ex_matrix_x[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            ex_values = np.dot(ex_matrix_x,
                               coeffs_x[-rows_x-m_size-splay_x:-rows_x-splay_x])
            add_coeffs = np.multiply(ex_values, coeffs_x[-rows_x:])
            coeffs_x[-rows_x-m_size-splay_x:-rows_x-splay_x] += add_coeffs
            coeffs_x[-rows_x:] = 0

            if (np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
                coeffs_x[:] = coeffs_x[::-1] # Flip for boundaries on left

        # One side outside in y direction
        if (not np.isnan(node['y_eta_r']) and np.isnan(node['y_eta_l'])) \
            or (np.isnan(node['y_eta_r']) and not np.isnan(node['y_eta_l'])):
            if (not np.isnan(node['y_eta_r']) and np.isnan(node['y_eta_l'])):
                xi_y = node['y_eta_r']%1
                rows_y = int(m_size-node['y_eta_r'])+1 # Rows of extrapolation matrix
            else: # Stencils will only be flipped at end
                xi_y = abs(node['y_eta_l'])%1
                rows_y = int(m_size+node['y_eta_l'])+1 # Rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_y < 0.5:
                splay_y = True
            else:
                splay_y = False
            ex_matrix_y = np.zeros((rows_y, m_size))
            for i in range(rows_y): # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_y) - xi_y)**(2*j+1))
                    rhs[j] = ((i+1) - xi_y)**(2*j+1)
                ex_matrix_y[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            ex_values = np.dot(ex_matrix_y,
                                 coeffs_y[-rows_y-m_size-splay_y:-rows_y-splay_y])
            add_coeffs = np.multiply(ex_values, coeffs_y[-rows_y:])
            coeffs_y[-rows_y-m_size-splay_y:-rows_y-splay_y] += add_coeffs
            coeffs_y[-rows_y:] = 0

            if (np.isnan(node['y_eta_r']) and not np.isnan(node['y_eta_l'])):
                coeffs_y[:] = coeffs_y[::-1] # Flip for boundaries on left

        # Only one side can be outside in z direction
        if not np.isnan(node['z_eta']):
            xi_z = node['z_eta']%1
            rows_z = int(m_size-node['z_eta'])+1 # rows of extrapolation matrix
            # If statement for splaying extrapolation
            if xi_z < 0.5:
                splay_z = True
            else:
                splay_z = False
            ex_matrix_z = np.zeros((rows_z, m_size))
            for i in range(rows_z): # Loop over matrix rows
                lhs = np.zeros((m_size, m_size))
                rhs = np.zeros(m_size)
                for j in range(m_size):
                    for k in range(m_size):
                        lhs[j, -1-k] = ((-(k+splay_z) - xi_z)**(2*j+1))
                    rhs[j] = ((i+1) - xi_z)**(2*j+1)
                ex_matrix_z[i] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            # Apply extrapolation matrix to coefficients
            ex_values = np.dot(ex_matrix_z,
                                 coeffs_z[-rows_z-m_size-splay_z:-rows_z-splay_z])
            add_coeffs = np.multiply(ex_values, coeffs_z[-rows_z:])
            coeffs_z[-rows_z-m_size-splay_z:-rows_z-splay_z] += add_coeffs
            coeffs_z[-rows_z:] = 0

        # Both sides outside in x direction
        # FIXME: Want to rearrange to run in decreasing complexity
        # FIXME: Return to this later
        if (not np.isnan(node['x_eta_r']) and not np.isnan(node['x_eta_l'])):
            ex_matrix_x = np.zeros((self._method_order+1, self._method_order+1))
            for i in range(int(node['x_eta_l'])+m_size, int(node['x_eta_r'])+m_size+1):
                ex_matrix_x[i, i] = 1

        return pd.Series({'x_coeffs':coeffs_x, 'y_coeffs':coeffs_y, 'z_coeffs':coeffs_z})


    def _construct_stencils(self, deriv_order):
        """
        Constructs stencils for a set of identified nodes.
        """

        self._modified_nodes[['x_coeffs', 'y_coeffs', 'z_coeffs']] \
        = self._modified_nodes.apply(self._generate_coefficients, axis=1, args=(deriv_order,))


    def _weight_function(self, deriv_order):
        """
        Creates three Devito functions containing weights needed for
        immersed boundary method. Each function contains weights for one
        dimension.
        """

        self._construct_stencils(deriv_order)

        # Create weight function
        s_dim = Dimension(name='s')
        ncoeffs = self._method_order+1

        wshape = list(self._shape)
        wshape.append(ncoeffs)
        wshape = as_tuple(wshape)

        wdims = list(self._dimensions)
        wdims.append(s_dim)
        wdims = as_tuple(wdims)

        self.w_x = Function(name='w_x', dimensions=wdims, shape=wshape)
        self.w_y = Function(name='w_y', dimensions=wdims, shape=wshape)
        self.w_z = Function(name='w_z', dimensions=wdims, shape=wshape)

        # Oh god this is an abomination
        self.w_x.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int'),
                      np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int'),
                      np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int'),
                      :] = np.vstack(self._modified_nodes['x_coeffs'].values)

        self.w_y.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int'),
                      np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int'),
                      np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int'),
                      :] = np.vstack(self._modified_nodes['y_coeffs'].values)

        self.w_z.data[np.round_(self._modified_nodes['x']/self._spacing[0]).astype('int'),
                      np.round_(self._modified_nodes['y']/self._spacing[1]).astype('int'),
                      np.round_(self._modified_nodes['z']/self._spacing[2]).astype('int'),
                      :] = np.vstack(self._modified_nodes['z_coeffs'].values)
