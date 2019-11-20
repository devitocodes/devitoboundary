"""
A module for implementation of topography in Devito via use of
the immersed boundary method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from itertools import combinations
from sympy import finite_diff_weights

__all__ = ['Boundary']


class Boundary():
    """
    An object to contain data relevant for implementing the
    immersed boundary method on a given domain.
    """

    def __init__(self, grid, boundary_data,
                 method_order=4):

        self._method_order = method_order

        # Derive useful properties from grid
        self._shape = np.asarray(grid.shape)
        self._extent = np.asarray(grid.extent)
        self._spacing = grid.spacing

        self._read_in(boundary_data)

        self._node_id()

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
        vector_3 = vertex_1 - vertex_3
        plane_grad = np.cross(vector_1, vector_2)
        plane_const = -np.sum(plane_grad*vertex_1, axis=1)

        # Return plane equation, vertices, and vectors of boundaries
        return vertex_1, vertex_2, vertex_3, \
        vector_1, vector_2, vector_3, \
        plane_grad, plane_const


    def _construct_loci(self, vertices):
        """
        Constructs a locus on the inner side of the plane. The thickness
        of the locus is equal to M/2 grid spacings where M is the order
        of the FD method.
        """

        vertex_1, vertex_2, vertex_3, \
        vector_1, vector_2, vector_3, \
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
                              'z':block_z.flatten()})

        # Remove all above boundary
        def above_bool(node, gradient, constant, vert_1, vert_2, vert_3):
            """
            Returns True if a point is located above the boundary. Returns
            False otherwise.
            """
            # Points are above
            loc_above = (node['x']*gradient[:, 0]
                         + node['y']*gradient[:, 1]
                         + node['z']*gradient[:, 2]
                         + constant[:] >= 0)
           
            
            verts = (vert_1, vert_2, vert_3)
            for v in combinations(range(3), 2):
                v_3 = np.setdiff1d(range(3), v)[0]

                # Points are within boundaries of triangle

                # Use x = my + c instead (for handling edges of type x = c)
                xy_flip = ((verts[v[0]][:, 0] - verts[v[1]][:, 0]) == 0)
                clean_dx = (verts[v[0]][:, 0] - verts[v[1]][:, 0]) # Zero-free denominator
                clean_dx[xy_flip] = 1 # Prevents undefined behaviour
                clean_dy = (verts[v[0]][:, 1] - verts[v[1]][:, 1])
                clean_dy[np.logical_not(xy_flip)] = 1

                # For edges of type y = mx + c
                m_x = (verts[v[0]][:, 1] - verts[v[1]][:, 1])/clean_dx
                c_x = verts[v[0]][:, 1] - m_x*verts[v[0]][:, 0]
                # For edges of type x = c
                m_y = (verts[v[0]][:, 0] - verts[v[1]][:, 0])/clean_dy
                c_y = verts[v[0]][:, 0] - m_y*verts[v[0]][:, 1]

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
                # Print number of trues (would expect there to be one)
                # (node can only be under one triangle unless it coincides with
                # surface point) <- Pick first index in this case (choice is arbitrary)
                # Print index of true (for tying to triangle for eta calculation)
                return True
            else:
                return False


        #print(above_bool(block.iloc[5], plane_grad, plane_const, vertex_1, vertex_2, vertex_3))
        #above_bool(block.iloc[5], plane_grad, plane_const, vertex_1, vertex_2, vertex_3)
        print(np.count_nonzero(block[block.apply(above_bool, axis=1, args=(plane_grad, plane_const, vertex_1, vertex_2, vertex_3))]))
        print(np.shape(self._mesh)[0])

        # Cut all points in z locus into new DataFrame
        # Cut all points in y locus into new DataFrame
        # Cut all points in x locus into new DataFrame


    def _node_id(self):
        """
        Generates a list of nodes where stencils will require modification
        in either x or y directions.
        """

        self._generate_triangles()

        self._modified_nodes = pd.DataFrame(columns=['x', 'y', 'z'])

        self._construct_loci(self._mesh)

    # def _plot_nodes(self)
