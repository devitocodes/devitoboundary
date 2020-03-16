"""
Data structures to be used in the implementation of immersed boundaries of any
sort.
"""

import numpy as np

from scipy.spatial import Delaunay, cKDTree

__all__ = ['PolyMesh']


class PolyMesh():
    """
    A polygonal mesh of points. Used to express a non-concave surface
    in 3D space.

    Parameters
    ----------
    data : array_like
        Array of surface coordinates grouped as [[x, y, z], [x, y, z], ...]
    grid : Devito Grid object
        The grid against which the boundary surface is to be defined

    Attributes
    ----------
    points : ndarray
        The points which make up the vertices of the polygonal surface
    simplices : ndarray
        The indicies of the simplices of each polygon
    neighbors : ndarray
        The indices of neighboring polygons of a particular polygon
    equations : ndarray
        The coefficients of the plane equation
    values : ndarray
        The constant of the plane equation
    """

    def __init__(self, data, grid):
        assert len(grid.dimensions) == 3, "PolyMesh is for 3D grids only"
        assert len(np.shape(data)) == 2, "Coordinates for each point should be in the form [x, y, z]"

        self._points = np.array(data, dtype=np.float32)
        # Initialise properties
        self._core_properties()
        # Characterize planes
        self._characterise_planes()
        # Set up K-d tree
        self._tree_setup()

    @property
    def points(self):
        """Datapoints which make up the topography point cloud"""
        return self._points

    @property
    def simplices(self):
        """Indices of the simplices of each polygon"""
        return self._simplices

    @property
    def neighbors(self):
        """The indices of neighboring polygons of each polygon"""
        return self._neighbors

    @property
    def polycount(self):
        """Number of polgons contained within mesh"""
        return self._polycount

    @property
    def equations(self):
        """The coefficents of the plane equation"""
        return self._equations

    @property
    def values(self):
        """The constant of the plane equation"""
        return self._values

    def _core_properties(self):
        """Set up all the core properties of the mesh"""
        mesh = Delaunay(self._points[:, :2])
        self._simplices = mesh.simplices
        self._neighbors = mesh.neighbors
        self._polycount = len(self._simplices)

    def _characterise_planes(self):
        """
        Find the equations for each plane.
        """
        plane_vector_1 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 1]
        plane_vector_2 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 2]
        # Equations of the plane
        self._equations = np.cross(plane_vector_1, plane_vector_2)
        self._values = np.sum(self.equations*self.points[self.simplices][:, 0], axis=1)

    def _tree_setup(self):
        """
        Initialise the K-d tree used to find nearest polygon to a node. This is
        used to accelerate node-finding.
        """
        self._tree = cKDTree(self._points)

    def query_points(self, x, k, distance_upper_bound=np.inf):
        """
        Query the tree of surface points.

        Parameters
        ----------
        x : array_like
            An array of points to query. Grouped as [[x, y, z], [x, y, z], ...]
        k : non-negative int
            Nearest k points to find
        distance_upper_bound : non-negative float
            Return only surface points within this radius

        Returns
        -------
        d : ndarray
            Distances to nearest neighbors
        i : ndarray
            Integers of nearest neighbors in 'points'
        """

        return self._tree.query(x, k, eps=np.finfo(np.float32).eps,
                                distance_upper_bound=distance_upper_bound,
                                n_jobs=-1)

    def query_polygons(self, x, k, distance_upper_bound=np.inf):
        """
        Query the tree of surface points and find attatched polygons.

        Parameters
        ----------
        x : array_like
            An array of points to query. Grouped as [[x, y, z], [x, y, z], ...]
        k : non-negative int
            Nearest k points to find
        distance_upper_bound : non-negative float
            Return only surface points within this radius

        Returns
        -------
        i : ndarray
            Indices of attatched polygons in 'simplices'
        """

        _, indices = self.query_points(x, k, distance_upper_bound)
        return indices
        # Might need a python-level loop here
