"""
Data structures to be used in the implementation of immersed boundaries. Includes
structures to express the boundary and its spatial relationships with the grid.
"""

import numpy as np

from scipy.spatial import Delaunay

__all__ = ['PolyMesh', 'PolyTree']


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

    @property
    def points(self):
        """Datapoints which make up the topogrpahy point cloud"""
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
        Find the equations in for each plane. Used to determine which half space
        of a plane a point is located in.
        """
        plane_vector_1 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 1]
        plane_vector_2 = self._points[self._simplices][:, 0] \
            - self._points[self._simplices][:, 2]
        # Equations of the plane
        self._equations = np.cross(plane_vector_1, plane_vector_2)
        self._values = np.sum(self.equations*self.points[self.simplices][:, 0], axis=1)

    def half_space(self, ref, indices):
        """
        Determine the half space inhabited by the polygons referred to by the
        indices given.

        Parameters
        ----------
        ref : int
            The index of the polygon along whose plane the polygon set is to be
            split
        indices : ndarray
            The indices of all the polygons in the set

        Returns
        -------
        spaces : ndarray
            The half space containing each point within each polygon. '1' refers
            to the positive half space, '-1' to the negative, and 0 signifies a
            point is located on the dividing plane.
        """
        spaces = np.sum(self._equations[ref]
                        * self._points[self._simplices[indices]], axis=2)
        spaces = np.sign(spaces - self._values[ref])  # Reduce to signs
        return spaces


class PolyNode():
    """
    A node of the binary spatial partition tree used to determine the spatial
    relationships between polygons and grid points.

    Parameters
    ----------
    polygon : int
        The index of the polygon located at the node
    """

    def __init__(self, polygon):
        # Index of the polygon in the BoundaryMesh object
        self.poly = polygon

        # Left (positive) and right (negative) branches
        self.pos = None
        self.neg = None


class PolyTree():
    """
    A binary spatial partition tree used to determine the spatial relationships
    between polygons by recursive subdivision.

    Parameters
    ----------
    data : array_like
        Array of surface coordinates grouped as [[x, y, z], [x, y, z], ...]
    grid : Devito Grid object
        The grid against which the boundary surface is to be defined
    """

    def __init__(self, data, grid):
        mesh = PolyMesh(data, grid)
        # Pick random polygon as tree root for balancing purposes
        self.root = PolyNode(np.random.randint(mesh.polycount-1))

        # Testing stuff
        polyset = list(range(mesh.polycount))
        del polyset[self.root.poly]
        half_spaces = mesh.half_space(self.root.poly, polyset)
        print(half_spaces)
